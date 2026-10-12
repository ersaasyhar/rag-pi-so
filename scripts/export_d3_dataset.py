"""Export ground-truth navigation trajectories from legal trees and benchmark dataset.

Generates training samples for fine-tuning vllm-sr/d3-edge on Indonesian legal tree navigation:
- Root -> BAB navigation decisions
- BAB -> Pasal navigation decisions
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
DATA_DIR = BASE_DIR / "data"
BENCHMARK_FILE = DATA_DIR / "benchmark.jsonl"
PARSED_DIR = DATA_DIR / "parsed"
DEFAULT_OUTPUT = DATA_DIR / "d3_navigation_train.jsonl"
from src.tree_reasoning.tree_index import build_tree_for_doc


def extract_trajectories_for_item(
    item: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract root->bab and bab->pasal decision samples for one benchmark item."""
    query = item.get("question", "").strip()
    gold_pages_list = item.get("gold_pages", [])
    if not query or not gold_pages_list:
        return []

    samples: list[dict[str, Any]] = []

    for gp in gold_pages_list:
        doc_id = gp.get("doc_id")
        gold_pasal = (gp.get("pasal") or "").strip().lower()
        gold_pages = set(gp.get("pages", []))
        if not doc_id:
            continue

        try:
            tree = build_tree_for_doc(doc_id)
        except Exception as e:
            logger.warning("Failed to build tree for %s: %s", doc_id, e)
            continue

        structure = [n for n in tree.get("structure", []) if n.get("nodes")]
        if not structure:
            continue

        doc_title = tree.get("title", doc_id)

        target_bab = None
        target_pasal = None

        for bab_node in structure:
            for p in bab_node.get("nodes", []):
                p_title = p.get("title", "").strip().lower()
                p_pages = set(range(p.get("start_index", 0), p.get("end_index", 0) + 1))
                if (gold_pasal and gold_pasal == p_title) or (
                    gold_pages and gold_pages.intersection(p_pages)
                ):
                    target_bab = bab_node
                    target_pasal = p
                    break
            if target_pasal is not None:
                break

        if not target_bab or not target_pasal:
            continue

        # Step 1: Root -> BAB choice
        bab_criteria = {
            b["node_id"]: f"{b['title']}. {b.get('summary', '')}".strip() for b in structure[:50]
        }
        samples.append(
            {
                "query": query,
                "state": {"query": query, "current_section": doc_title},
                "question_type": "choice",
                "instructions": "Pilih bagian dokumen yang paling mungkin memuat jawaban pertanyaan.",
                "criteria": bab_criteria,
                "target_choice": target_bab["node_id"],
                "doc_id": doc_id,
                "level": "root_to_bab",
                "benchmark_id": item.get("id"),
            }
        )

        # Step 2: BAB -> Pasal choice
        child_pasals = target_bab.get("nodes", [])
        if child_pasals:
            pasal_criteria = {
                p["node_id"]: f"{p['title']}. {p.get('summary', '')}".strip()
                for p in child_pasals[:50]
            }
            samples.append(
                {
                    "query": query,
                    "state": {"query": query, "current_section": target_bab.get("title", "")},
                    "question_type": "choice",
                    "instructions": "Pilih bagian dokumen yang paling mungkin memuat jawaban pertanyaan.",
                    "criteria": pasal_criteria,
                    "target_choice": target_pasal["node_id"],
                    "doc_id": doc_id,
                    "level": "bab_to_pasal",
                    "benchmark_id": item.get("id"),
                }
            )

    return samples


def clean_clause_text(raw: str) -> str:
    """Clean OCR headers, line numbers, and formatting from statutory text."""
    lines = raw.split("\n")
    cleaned: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.search(
            r"^(PRESIDEN|REPUBLIK INDONESIA|SALINAN|Menimbang|Mengingat|SK No|- \d+ -|\d+ -)",
            line,
            re.I,
        ):
            continue
        cleaned.append(line)
    text = " ".join(cleaned)
    text = re.sub(r"^(Pasal\s+\d+[A-Za-z]?\s*)", "", text, flags=re.I)
    text = re.sub(r"^\s*\([0-9IlL]+\)\s*", "", text)
    text = re.sub(r"\s*\([0-9IlL]+\)\s*", " ", text)
    text = re.sub(r"\s+[a-z]\.\s+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_first_clean_sentence(raw: str) -> str:
    """Extract the first substantive sentence of a legal clause."""
    cleaned = clean_clause_text(raw)
    for part in cleaned.split("."):
        part = part.strip()
        if len(part) > 25:
            return re.sub(r"\s+sebagaimana\s+dimaksud.*$", "", part, flags=re.I)
    return cleaned[:120]


def generate_natural_variations(
    doc_title: str,
    jenis: str,
    nomor: str,
    tahun: str,
    pasal_name: str,
    bab_title: str,
    text: str,
    max_variations: int = 4,
) -> list[str]:
    """Generate realistic, diverse conversational legal queries for an article."""
    sent = extract_first_clean_sentence(text)
    if not sent or len(sent) < 20:
        return []

    variations: set[str] = set()
    ref_short = f"{jenis} {nomor}/{tahun}" if nomor and tahun else doc_title

    # 1. Ketentuan Umum / Defined terms (Pasal 1)
    if "pasal 1" in pasal_name.lower() or "ketentuan umum" in bab_title.lower():
        terms = re.findall(r"\b\d+\.\s+([A-Za-z\s]+?)\s+adalah\b", text)
        for t in terms[:3]:
            t_clean = " ".join(t.split())
            if len(t_clean) > 3:
                variations.add(f"Apa yang dimaksud dengan {t_clean} menurut {ref_short}?")
                variations.add(f"Bagaimana definisi {t_clean} dalam ketentuan peraturan ini?")

    # 2. Liability / Responsibility (tidak bertanggung jawab / bertanggung jawab)
    m_resp = re.search(
        r"^(.*?)\s+tidak\s+bertanggung\s+jawab\s+(?:secara pribadi\s+)?atas\s+(.*)", sent, re.I
    )
    if m_resp:
        subj = m_resp.group(1).strip()
        obj = m_resp.group(2).strip()
        variations.add(f"Apakah {subj} bertanggung jawab atas {obj[:70]}?")
        variations.add(f"Bagaimana batas tanggung jawab hukum {subj} menurut {ref_short}?")

    m_resp_pos = re.search(r"^(.*?)\s+bertanggung\s+jawab\s+atas\s+(.*)", sent, re.I)
    if m_resp_pos and not m_resp:
        subj = m_resp_pos.group(1).strip()
        obj = m_resp_pos.group(2).strip()
        variations.add(f"Siapa yang bertanggung jawab atas {obj[:70]}?")
        variations.add(f"Bagaimana tanggung jawab hukum {subj} terkait {obj[:60]}?")

    # 3. Establishment / Formation (didirikan oleh)
    m_dirikan = re.search(r"^(.*?)\s+didirikan\s+oleh\s+(.*)", sent, re.I)
    if m_dirikan:
        subj = m_dirikan.group(1).strip()
        variations.add(f"Berapa jumlah minimal pendiri dan apa syarat pendirian {subj}?")
        variations.add(f"Bagaimana ketentuan pendirian {subj} menurut {ref_short}?")

    # 4. Deadlines (paling lambat)
    m_deadline = re.search(r"paling\s+lambat\s+([^,;]+)", sent, re.I)
    if m_deadline:
        variations.add(f"Berapa batas waktu paling lambat terkait {sent[:70]}?")
        variations.add(f"Kapan tenggat waktu pelaksanaan ketentuan {sent[:60]}?")

    # 5. Criminal Sanctions (dipidana)
    m_pid = re.search(
        r"dipidana\s+(?:dengan\s+pidana\s+penjara\s+paling\s+lama\s+([^,]+)|karena\s+([^,]+))",
        sent,
        re.I,
    )
    m_actor = re.search(r"Setiap\s+Orang\s+yang\s+([^,;]+)", sent, re.I)
    if m_pid and m_actor:
        act = m_actor.group(1).strip()
        variations.add(f"Berapa ancaman sanksi pidana bagi orang yang {act[:70]}?")
        variations.add(f"Apakah perbuatan {act[:65]} dapat dipidana penjara menurut {ref_short}?")

    # 6. Prohibitions (dilarang)
    m_lar = re.search(r"(?:dilarang|tidak\s+boleh)\s+([^,;]+)", sent, re.I)
    if m_lar:
        act = m_lar.group(1).strip()
        variations.add(f"Apakah diperbolehkan untuk {act[:70]}?")
        variations.add(f"Apa saja larangan yang diatur mengenai {act[:65]}?")

    # 7. Rights (berhak)
    m_hak = re.search(r"^(.*?)\s+berhak\s+([^,;]+)", sent, re.I)
    if m_hak:
        subj = m_hak.group(1).strip()
        right = m_hak.group(2).strip()
        variations.add(f"Apakah {subj} berhak {right[:70]}?")
        variations.add(f"Apa saja hak yang dimiliki oleh {subj} dalam ketentuan ini?")

    # 8. Obligations (wajib)
    m_waj = re.search(r"^(.*?)\s+wajib\s+([^,;]+)", sent, re.I)
    if m_waj:
        subj = m_waj.group(1).strip()
        duty = m_waj.group(2).strip()
        variations.add(f"Apa kewajiban hukum {subj} terkait {duty[:70]}?")
        variations.add(f"Kapan {subj} diwajibkan untuk {duty[:65]}?")

    # 9. Natural topic fallback if under threshold
    clean_words = [
        w
        for w in re.findall(r"[A-Za-z\-]{4,}", sent)
        if w.lower()
        not in {
            "pasal",
            "ayat",
            "dalam",
            "untuk",
            "yang",
            "pada",
            "atau",
            "serta",
            "dapat",
            "oleh",
            "sebagaimana",
            "dimaksud",
            "ketentuan",
            "lebih",
            "lanjut",
            "aturan",
            "setiap",
            "orang",
        }
    ]
    topic = " ".join(clean_words[:5])
    if topic and len(variations) < max_variations:
        variations.add(f"Bagaimana ketentuan mengenai {topic} menurut aturan hukum?")
        variations.add(f"Dalam {ref_short}, apa yang diatur terkait {topic}?")

    return sorted(list(variations))[:max_variations]


def slice_criteria_with_target(
    nodes: list[dict[str, Any]], target_key: str, max_items: int = 50
) -> dict[str, str]:
    """Build criteria dictionary ensuring target_key is always included."""
    if len(nodes) <= max_items:
        selected = nodes
    else:
        target_idx = 0
        for i, node in enumerate(nodes):
            if node["node_id"] == target_key:
                target_idx = i
                break
        start = max(0, min(target_idx - (max_items // 2), len(nodes) - max_items))
        selected = nodes[start : start + max_items]

    return {n["node_id"]: f"{n['title']}. {n.get('summary', '')}".strip() for n in selected}


def extract_synthetic_trajectories_for_doc(
    parsed_path: Path, max_variations_per_pasal: int = 4
) -> list[dict[str, Any]]:
    """Extract natural synthetic trajectories across all chapters and articles of a document."""
    with open(parsed_path, "r", encoding="utf-8") as fp:
        doc_data = json.load(fp)

    doc_id = doc_data.get("doc_id", "")
    if not doc_id:
        return []

    meta = doc_data.get("metadata", {})
    jenis = meta.get("jenis_peraturan", "UU")
    nomor = meta.get("nomor", "")
    tahun = meta.get("tahun", "")
    judul = meta.get("judul", doc_id)
    pasal_map = doc_data.get("pasal_map", {})

    try:
        tree = build_tree_for_doc(doc_id)
    except Exception as e:
        logger.warning("Failed to build tree for %s: %s", doc_id, e)
        return []

    structure = [n for n in tree.get("structure", []) if n.get("nodes")]
    if not structure:
        return []

    doc_title = tree.get("title", judul)
    samples: list[dict[str, Any]] = []

    for bab_node in structure:
        bab_title = bab_node.get("title", "")
        child_pasals = bab_node.get("nodes", [])
        if not child_pasals:
            continue

        for p_node in child_pasals:
            p_name = p_node.get("title", "").strip()
            p_info = pasal_map.get(p_name)
            p_text = p_info.get("text", "") if p_info else p_node.get("summary", "")
            if len(p_text) < 30:
                continue

            queries = generate_natural_variations(
                doc_title=doc_title,
                jenis=jenis,
                nomor=nomor,
                tahun=tahun,
                pasal_name=p_name,
                bab_title=bab_title,
                text=p_text,
                max_variations=max_variations_per_pasal,
            )

            for q_idx, query in enumerate(queries):
                synth_id = f"synth_{doc_id}_{p_node['node_id']}_{q_idx}"

                # Step 1: Root -> BAB choice
                bab_criteria = slice_criteria_with_target(
                    structure, bab_node["node_id"], max_items=50
                )
                samples.append(
                    {
                        "query": query,
                        "state": {"query": query, "current_section": doc_title},
                        "question_type": "choice",
                        "instructions": "Pilih bagian dokumen yang paling mungkin memuat jawaban pertanyaan.",
                        "criteria": bab_criteria,
                        "target_choice": bab_node["node_id"],
                        "doc_id": doc_id,
                        "level": "root_to_bab",
                        "benchmark_id": synth_id,
                    }
                )

                # Step 2: BAB -> Pasal choice
                pasal_criteria = slice_criteria_with_target(
                    child_pasals, p_node["node_id"], max_items=50
                )
                samples.append(
                    {
                        "query": query,
                        "state": {"query": query, "current_section": bab_title},
                        "question_type": "choice",
                        "instructions": "Pilih bagian dokumen yang paling mungkin memuat jawaban pertanyaan.",
                        "criteria": pasal_criteria,
                        "target_choice": p_node["node_id"],
                        "doc_id": doc_id,
                        "level": "bab_to_pasal",
                        "benchmark_id": synth_id,
                    }
                )

    return samples


def export_dataset(
    benchmark_path: Path = BENCHMARK_FILE,
    output_path: Path = DEFAULT_OUTPUT,
    include_benchmark: bool = True,
    include_synthetic: bool = False,
    max_variations_per_pasal: int = 4,
) -> int:
    """Read benchmark items, extract trajectories, and write JSONL dataset."""
    all_samples: list[dict[str, Any]] = []

    if include_benchmark:
        if benchmark_path.exists():
            with open(benchmark_path, "r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    trajectories = extract_trajectories_for_item(item)
                    all_samples.extend(trajectories)
            logger.info("Loaded %d benchmark trajectories", len(all_samples))
        else:
            logger.warning("Benchmark file not found: %s", benchmark_path)

    if include_synthetic:
        parsed_files = sorted(PARSED_DIR.glob("*.json"))
        parsed_files = [p for p in parsed_files if not p.name.startswith("test_")]
        logger.info("Generating synthetic trajectories from %d parsed files", len(parsed_files))
        synth_total = 0
        for pf in parsed_files:
            doc_samples = extract_synthetic_trajectories_for_doc(
                pf, max_variations_per_pasal=max_variations_per_pasal
            )
            all_samples.extend(doc_samples)
            synth_total += len(doc_samples)
        logger.info("Generated %d synthetic navigation trajectories", synth_total)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        for sample in all_samples:
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info("Total exported navigation samples: %d -> %s", len(all_samples), output_path)
    return len(all_samples)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export D3 Edge supervised navigation trajectories for fine-tuning."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=BENCHMARK_FILE,
        help="Path to benchmark.jsonl input file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to output JSONL trajectory dataset",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate comprehensive synthetic trajectories across all parsed regulations",
    )
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Do not include benchmark trajectories in the output",
    )
    parser.add_argument(
        "--max-variations",
        type=int,
        default=4,
        help="Maximum natural question variations per statutory article (default: 4)",
    )
    args = parser.parse_args()

    count = export_dataset(
        benchmark_path=args.benchmark,
        output_path=args.output,
        include_benchmark=not args.no_benchmark,
        include_synthetic=args.synthetic,
        max_variations_per_pasal=args.max_variations,
    )
    print(f"Successfully generated {count} navigation samples at {args.output}")


if __name__ == "__main__":
    main()
