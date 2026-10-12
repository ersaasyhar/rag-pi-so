"""Export ground-truth navigation trajectories from legal trees and benchmark dataset.

Generates training samples for fine-tuning vllm-sr/d3-edge on Indonesian legal tree navigation:
- Root -> BAB navigation decisions
- BAB -> Pasal navigation decisions
"""

from __future__ import annotations

import argparse
import json
import logging
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
            b["node_id"]: f"{b['title']}. {b.get('summary', '')}".strip()
            for b in structure[:50]
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


def export_dataset(
    benchmark_path: Path = BENCHMARK_FILE,
    output_path: Path = DEFAULT_OUTPUT,
) -> int:
    """Read benchmark items, extract trajectories, and write JSONL dataset."""
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    all_samples: list[dict[str, Any]] = []
    with open(benchmark_path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            trajectories = extract_trajectories_for_item(item)
            all_samples.extend(trajectories)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        for sample in all_samples:
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info("Exported %d navigation samples to %s", len(all_samples), output_path)
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
    args = parser.parse_args()

    count = export_dataset(args.benchmark, args.output)
    print(f"Successfully generated {count} navigation samples at {args.output}")


if __name__ == "__main__":
    main()
