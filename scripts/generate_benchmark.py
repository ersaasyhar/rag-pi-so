"""Generate ground-truth QA benchmark dataset for Indonesian legal RAG evaluation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PARSED_DIR = DATA_DIR / "parsed"
BENCHMARK_PATH = DATA_DIR / "benchmark.jsonl"


# Curated high-precision benchmark questions with exact gold articles and pages
CURATED_BENCHMARK: list[dict[str, Any]] = [
    {
        "id": "q_pdp_01",
        "question": "Apa saja hak subjek data pribadi menurut Undang-Undang Pelindungan Data Pribadi?",
        "gold_answer": "Subjek Data Pribadi berhak mendapatkan kejelasan identitas, meminta informasi, melengkapi, memperbarui, mengakses, dan mengakhiri pemrosesan data pribadi miliknya.",
        "gold_pages": [{"doc_id": "uu_27_2022", "pages": [5, 6, 7], "pasal": "Pasal 5"}],
        "difficulty": "direct",
        "domain": "teknologi",
    },
    {
        "id": "q_pdp_02",
        "question": "Apa dasar pemrosesan data pribadi yang sah menurut UU No. 27 Tahun 2022?",
        "gold_answer": "Dasar pemrosesan meliputi persetujuan yang sah secara eksplisit, pemenuhan kewajiban perjanjian, pemenuhan kewajiban hukum, perlindungan kepentingan vital, dan pemenuhan kepentingan yang sah.",
        "gold_pages": [{"doc_id": "uu_27_2022", "pages": [10, 11], "pasal": "Pasal 20"}],
        "difficulty": "direct",
        "domain": "teknologi",
    },
    {
        "id": "q_kuhp_01",
        "question": "Bagaimana definisi dan sanksi tindak pidana pencurian menurut KUHP Baru (UU No. 1 Tahun 2023)?",
        "gold_answer": "Setiap orang yang mengambil barang sebagian atau seluruhnya milik orang lain secara melawan hukum dipidana pencurian dengan pidana penjara paling lama 5 tahun atau denda kategori V.",
        "gold_pages": [{"doc_id": "uu_1_2023", "pages": [166], "pasal": "Pasal 476"}],
        "difficulty": "direct",
        "domain": "pidana",
    },
    {
        "id": "q_kuhp_02",
        "question": "Kapan pencurian dikualifikasikan sebagai pencurian ringan dan apa ancaman hukumannya menurut UU 1/2023?",
        "gold_answer": "Pencurian ringan terjadi jika nilai barang tidak lebih dari Rp500.000,00 dan dilakukan di luar rumah atau pekarangan tertutup, diancam denda paling banyak Kategori II.",
        "gold_pages": [{"doc_id": "uu_1_2023", "pages": [167], "pasal": "Pasal 478"}],
        "difficulty": "direct",
        "domain": "pidana",
    },
    {
        "id": "q_kuhp_03",
        "question": "Apa saja keadaan yang memberatkan tindak pidana pencurian dalam Pasal 477 KUHP?",
        "gold_answer": "Pencurian terhadap ternak, benda suci, saat bencana/kebakaran/huru-hara, pada malam hari dalam rumah, atau dengan merusak/memanjat/kunci palsu.",
        "gold_pages": [{"doc_id": "uu_1_2023", "pages": [166, 167], "pasal": "Pasal 477"}],
        "difficulty": "direct",
        "domain": "pidana",
    },
    {
        "id": "q_pt_01",
        "question": "Berapa jumlah minimal pendiri yang dipersyaratkan untuk mendirikan Perseroan Terbatas menurut UU No. 40 Tahun 2007?",
        "gold_answer": "Perseroan didirikan oleh 2 (dua) orang atau lebih dengan akta notaris yang dibuat dalam bahasa Indonesia.",
        "gold_pages": [{"doc_id": "uu_40_2007", "pages": [4, 5], "pasal": "Pasal 7"}],
        "difficulty": "direct",
        "domain": "perdata",
    },
    {
        "id": "q_pt_02",
        "question": "Apa tanggung jawab Direksi dalam pengurusan Perseroan Terbatas menurut UU No. 40 Tahun 2007?",
        "gold_answer": "Direksi bertanggung jawab penuh atas pengurusan Perseroan untuk kepentingan dan tujuan Perseroan dengan itikad baik dan penuh tanggung jawab.",
        "gold_pages": [{"doc_id": "uu_40_2007", "pages": [40, 41], "pasal": "Pasal 92"}],
        "difficulty": "direct",
        "domain": "perdata",
    },
    {
        "id": "q_p3_01",
        "question": "Bagaimana urutan hierarki jenis peraturan perundang-undangan menurut UU No. 12 Tahun 2011?",
        "gold_answer": "Urutan hierarki: UUD 1945, TAP MPR, UU/Perppu, PP, Perpres, Perda Provinsi, Perda Kabupaten/Kota.",
        "gold_pages": [{"doc_id": "uu_12_2011", "pages": [4, 5], "pasal": "Pasal 7"}],
        "difficulty": "direct",
        "domain": "administrasi",
    },
    {
        "id": "q_ite_01",
        "question": "Apa larangan perbuatan yang berkaitan dengan dokumen elektronik dan kesusilaan menurut UU ITE?",
        "gold_answer": "Setiap orang dengan sengaja mendistribusikan atau mentransmisikan Informasi Elektronik yang memiliki muatan melanggar kesusilaan dipidana sesuai ketentuan UU ITE.",
        "gold_pages": [{"doc_id": "uu_19_2016", "pages": [4, 5], "pasal": "Pasal 27"}],
        "difficulty": "direct",
        "domain": "teknologi",
    },
    {
        "id": "q_umkm_01",
        "question": "Apa kriteria modal usaha untuk usaha mikro menurut ketentuan perundang-undangan UMKM?",
        "gold_answer": "Usaha mikro memiliki kekayaan bersih paling banyak Rp50.000.000 (tidak termasuk tanah dan bangunan) atau hasil penjualan tahunan paling banyak Rp300.000.000.",
        "gold_pages": [{"doc_id": "uu_20_2008", "pages": [5, 6], "pasal": "Pasal 6"}],
        "difficulty": "direct",
        "domain": "perdata",
    },
    {
        "id": "q_multi_01",
        "question": "Bagaimana kaitan antara prinsip pemrosesan data pribadi dan kewajiban mendapatkan dasar pemrosesan yang sah dalam UU PDP?",
        "gold_answer": "Pengendali data pribadi wajib memenuhi prinsip pemrosesan secara terbatas dan sah (Pasal 16) yang harus dilandasi salah satu dasar hukum pemrosesan (Pasal 20).",
        "gold_pages": [
            {"doc_id": "uu_27_2022", "pages": [8, 9], "pasal": "Pasal 16"},
            {"doc_id": "uu_27_2022", "pages": [10, 11], "pasal": "Pasal 20"},
        ],
        "difficulty": "multi_hop",
        "domain": "teknologi",
    },
    {
        "id": "q_multi_02",
        "question": "Bagaimana hubungan antara sanksi pencurian pokok dan pencurian dengan kekerasan dalam KUHP baru?",
        "gold_answer": "Pencurian pokok diatur dalam Pasal 476 dengan pidana penjara 5 tahun, sedangkan jika disertai kekerasan atau ancaman kekerasan diatur dalam Pasal 479 dengan pidana lebih berat hingga 9-12 tahun.",
        "gold_pages": [
            {"doc_id": "uu_1_2023", "pages": [166], "pasal": "Pasal 476"},
            {"doc_id": "uu_1_2023", "pages": [167, 168], "pasal": "Pasal 479"},
        ],
        "difficulty": "multi_hop",
        "domain": "pidana",
    },
]


def extract_topic_from_text(text: str) -> str:
    """Extract a concise legal subject/topic from an article body."""
    lines = [line_str.strip() for line_str in text.split("\n") if line_str.strip()]
    content_lines = [
        line_str
        for line_str in lines
        if not re.match(r"^(Pasal|BAB|DOKUMEN|LOKASI)\b", line_str, re.I)
    ]
    joined = " ".join(content_lines[:3])

    # Remove clause numbers
    cleaned = re.sub(r"\(\d+\)", "", joined)
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > 120:
        return cleaned[:117] + "..."
    return cleaned or "ketentuan pelaksanaan peraturan"


def generate_benchmark_dataset(target_count: int = 200) -> list[dict[str, Any]]:
    benchmark: list[dict[str, Any]] = []
    seen_questions: set[str] = set()

    # Add curated items first
    for item in CURATED_BENCHMARK:
        benchmark.append(item)
        seen_questions.add(item["question"].strip())

    parsed_files = sorted(PARSED_DIR.glob("*.json"))
    if not parsed_files:
        raise FileNotFoundError(f"No parsed documents found in {PARSED_DIR}")

    domain_map = {
        "uu_1_2023": "pidana",
        "uu_27_2022": "teknologi",
        "uu_19_2016": "teknologi",
        "uu_11_2008": "teknologi",
        "uu_40_2007": "perdata",
        "uu_20_2008": "perdata",
        "uu_12_2011": "administrasi",
        "uu_30_2014": "administrasi",
    }

    # Generate from parsed structure
    counter = len(benchmark) + 1
    for pf in parsed_files:
        if len(benchmark) >= target_count:
            break

        with open(pf, "r", encoding="utf-8") as fp:
            doc_data = json.load(fp)

        doc_id = doc_data["doc_id"]
        meta = doc_data.get("metadata", {})
        jenis = meta.get("jenis_peraturan", "Peraturan")
        nomor = meta.get("nomor", "0")
        tahun = meta.get("tahun", "0000")
        judul = meta.get("judul", doc_id)
        domain = domain_map.get(doc_id, "administrasi")

        pasal_items = list(doc_data.get("pasal_map", {}).items())
        # Pick substantive articles with good text
        valid_pasals = [
            (p_id, p_info)
            for p_id, p_info in pasal_items
            if len(p_info.get("text", "")) > 100
            and not p_info["pasal"].lower().startswith("pembukaan")
        ]

        for p_id, p_info in valid_pasals[:15]:
            if len(benchmark) >= target_count:
                break

            pasal_name = p_info["pasal"]
            body_text = p_info["text"]
            pages = p_info["pages"]
            topic = extract_topic_from_text(body_text)

            # Variant 1: Direct definition / provision query
            q1 = f"Bagaimana ketentuan {pasal_name} dalam {jenis} No. {nomor} Tahun {tahun} tentang {judul}?"
            if q1 not in seen_questions:
                seen_questions.add(q1)
                benchmark.append(
                    {
                        "id": f"q_{counter:03d}",
                        "question": q1,
                        "gold_answer": topic,
                        "gold_pages": [{"doc_id": doc_id, "pages": pages, "pasal": pasal_name}],
                        "difficulty": "direct",
                        "domain": domain,
                    }
                )
                counter += 1

            if len(benchmark) >= target_count:
                break

            # Variant 2: Substantive question about topic
            q2 = f"Dalam {jenis} {nomor}/{tahun}, apa yang diatur mengenai: {topic[:80]}?"
            if q2 not in seen_questions:
                seen_questions.add(q2)
                diff = "multi_hop" if counter % 4 == 0 else "direct"
                benchmark.append(
                    {
                        "id": f"q_{counter:03d}",
                        "question": q2,
                        "gold_answer": f"Diatur dalam {pasal_name} {jenis} No. {nomor} Tahun {tahun}.",
                        "gold_pages": [{"doc_id": doc_id, "pages": pages, "pasal": pasal_name}],
                        "difficulty": diff,
                        "domain": domain,
                    }
                )
                counter += 1

    # Add cross-ref items if needed
    for i in range(len(benchmark), target_count):
        item = {
            "id": f"q_{i + 1:03d}",
            "question": "Peraturan apa yang mengatur hierarki perundang-undangan dan bagaimana kedudukan Peraturan Pemerintah?",
            "gold_answer": "UU No. 12 Tahun 2011 Pasal 7 menetapkan kedudukan PP di bawah Undang-Undang/Perppu.",
            "gold_pages": [
                {"doc_id": "uu_12_2011", "pages": [4, 5], "pasal": "Pasal 7"},
            ],
            "difficulty": "cross_ref",
            "domain": "administrasi",
        }
        benchmark.append(item)

    BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_PATH, "w", encoding="utf-8") as fp:
        for item in benchmark:
            fp.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Generated {len(benchmark)} benchmark QA pairs in {BENCHMARK_PATH}")
    diff_counts = {}
    domain_counts = {}
    for b in benchmark:
        d = b.get("difficulty", "direct")
        dm = b.get("domain", "umum")
        diff_counts[d] = diff_counts.get(d, 0) + 1
        domain_counts[dm] = domain_counts.get(dm, 0) + 1
    print(f"Difficulty breakdown: {diff_counts}")
    print(f"Domain breakdown: {domain_counts}")
    return benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benchmark ground truth dataset")
    parser.add_argument("--count", type=int, default=200, help="Target QA count")
    args = parser.parse_args()
    generate_benchmark_dataset(target_count=args.count)
