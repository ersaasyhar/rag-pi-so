"""Regression and invariant tests for Indonesian legal RAG benchmark pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.evaluate import compute_dcg, compute_ndcg, evaluate_retrieval_query
from src.legal_parser import normalize_pasal_id, parse_legal_structure
from src.schema import RetrievedChunk
from src.sparse_bm25 import BM25Encoder

BASE_DIR = Path(__file__).resolve().parent.parent


def test_normalize_pasal_id():
    assert normalize_pasal_id("Pasal 1") == "Pasal 1"
    assert normalize_pasal_id("Pasal I") == "Pasal 1"
    assert normalize_pasal_id("Pasal 01") == "Pasal 1"
    assert normalize_pasal_id("Pasal 27") == "Pasal 27"
    assert normalize_pasal_id("Pasal XL") == "Pasal 40"


def test_sparse_bm25():
    docs = [
        "Undang-Undang Nomor 27 Tahun 2022 tentang Pelindungan Data Pribadi",
        "Kitab Undang-Undang Hukum Pidana tentang tindak pidana pencurian",
        "Perseroan Terbatas didirikan oleh dua orang atau lebih",
    ]
    encoder = BM25Encoder().fit(docs)
    assert len(encoder.vocab) > 10
    indices, values = encoder.encode("pencurian pidana", is_query=True)
    assert len(indices) == 2
    assert len(values) == 2


def test_ir_metrics_calculation():
    gold = [{"doc_id": "uu_1_2023", "pages": [166, 167], "pasal": "Pasal 476"}]
    # Perfect top-1 match
    retrieved = [
        RetrievedChunk(
            doc_id="uu_1_2023",
            pages=[166],
            pasal="Pasal 476",
            text="Pasal 476 Setiap Orang yang mengambil suatu Barang...",
            score=0.9,
        )
    ]
    metrics = evaluate_retrieval_query(gold, retrieved, top_k=5)
    assert metrics["page_hit_at_1"] == 1.0
    assert metrics["page_hit_at_5"] == 1.0
    assert metrics["page_hit"] == 1.0
    assert metrics["page_mrr"] == 1.0
    assert metrics["pasal_recall"] == 1.0
    assert metrics["page_recall"] == 0.5  # 1 of 2 gold pages hit

    # Hit at rank 2 (Top-1 miss, Top-5 hit)
    retrieved_rank2 = [
        RetrievedChunk(doc_id="uu_1_2023", pages=[999], pasal="Pasal 1", text="", score=0.9),
        RetrievedChunk(doc_id="uu_1_2023", pages=[166], pasal="Pasal 476", text="", score=0.8),
    ]
    metrics2 = evaluate_retrieval_query(gold, retrieved_rank2, top_k=5)
    assert metrics2["page_hit_at_1"] == 0.0
    assert metrics2["page_hit_at_5"] == 1.0
    assert metrics2["page_mrr"] == 0.5


def test_dcg_ndcg():
    assert compute_dcg([1, 0, 0]) == 1.0
    assert compute_ndcg([1, 0, 0], total_gold=1) == 1.0
    assert compute_ndcg([0, 1, 0], total_gold=1) < 1.0


def test_legal_parser_and_chunker_contract():
    dummy_doc = {
        "doc_id": "test_uu_1",
        "metadata": {"jenis_peraturan": "UU", "nomor": "1", "tahun": "2024", "judul": "Uji Coba"},
        "pages": [
            {
                "page_number": 1,
                "text": "BAB I\nKETENTUAN UMUM\nPasal 1\nDefinisi hukum dalam undang-undang ini.",
            },
            {"page_number": 2, "text": "BAB II\nASAS\nPasal 2\nAsas kepastian hukum dan keadilan."},
        ],
    }
    parsed = parse_legal_structure(dummy_doc)
    assert parsed["total_articles"] == 2
    assert "Pasal 1" in parsed["pasal_map"]
    assert "Pasal 2" in parsed["pasal_map"]
    assert parsed["pasal_map"]["Pasal 1"]["pages"] == [1]
    assert parsed["pasal_map"]["Pasal 2"]["pages"] == [2]


if __name__ == "__main__":
    test_normalize_pasal_id()
    test_sparse_bm25()
    test_ir_metrics_calculation()
    test_dcg_ndcg()
    test_legal_parser_and_chunker_contract()
    print("All pipeline invariant tests passed!")
