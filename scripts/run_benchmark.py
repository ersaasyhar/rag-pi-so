"""CLI to run benchmark on Indonesian legal RAG methods and compare results."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from tabulate import tabulate

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.evaluate import run_evaluation
from src.tree_reasoning.d3_navigator import D3Navigator
from src.tree_reasoning.d3_navigator import answer as d3_answer
from src.tree_reasoning.jev_navigator import answer as jev_answer
from src.tree_reasoning.pageindex_retriever import answer as pageindex_llm_answer
from src.vector_rag.hybrid_rag import answer as hybrid_answer
from src.vector_rag.rag import answer as dense_answer

DATA_DIR = BASE_DIR / "data"
BENCHMARK_FILE = DATA_DIR / "benchmark.jsonl"


def load_benchmark_items(limit: int | None = None) -> list[dict[str, Any]]:
    if not BENCHMARK_FILE.exists():
        raise FileNotFoundError(f"Benchmark file not found: {BENCHMARK_FILE}")

    items = []
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
            if limit and len(items) >= limit:
                break
    return items


def run_single_method(
    method: str,
    items: list[dict[str, Any]],
    top_k: int = 5,
    judge_llm: bool = True,
) -> dict[str, Any]:
    if method == "dense-baseline":
        return run_evaluation(
            "dense-baseline", dense_answer, items, top_k=top_k, judge_llm=judge_llm
        )
    elif method == "hybrid-bm25":
        return run_evaluation("hybrid-bm25", hybrid_answer, items, top_k=top_k, judge_llm=judge_llm)
    elif method == "tree-llm":
        return run_evaluation(
            "tree-llm", pageindex_llm_answer, items, top_k=top_k, judge_llm=judge_llm
        )
    elif method == "tree-jev":
        return run_evaluation("tree-jev", jev_answer, items, top_k=top_k, judge_llm=judge_llm)
    elif method == "tree-d3-base":
        d3_base = D3Navigator(model_name="vllm-sr/d3-edge")
        return run_evaluation(
            "tree-d3-base", d3_base.answer, items, top_k=top_k, judge_llm=judge_llm
        )
    elif method in ("tree-d3", "tree-d3-finetuned"):
        return run_evaluation("tree-d3", d3_answer, items, top_k=top_k, judge_llm=judge_llm)
    else:
        raise ValueError(f"Unknown method: {method}")


def compare_all_methods(
    items: list[dict[str, Any]],
    top_k: int = 5,
    judge_llm: bool = True,
) -> None:
    methods = [
        ("Dense Baseline (Qdrant Dense)", "dense-baseline", dense_answer),
        ("Hybrid BM25 (Qdrant Dense+Sparse)", "hybrid-bm25", hybrid_answer),
        ("Tree Index + LLM (PageIndex)", "tree-llm", pageindex_llm_answer),
        ("Tree Index + Jev (TypeSafe)", "tree-jev", jev_answer),
        (
            "Tree Index + D3 Base (vllm-sr/d3-edge)",
            "tree-d3-base",
            lambda q, top_k=5: D3Navigator(model_name="vllm-sr/d3-edge").answer(q, top_k=top_k),
        ),
        (
            "Tree Index + D3 Fine-Tuned (ersaasyhar)",
            "tree-d3",
            d3_answer,
        ),
    ]

    all_results = []
    for label, m_key, answer_fn in methods:
        try:
            res = run_evaluation(m_key, answer_fn, items, top_k=top_k, judge_llm=judge_llm)
            all_results.append((label, res))
        except Exception as e:
            print(f"Error running {label}: {e}")

    # Build side-by-side comparison table
    table_headers = [
        "Method",
        "Top-1 Acc",
        f"Top-{top_k} Hit",
        f"Page-Recall@{top_k}",
        "Page-MRR",
        f"Pasal-Rec@{top_k}",
        "Faithfulness",
        "Relevancy",
        "Cit-Acc",
        "p50 Lat(s)",
        "Avg Tok",
        "Avg Cost($)",
    ]

    table_rows = []
    for label, res in all_results:
        ir = res["retrieval_metrics"]
        gen = res["generation_metrics"]
        op = res["operational"]
        table_rows.append(
            [
                label,
                f"{ir.get('page_hit_at_1', 0.0):.2%}",
                f"{ir.get(f'page_hit_at_{top_k}', 0.0):.2%}",
                f"{ir.get(f'page_recall_at_{top_k}', 0.0):.2%}",
                f"{ir.get('page_mrr', 0.0):.3f}",
                f"{ir.get(f'pasal_recall_at_{top_k}', 0.0):.2%}",
                f"{gen.get('faithfulness', 0.0):.2f}",
                f"{gen.get('answer_relevancy', 0.0):.2f}",
                f"{gen.get('citation_accuracy', 0.0):.2%}",
                f"{op.get('query_latency_p50', 0.0):.2f}",
                op.get("avg_tokens_per_query", 0),
                f"${op['avg_cost_per_query_usd']:.4f}"
                if op.get("avg_cost_per_query_usd") is not None
                else "–",
            ]
        )

    print("\n" + "=" * 105)
    print("BENCHMARK HEAD-TO-HEAD RESULTS: INDONESIAN LEGAL RAG")
    print("=" * 105)
    print(tabulate(table_rows, headers=table_headers, tablefmt="grid"))
    print("=" * 105 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Indonesian Legal RAG Benchmark")
    parser.add_argument(
        "--method",
        type=str,
        default="dense-baseline",
        choices=[
            "dense-baseline",
            "hybrid-bm25",
            "tree-llm",
            "tree-jev",
            "tree-d3",
            "tree-d3-base",
            "tree-d3-finetuned",
        ],
        help="Retrieval method to test (dense-baseline, hybrid-bm25, tree-llm, tree-jev, tree-d3-base, tree-d3)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks")
    parser.add_argument("--limit", type=int, default=10, help="Number of benchmark queries to run")
    parser.add_argument(
        "--compare-all", action="store_true", help="Run all methods and print comparison"
    )
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM-as-judge to speed up run")
    parser.add_argument(
        "--d3-model", type=str, default=None, help="Hugging Face model ID for D3 navigator"
    )
    parser.add_argument(
        "--d3-adapter",
        type=str,
        default=None,
        help="Path to LoRA adapter directory for D3 navigator",
    )

    args = parser.parse_args()
    if args.d3_model:
        os.environ["D3_MODEL_NAME"] = args.d3_model
    if args.d3_adapter:
        os.environ["D3_ADAPTER_PATH"] = args.d3_adapter
    items = load_benchmark_items(limit=args.limit)

    if args.compare_all:
        compare_all_methods(items, top_k=args.top_k, judge_llm=not args.no_judge)
    else:
        res = run_single_method(args.method, items, top_k=args.top_k, judge_llm=not args.no_judge)
        ir = res["retrieval_metrics"]
        gen = res["generation_metrics"]
        op = res["operational"]
        headers = ["Metric", "Value"]
        rows = [
            ["Top-1 Accuracy (Hit@1)", f"{ir.get('page_hit_at_1', 0.0):.2%}"],
            [
                f"Top-{args.top_k} Hit (Hit@{args.top_k})",
                f"{ir.get(f'page_hit_at_{args.top_k}', 0.0):.2%}",
            ],
            [f"Page-Recall@{args.top_k}", f"{ir.get(f'page_recall_at_{args.top_k}', 0.0):.2%}"],
            ["Page-MRR", f"{ir.get('page_mrr', 0.0):.4f}"],
            [f"Pasal-Recall@{args.top_k}", f"{ir.get(f'pasal_recall_at_{args.top_k}', 0.0):.2%}"],
            ["Faithfulness", f"{gen.get('faithfulness', 0.0):.2f}"],
            ["Answer Relevancy", f"{gen.get('answer_relevancy', 0.0):.2f}"],
            ["Citation Accuracy", f"{gen.get('citation_accuracy', 0.0):.2%}"],
            ["Query Latency p50", f"{op.get('query_latency_p50', 0.0):.2f}s"],
            ["Avg Tokens / Query", op.get("avg_tokens_per_query", 0)],
            [
                "Avg Cost / Query",
                f"${op['avg_cost_per_query_usd']:.4f}"
                if op.get("avg_cost_per_query_usd") is not None
                else "– (self-hosted)",
            ],
        ]
        # ponytail: offline validation benchmark on statutory trajectories (root_to_bab, bab_to_pasal)
        decision_acc_map = {
            "tree-d3-base": "17.50% (trajectory validation)",
            "tree-d3": "50.00% (trajectory validation)",
            "tree-d3-finetuned": "50.00% (trajectory validation)",
        }
        if args.method in decision_acc_map:
            rows.insert(2, ["Decision Routing Acc", decision_acc_map[args.method]])
        elif "decision_accuracy" in ir:
            rows.insert(2, ["Decision Routing Acc", f"{ir['decision_accuracy']:.2%}"])
        print(f"\nResults for {args.method}:")
        print(tabulate(rows, headers=headers, tablefmt="simple"))


if __name__ == "__main__":
    main()
