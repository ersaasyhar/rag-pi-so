"""Interactive CLI chat assistant for Indonesian legal regulations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.schema import RAGResponse
from src.tree_reasoning.d3_navigator import answer as d3_answer
from src.tree_reasoning.jev_navigator import answer as jev_answer
from src.tree_reasoning.pageindex_retriever import answer as pageindex_llm_answer
from src.vector_rag.hybrid_rag import answer as hybrid_answer
from src.vector_rag.rag import answer as dense_answer


def get_answer_fn(method: str):
    if method == "dense-baseline":
        return dense_answer
    elif method == "hybrid-bm25":
        return hybrid_answer
    elif method == "tree-llm":
        return pageindex_llm_answer
    elif method == "tree-jev":
        return jev_answer
    elif method == "tree-d3":
        return d3_answer
    else:
        raise ValueError(f"Unknown method: {method}")


def format_chat_output(resp: RAGResponse, method: str) -> str:
    lines = [
        "\n📖 Jawaban:",
        resp.answer,
        "\n📑 Sumber:",
    ]
    seen_sources = set()
    for c in resp.citations:
        key = (c.doc_id, c.pasal, c.page_number)
        if key in seen_sources:
            continue
        seen_sources.add(key)
        lines.append(
            f"- {c.doc_id.upper()}, {c.pasal} (hal. {c.page_number}) [relevance: {c.relevance_score:.2f}]"
        )

    lines.append(f"\n🔍 Method: {method}")
    lines.append(
        f"⏱️ Latency: {resp.usage.latency_seconds:.2f}s | Tokens: {resp.usage.total_tokens}"
    )
    return "\n".join(lines)


def run_repl(method: str, top_k: int = 5) -> None:
    answer_fn = get_answer_fn(method)
    print("=" * 60)
    print(f"🏛️ Asisten Hukum Indonesia (Backend: {method})")
    print("Ketik pertanyaan Anda (ketik 'exit' atau 'keluar' untuk berhenti):")
    print("=" * 60)

    while True:
        try:
            query = input("\n⚖️ Tanya: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "keluar", "q"):
                print("Terima kasih.")
                break

            resp = answer_fn(query, top_k=top_k)
            print(format_chat_output(resp, method))

        except (KeyboardInterrupt, EOFError):
            print("\nTerima kasih.")
            break
        except Exception as e:
            print(f"\nTerjadi kesalahan: {e}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Indonesian Legal RAG Chat Assistant")
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
        ],
        help="Retrieval backend to use (dense-baseline, hybrid-bm25, tree-llm, tree-jev, tree-d3)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks")
    parser.add_argument("--query", type=str, default="", help="Single-shot query (non-interactive)")

    args = parser.parse_args()

    if args.query:
        answer_fn = get_answer_fn(args.method)
        resp = answer_fn(args.query, top_k=args.top_k)
        print(format_chat_output(resp, args.method))
    else:
        run_repl(args.method, top_k=args.top_k)


if __name__ == "__main__":
    main()
