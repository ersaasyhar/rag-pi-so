# AGENTS.md

## Repository Overview

Two-paradigm benchmark comparing vector RAG (`dense-baseline`, `hybrid-bm25`) and tree index retrieval (`tree-llm`, `tree-jev`) on Indonesian legal documents (UU, PP, Perpres, Permen).

## Commands

- Install dependencies: `uv sync`
- Format & lint: `uv run ruff check .` / `uv run ruff format .`
- Download corpus: `uv run python scripts/download_corpus.py --limit 30`
- Run extraction: `uv run python -c "from src.extractor import extract_all; extract_all()"`
- Run parsing: `uv run python -c "from src.legal_parser import parse_all; parse_all()"`
- Run chunking: `uv run python -c "from src.chunker import chunk_all; chunk_all()"`
- Ingest indexes:
  - Dense baseline: `uv run python src/vector_rag/ingest.py`
  - Hybrid BM25: `uv run python src/vector_rag/hybrid_ingest.py`
  - Tree index: `uv run python src/tree_reasoning/tree_index.py`
- Generate benchmark dataset: `uv run python scripts/generate_benchmark.py --count 200`
- Run benchmark: `uv run python scripts/run_benchmark.py --compare-all --top-k 5`
- Chat assistant CLI: `uv run python src/chat.py --method hybrid-bm25`

## Environment Variables

Defined in `.env`:
- `BASE_URL`: OpenAI-compatible endpoint with `/v1` suffix (e.g. `http://your-api-gateway:8080/v1`)
- `API_KEY`: Gateway API key
- `CHAT_MODEL`: Generation model for answers (e.g. `code`)
- `CHAT_REASONING_EFFORT`: Reasoning level for generation (`low`, `medium`, `high`)
- `RETRIEVAL_MODEL`: Model for tree reasoning / navigation (e.g. `code`)
- `RETRIEVAL_REASONING_EFFORT`: Reasoning level for navigation (`low`, `medium`, `high`)
- `TYPESAFE_API_KEY`: TypeSafe AI key for Jev navigator

## Directory Structure

- `data/`: Corpus PDFs, extracted JSON, parsed trees, chunks, Qdrant stores, benchmark datasets.
- `scripts/`: Download corpus, generate benchmark, run benchmark CLI.
- `src/`: Extractor, legal parser, chunker, LLM client, context formatter, evaluation harness, chat assistant.
  - `src/vector_rag/`: Dense baseline (ingest, retriever, RAG) and Hybrid BM25 (hybrid_ingest, hybrid_retriever, hybrid_rag).
  - `src/tree_reasoning/`: PageIndex tree builder and retrievers (`tree-llm`, `tree-jev`, `tree-d3`).

## Retrieval Methods

1. `dense-baseline`: Qdrant dense cosine search (`indo_law_v1`).
2. `hybrid-bm25`: Qdrant dense + sparse BM25 with Reciprocal Rank Fusion (`indo_law_v2`).
3. `tree-llm`: Hierarchical tree index (BAB $\rightarrow$ Pasal) with LLM reasoning navigator.
4. `tree-jev`: Hierarchical tree index with TypeSafe AI Jev System 1 navigator.
5. `tree-d3`: Hierarchical tree index with local open-weight `vllm-sr/d3-edge` (0.6B) System 1 navigator.
