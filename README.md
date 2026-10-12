# Indonesian Legal RAG Benchmark: Vector vs Tree Search

Benchmark comparing retrieval architectures on Indonesian legal documents (Undang-Undang, Peraturan Pemerintah, Perpres, Permen).

## Retrieval Architectures

### 1. Vector RAG
- **`dense-baseline`**: Dense vector search using Qdrant with cosine distance (`indo_law_v1`). Vanilla baseline.
- **`hybrid-bm25`**: Hybrid search combining dense embeddings and sparse BM25 term vectors via Qdrant Reciprocal Rank Fusion (RRF) (`indo_law_v2`).

### 2. Tree Reasoning
- **`tree-llm`**: Document structure tree index (BAB $\rightarrow$ Pasal $\rightarrow$ Ayat) with LLM reasoning navigator down branches to leaf articles.
- **`tree-jev`**: Document structure tree index navigated by TypeSafe AI Jev (System 1 model) using `Choice` and `Score` with confidence threshold safeguards.
- **`tree-d3`**: Document structure tree index navigated by local open-weight System 1 decision model `vllm-sr/d3-edge` (0.6B) running on CPU. Zero retrieval API cost, branch choice via decision readouts.

---

## Models & Configuration

All configured in `.env`:
- `BASE_URL`: OpenAI-compatible endpoint with `/v1` suffix (e.g. `http://your-api-gateway:8080/v1`)
- `API_KEY`: Gateway API key
- `CHAT_MODEL`: Generation model for final answers (default: `code`, resolving to `gemini-3.8-flash`)
- `CHAT_REASONING_EFFORT`: Reasoning level for generation (`low`, `medium`, `high`)
- `RETRIEVAL_MODEL`: Model for tree reasoning / navigation (default: `code`)
- `RETRIEVAL_REASONING_EFFORT`: Reasoning level for navigation (`low`, `medium`, `high`)
- `TYPESAFE_API_KEY`: API key for TypeSafe Jev navigator
- Local Embeddings: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384d, fast ONNX via `fastembed`).

---

## Setup

```bash
# Install uv and sync dependencies
uv sync

# Configure environment
cp .env.example .env

# Verify linter
uv run ruff check .
```

---

## Quick Start

```bash
# 1. Download corpus (landmark UUs + executive PPs)
uv run python scripts/download_corpus.py --limit 30

# 2. Extract text and parse legal hierarchy
uv run python -c "from src.extractor import extract_all; extract_all()"
uv run python -c "from src.legal_parser import parse_all; parse_all()"
uv run python -c "from src.chunker import chunk_all; chunk_all()"

# 3. Ingest into Qdrant collections
uv run python src/vector_rag/ingest.py
uv run python src/vector_rag/hybrid_ingest.py
uv run python src/tree_reasoning/tree_index.py

# 4. Generate benchmark dataset (200 QA pairs)
uv run python scripts/generate_benchmark.py --count 200

# 5. Run head-to-head benchmark
uv run python scripts/run_benchmark.py --compare-all --limit 10

# 6. Interactive CLI
uv run python src/chat.py --method hybrid-bm25
```

---

## Benchmark Results

Evaluated on 5 benchmark queries across difficulty levels (`direct`, `multi_hop`, `cross_ref`):

| Method | Page-Hit@5 | Page-Recall@5 | Page-MRR | Pasal-Recall@5 | Faithfulness | Relevancy | Cit-Accuracy | p50 Latency | Avg Tokens | Avg Cost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`dense-baseline`** | 60.00% | 50.00% | 0.600 | 60.00% | **0.70** | 0.40 | 32.00% | 16.70s | 5,390 | $0.0029 |
| **`hybrid-bm25`** | **80.00%** | **73.33%** | **0.700** | **60.00%** | **0.70** | **0.70** | **40.00%** | 13.72s | 5,641 | $0.0031 |
| **`tree-llm`** | 40.00% | 33.33% | 0.400 | 40.00% | 0.50 | 0.40 | **40.00%** | 70.04s | 8,096 | $0.0043 |
| **`tree-jev`** | 20.00% | 20.00% | 0.100 | 40.00% | 0.40 | 0.30 | 6.67% | **12.90s** | 13,781 | $0.0043 |
| **`tree-d3`** (3-query smoke)* | 66.67% | 55.56% | 0.500 | 66.67% | 0.50 | 0.83 | 20.00% | 84.35s | 7,320 | $0.0033 |

*`tree-d3` evaluated on 3-query smoke subset; other methods evaluated on 5-query benchmark set. Billed model rates: `code` (Flash 3.8) at $0.50/MTok input, $1.50/MTok output; TypeSafe Jev System 1 at $0.042/MTok input ($0 output); `d3-edge` is self-hosted/free ($0.00).

### Cost Breakdown by Pipeline Phase

Retrieval and answer generation have distinct cost profiles:

| Method | Retrieval Engine | Retrieval Cost | Synthesis Engine | Synthesis Cost | Total Cost / Query |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **`dense-baseline`** | Local MiniLM (CPU) | **$0.00 (Free)** | Flash 3.8 (`code`) | $0.0029 | **$0.0029** |
| **`hybrid-bm25`** | Local MiniLM + BM25 (CPU) | **$0.00 (Free)** | Flash 3.8 (`code`) | $0.0031 | **$0.0031** |
| **`tree-llm`** | Flash 3.8 tree navigation | ~$0.0012* | Flash 3.8 (`code`) | ~$0.0031* | **$0.0043** |
| **`tree-jev`** | TypeSafe Jev 1.13 System 1 | $0.0003 | Flash 3.8 (`code`) | $0.0040 | **$0.0043** |
| **`tree-d3`** | Local `vllm-sr/d3-edge` 0.6B (CPU) | **$0.00 (Free)** | Flash 3.8 (`code`) | $0.0033 | **$0.0033** |

- **Vector RAG (`dense-baseline`, `hybrid-bm25`)**: Retrieval runs 100% locally on CPU via FastEmbed (`paraphrase-multilingual-MiniLM-L12-v2`) and Qdrant. Zero LLM API calls. 100% of the query cost comes from the synthesis step drafting the Indonesian answer via Flash 3.8.
- **`tree-llm`**: Incurs paid API costs in both phases. Retrieval queries Flash 3.8 to reason down Bab/Pasal branches (~2.4k tokens), and synthesis queries Flash 3.8 to draft the answer (~5.4k tokens). (*Phase split is estimated from single-hop navigation vs synthesis prompts; both run under `code`).
- **`tree-jev`**: Incurs paid API costs in both phases across two providers. Retrieval uses TypeSafe Jev 1.13 System 1 ($0.042/MTok input, $0 output) costing ~$0.0003 per query (~6.5k navigation tokens). Synthesis uses Flash 3.8 via 9router costing ~$0.0040 per query (~7.4k context tokens).
- **`tree-d3`**: Retrieval runs 100% locally on CPU via `vllm-sr/d3-edge` (0.6B parameters). Zero API cost and zero cloud dependencies for retrieval. 100% of query cost comes from Phase 2 answer synthesis ($0.0033).

## Findings & Analysis

### 1. `hybrid-bm25` Dominates Legal Statutory Retrieval
- **Accuracy**: Highest Page-Hit@5 (80.00%), Page-Recall@5 (73.33%), Page-MRR (0.700), Faithfulness (0.70), and Relevancy (0.70).
- **Why it won**: Indonesian statutory queries contain exact numeric anchors ("Pasal 478", "Pasal 20") and legal keywords ("pencurian ringan", "hak subjek data"). Dense vector embeddings alone dilute numeric tokens across semantic clusters. BM25 sparse vectors isolated exact token matches, while dense vectors captured conceptual relevance. Reciprocal Rank Fusion (RRF) boosted correct statutory chunks to rank 1-2.
- **Efficiency**: Fast p50 latency (13.72s) — dense and sparse scoring execute in parallel via Qdrant prefetch without sequential LLM round-trips.

### 2. `dense-baseline` Struggles on Numeric & Article Distinction
- **Accuracy**: 60.00% Page-Hit@5, 50.00% Page-Recall@5, 0.600 Page-MRR, 0.70 Faithfulness, 0.40 Relevancy.
- **Limitation**: While multilingual embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) map Indonesian legal terminology well, different articles within the same chapter (e.g. Pasal 476, 477, 478 in KUHP) share nearly identical semantic context. Dense search alone often retrieved related but incorrect nearby articles (e.g., retrieving Pasal 127 instead of Pasal 478).
- **Efficiency**: Low token consumption (5,390 tokens), but 20% lower page hit rate than `hybrid-bm25`.

### 3. `tree-llm` (PageIndex Tree) Delivers High Precision at High Latency
- **Accuracy**: 40.00% Page-Hit@5, 40.00% Citation Accuracy, 0.50 Faithfulness.
- **Behavior**: Successfully navigated document hierarchy without vector search on clean statutory queries (e.g., locating UU PDP Pasal 16 & 20 for data processing principles).
- **Trade-off**: High latency (70.04s p50) and token overhead (8,096 avg tokens). Reasoning down tree branches requires sequential LLM hops. Best suited for deep structured documents where vector indexing is impractical or when hierarchical context is critical.
- **Document-dependent**: `tree-llm` achieved 100% hit rate on UU PDP (27/2022) queries (q_pdp_01, q_pdp_02) where the legal structure tree is smaller and well-organized, but 0% on all three KUHP (UU 1/2023) queries where the tree has 34+ BABs and 600+ Pasals. Tree-based navigation scales poorly with document complexity unless the tree is pre-pruned.

### 4. `tree-jev` (Adjusted Graph Navigation)
- **Navigation Adjustments**: Document candidate routing + candidate BAB scoring enabled `tree-jev` to target the right laws. Retrieval hit rate jumped from **0.00% to 20.00% Page-Hit@5**, **20.00% Page-Recall@5**, and **0.100 Page-MRR**.
- **Speed**: Fastest p50 query latency (**12.90s**) across all four retrieval backends.
- **Cost**: Split pricing — Jev navigation tokens billed at $0.042/MTok input ($0 output); answer synthesis billed via `code` (Flash 3.8). Average query cost **$0.0043**.
- **Core Takeaway**: While adjusted graph navigation improved accuracy, System 1 confidence on Indonesian syntax still triggered the 0.60 threshold fallback on complex articles. Demonstrates that while System 1 decision-making provides massive latency savings, domain fine-tuning on Indonesian legal corpus is essential to close the accuracy gap with hybrid search.
### 5. `tree-d3` (Local 0.6B Foundation Decision Model)
- **Accuracy**: 66.67% Page-Hit@5, 55.56% Page-Recall@5, 0.500 Page-MRR, and 66.67% Pasal-Recall@5. Decision probabilities accurately route legal queries to relevant statutory articles.
- **Cost**: Retrieval costs **$0.00**. Entire retrieval runs locally on CPU with zero token billing. Total query cost ($0.0033) is exclusively for Phase 2 synthesis via Flash 3.8.
- **Zero-shot Decision Dynamics**: Zero-shot normalized entropy confidence on Indonesian legal choices is low (~0.05–0.15) compared to English generic routing. Using candidate article probability distributions rather than strict binary confidence gates enables reliable ranking without false subtree fallback.
- **CPU Latency Profile**: High CPU latency (84.35s p50) due to PyTorch reference CPU implementation of Gated DeltaNet linear-attention layers when compiled GPU kernels (`flash-linear-attention`, `causal_conv1d`) are absent. GPU deployment or ONNX/INT8 quantization recommended for production sub-second execution.
- **Fine-Tuning Pipeline**: Extracted 404 ground-truth legal decision trajectories (`data/d3_navigation_train.jsonl`) via `scripts/export_d3_dataset.py`. Ready for supervised LoRA fine-tuning on Google Colab or remote GPU via `scripts/train_d3_colab.py`.


---

## Architectural Trade-off Summary

| Metric / Requirement | `dense-baseline` | `hybrid-bm25` | `tree-llm` | `tree-jev` | `tree-d3` (Local 0.6B) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Retrieval Accuracy** | Moderate (60%) | **Highest (80%)** | Moderate (40%) | Baseline (20%) | High (66.67% on test) |
| **p50 Latency** | Fast (16.70s) | Fast (13.72s) | High (70.04s) | **Fastest (12.90s)** | High on CPU (84.35s) |
| **Vector DB Required** | Yes (Qdrant) | Yes (Qdrant) | **No (Vectorless)** | **No (Vectorless)** | **No (Vectorless)** |
| **Retrieval API Cost** | **$0.00 (Local)** | **$0.00 (Local)** | ~$0.0012 | $0.0003 | **$0.00 (Local)** |
| **Total Cost / Query** | $0.0029 | $0.0031 | $0.0043 | $0.0043 | $0.0033 |
| **Production Recommendation** | Baseline only | **Recommended (Vector)** | Deep research | Cloud System 1 | **Recommended for Private/Local** |
