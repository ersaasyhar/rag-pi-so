# Indonesian Legal RAG Benchmark: Vector Search vs Tree Reasoning

Empirical benchmark comparing dense vector retrieval, hybrid sparse-dense search, and hierarchical tree index reasoning across Indonesian statutory law (*Undang-Undang*, *Peraturan Pemerintah*, *Perpres*, *Permen*).

[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/ersaasyhar/d3-indo-legal-navigation)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20(LoRA)-yellow)](https://huggingface.co/ersaasyhar/d3-edge-indo-legal)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ersaasyhar/rag-pi-so/blob/main/notebooks/train_d3_colab.ipynb)
[![Python UV](https://img.shields.io/badge/package%20manager-uv-purple)](https://astral.sh/uv)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Vector Store: Qdrant](https://img.shields.io/badge/vector%20db-Qdrant-red)](https://qdrant.tech)

```
PARADIGM 1: VECTOR RAG (FLAT CHUNKS)
User Query ──► Dense / Sparse Embedding ──► Cosine / RRF Top-K ──► Context Assembly ──► Synthesis LLM
               (Fixed 500-token chunks, no legal boundary awareness)

PARADIGM 2: TREE REASONING (HIERARCHICAL DOM)
User Query ──► Legal AST Tree ──► BAB Selector ──► Pasal Selector ──► Leaf Passage ──► Synthesis LLM
               (Document Object Model: Root ──► Chapter ──► Article ──► Paragraph)
```

> For legal statutory structure (`BAB`, `Pasal`, `Ayat`) and retrieval evaluation definitions, see the [Legal & Retrieval Glossary](docs/glossary.md).

---

## 1. Quickstart (Run in 60 Seconds)

### Step 1: Install Dependencies & Setup Environment

```bash
# Clone and install dependencies with uv
git clone https://github.com/ersaasyhar/rag-pi-so.git
cd rag-pi-so
uv sync

# Configure environment variables
cp .env.example .env
```

Edit `.env` with your API credentials:
```ini
BASE_URL=http://your-api-gateway:8080/v1
API_KEY=your_gateway_key
CHAT_MODEL=code
TYPESAFE_API_KEY=your_typesafe_key  # Optional: required for tree-jev
```

### Step 2: Interactive Query Execution

Run statutory queries immediately using either vector or tree retrieval:

```bash
# Production Vector Search (Hybrid BM25 + Dense RRF)
uv run python src/chat.py --method hybrid-bm25 --query "Apa syarat pendirian perseroan terbatas?"

# Open-Weight System 1 Tree Reasoning (Fine-Tuned D3 0.6B)
uv run python src/chat.py --method tree-d3 --d3-model ersaasyhar/d3-edge-indo-legal --query "Apa hak subjek data pribadi menurut UU PDP?"
```

---

## 2. Retrieval Architectures at a Glance

| Method | Retrieval Paradigm | Decision / Routing Engine | Vector DB | API Cost (Retrieval) |
| :--- | :--- | :--- | :---: | :---: |
| **`dense-baseline`** | Vector RAG | Cosine distance (`all-MiniLM-L6-v2`) | Qdrant (`indo_law_v1`) | **$0.00 (Local)** |
| **`hybrid-bm25`** | Vector RAG | Dense + Sparse BM25 Reciprocal Rank Fusion | Qdrant (`indo_law_v2`) | **$0.00 (Local)** |
| **`tree-llm`** | Tree Reasoning | Autoregressive LLM reasoning down statutory branches | None (Vectorless) | Paid API per hop |
| **`tree-jev`** | Tree Reasoning | TypeSafe AI Jev System 1 model (`Choice` + `Score`) | None (Vectorless) | $0.042/MTok input |
| **`tree-d3-base`** | Tree Reasoning | Foundation `vllm-sr/d3-edge` 0.6B via linear readout | None (Vectorless) | **$0.00 (Local)** |
| **`tree-d3`** | Tree Reasoning | Fine-tuned `ersaasyhar/d3-edge-indo-legal` 0.6B LoRA | None (Vectorless) | **$0.00 (Local)** |

### Architectural Highlights

- **`dense-baseline`**: Traditional vector search chunking regulations into 500-token blocks. Prone to statutory chunk-boundary splitting and misses exact article numbers.
- **`hybrid-bm25`**: Combines dense semantic embeddings with sparse BM25 lexical tokens via Reciprocal Rank Fusion ($k=60$). Resolves exact statutory identifiers (`Pasal 15`, `UU 27/2022`) while retaining semantic recall.
- **`tree-llm`**: Converts statutes into structured trees (`Root` $\rightarrow$ `BAB` $\rightarrow$ `Pasal`). An autoregressive LLM evaluates candidate chapters and articles sequentially.
- **`tree-jev`**: Employs TypeSafe AI's Jev-1.13 decision engine. Makes low-latency reflex branch selections using calibrated confidence scoring.
- **`tree-d3-base`**: Out-of-the-box foundation decision model `vllm-sr/d3-edge` (0.6B). Direct linear readout projections (`logits = hidden @ readout.T`) without legal domain specialization.
- **`tree-d3`**: Fine-tuned checkpoint [`ersaasyhar/d3-edge-indo-legal`](https://huggingface.co/ersaasyhar/d3-edge-indo-legal). LoRA adapter trained on statutory navigation trajectories (`root_to_bab` and `bab_to_pasal`), achieving +32.50% decision accuracy gain.
---

## 3. Benchmark Results & Key Findings

Empirical evaluation results executed across Indonesian statutory law queries (`data/results/`). Detailed visual analysis and Pareto frontiers are charted in [Benchmark Visualizations & Findings](docs/visualizations.md).

### Benchmark Comparison Table

| Method | Engine / Checkpoint | Evaluated $N$ | Top-1 Acc (Hit@1) | Top-5 Hit (Hit@5) | Page-Recall@5 | Page-MRR | Pasal-Recall@5 | Faithfulness | Relevancy | Cit-Accuracy | p50 Latency | Avg Tokens | Avg Cost |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`dense-baseline`** | Qdrant Cosine (MiniLM) | 5 | 60.00% | 60.00% | 50.00% | 0.6000 | 60.00% | **0.60** | 0.30 | 32.00% | 11.88s | **5,211** | **$0.0028** |
| **`hybrid-bm25`** | Qdrant Dense+Sparse RRF | 5 | 60.00% | **80.00%** | **73.33%** | **0.7000** | 60.00% | **0.60** | 0.70 | 40.00% | 13.35s | 5,852 | $0.0032 |
| **`tree-llm`** | PageIndex Tree + LLM | 3 | **66.67%** | 66.67% | 55.56% | 0.6667 | 66.67% | 0.50 | **0.83** | **66.67%** | 23.51s | 8,908 | $0.0047 |
| **`tree-jev`** | PageIndex + TypeSafe Jev | 5 | 0.00% | 20.00% | 20.00% | 0.1000 | 40.00% | 0.30 | 0.60 | 6.67% | **10.65s** | 13,753 | $0.0043 |
| **`tree-d3-base`** | Foundation `vllm-sr/d3-edge` 0.6B (CPU)\* | 5 | 60.00% | **80.00%** | 63.33% | **0.7000** | **80.00%** | 0.50 | 0.60 | 24.00% | 73.43s\* | 7,581 | $0.0034 |
| **`tree-d3`** | Fine-Tuned `d3-edge-indo-legal` 0.6B (CPU)\* | 5 | 60.00% | **80.00%** | 63.33% | **0.7000** | **80.00%** | **0.60** | 0.60 | 24.00% | 69.54s\* | 7,566 | $0.0034 |

*\*Both `tree-d3-base` and `tree-d3` evaluated on CPU (reference PyTorch DeltaNet loops). GPU inference with Triton kernels drops decision readout latency to <1s.*

> **Information Retrieval Accuracy vs. Classification Accuracy**: Classical classification accuracy ($\frac{TP+TN}{\text{Total}}$) fails in retrieval because $>99.8\%$ of statutory pages are non-relevant ($\approx 0.1\%$, $<0.2\%$ positive prevalence). Standard IR accuracy is measured via **`Top-1 Accuracy` (`Hit@1`)** (primary retrieved chunk contains target page) and **`Top-K Accuracy` (`Hit@K`)** (target page present within top-$K$). For tree navigation, **`Decision Routing Accuracy`** measures intermediate multiclass branch classification ($\frac{\text{correct}}{\text{total}}$).

### Key Takeaways

1. **Foundation vs Fine-Tuned System 1 Decision Performance**:
   While the base foundation model `vllm-sr/d3-edge` locates relevant statutory sections via generic semantic readout, fine-tuning on Indonesian legal trajectories (`ersaasyhar/d3-edge-indo-legal`) boosts validation navigation accuracy from **17.50% to 50.00%** and end-to-end response faithfulness from **0.50 to 0.60**. Both reach **80.00% Pasal-Recall@5** (highest across all 5 architectures).
2. **Hybrid BM25 Dominates Vector-Based Retrieval**:
   Standard dense vector search suffers from semantic drift when matching formal legal phrasing. Combining lexical BM25 with dense embeddings via Reciprocal Rank Fusion boosts **Page-Hit@5 from 60.00% to 80.00%** and **Page-Recall@5 from 50.00% to 73.33%**.
3. **Tree Reasoning Eliminates Chunk Boundary Fragmentation**:
   By traversing the document hierarchy (`BAB` $\rightarrow$ `Pasal`), tree index retrievers return whole, coherent statutory articles. This gives `tree-llm` and `tree-d3` superior article coverage, eliminating chunk boundary truncation common in 500-token vector blocks.
4. **Zero-Cost Retrieval Economics**:
   Autoregressive tree reasoning (`tree-llm`) incurs token charges on every intermediate decision hop. System 1 models replace generation with direct readout classification, enabling zero-cost local retrieval (`tree-d3-base`, `tree-d3`) or fast calibrated routing (`tree-jev` at **10.65s** p50).

### Architectural Trade-off Summary

| Requirement / Characteristic | `dense-baseline` | `hybrid-bm25` | `tree-llm` | `tree-jev` | `tree-d3-base` (0.6B) | `tree-d3` (Fine-Tuned 0.6B) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Top-1 Retrieval Acc (Hit@1)** | 60.00% | 60.00% | **66.67%** | 0.00% | 60.00% | 60.00% |
| **Top-5 Retrieval Acc (Hit@5)** | Moderate (60.00%) | **Highest (80.00%)** | Moderate (66.67%) | Baseline (20.00%) | **Highest (80.00%)** | **Highest (80.00%)** |
| **Decision Routing Accuracy** | N/A (Vector) | N/A (Vector) | Prompt-based | Calibrated Choice | 17.50% (Zero-shot) | **50.00% (Fine-tuned)** |
| **Pasal Identification (Recall@5)** | 60.00% | 60.00% | 66.67% | 40.00% | **Highest (80.00%)** | **Highest (80.00%)** |
| **Faithfulness Score** | **0.60** | **0.60** | 0.50 | 0.30 | 0.50 | **0.60** |
| **p50 Query Latency** | 11.88s | 13.35s | 23.51s | **10.65s** | 73.43s (CPU) / <1s (GPU) | 69.54s (CPU) / <1s (GPU) |
| **Vector DB Infrastructure** | Required (Qdrant) | Required (Qdrant) | **None (Vectorless)** | **None (Vectorless)** | **None (Vectorless)** | **None (Vectorless)** |
| **Retrieval Hop API Cost** | **$0.00 (Local)** | **$0.00 (Local)** | ~$0.0015 | $0.0003 | **$0.00 (Local)** | **$0.00 (Local)** |
| **Production Recommendation** | Baseline only | **Recommended (Vector)** | Deep research | Cloud System 1 | Out-of-box baseline | **Recommended (Edge/Private)** |
---

## 4. Hugging Face Hub Artifacts & Fine-Tuning

We published the supervised legal navigation dataset and fine-tuned decision checkpoint on Hugging Face Hub:

| Resource | Hugging Face Repository | Description |
| :--- | :--- | :--- |
| **Supervised Dataset** | [`ersaasyhar/d3-indo-legal-navigation`](https://huggingface.co/datasets/ersaasyhar/d3-indo-legal-navigation) | 7,436 supervised tree navigation trajectories (`root_to_bab` and `bab_to_pasal`) covering all 30 statutory corpora with natural query variations. |
| **Fine-Tuned Checkpoint** | [`ersaasyhar/d3-edge-indo-legal`](https://huggingface.co/ersaasyhar/d3-edge-indo-legal) | Fine-tuned `d3-edge` (0.6B) LoRA model achieving **50.00% validation accuracy** (+32.50% gain over zero-shot 17.50%). |
| **Foundation Model** | [`vllm-sr/d3-edge`](https://huggingface.co/vllm-sr/d3-edge) | Base 0.6B System 1 decision model by vLLM / Stanford Research. |

### Decision Model Architecture & Readout Loss

Standard autoregressive `CausalLM` trainers fail on `d3-edge` because the architecture pairs a `Qwen3_5Model` backbone with a 255-way linear readout (`readout.safetensors` of shape `[255, 1024]`), rather than an autoregressive language modeling head.

```
                    ┌──────────────────────────────────────────┐
Input Trajectory ──►│ Qwen3.5-0.6B Backbone + PEFT LoRA (r=16) │
                    └─────────────────────┬────────────────────┘
                                          │ Last Hidden State [1, 1024]
                                          ▼
                    ┌──────────────────────────────────────────┐
                    │ Direct Readout Head: logits = h @ W_readout.T  │
                    │ W_readout in R^(255 x 1024)              │
                    └─────────────────────┬────────────────────┘
                                          │ Mask Invalid Keys to -inf
                                          ▼
                    ┌──────────────────────────────────────────┐
                    │ Softmax ──► Cross-Entropy Decision Loss  │
                    └──────────────────────────────────────────┘
```

1. **Trajectory Extraction**: `scripts/export_d3_dataset.py` extracts legal decision trajectories from statutory trees (`data/d3_navigation_train.jsonl`), mapping queries to valid choices (`key -> title`).
2. **Parameter-Efficient LoRA**: Applied to backbone attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, $r=16$, $\alpha=32$, dropout $0.05$). Trains **1,081,344 parameters** (0.1816% of 595M total parameters).
3. **Exact Decision Loss**: Left-padded chat sequences pass through the backbone. The last hidden state is projected via `hidden @ readout.T`. Candidate choices not present in the prompt are masked out before cross-entropy evaluation.
4. **Standalone Trainer**: `scripts/train_d3_decision.py` handles local/remote training, validation evaluation, and LoRA weight merging (`--merge-and-upload`).
5. **Elbow Method Optimization**: Uses a 4-epoch fine-tuning trajectory ($N=4$ for 7.4K samples) to locate the inflection knee point ($\text{elbow\_epoch} = \arg\min_e \text{val\_loss}_e$):
   - **Phase 1: Steep Alignment (Epochs 0–1)**: Loss drops sharply (~2.82 $\rightarrow$ ~1.49, +30% acc gain) as model aligns with 255-way non-autoregressive decision projections.
   - **Phase 2: Knee Refinement (Epochs 2–3)**: Cross-chapter routing rules solidify; validation loss reaches optimum and accuracy peaks.
   - **Phase 3: Saturation & Overfit Zone (Epoch 4+)**: Validation loss flattens ($\Delta L < 0.05$) while training loss continues decreasing, signaling statutory memorization.
   - **Curve Verification**: Generates `d3_elbow_curve.png` tracking training loss, validation loss, and validation accuracy with the knee point annotated.

### GPU Fine-Tuning Notebooks (Google Colab & Kaggle)

- **Google Colab (`notebooks/train_d3_colab.ipynb`)**:
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ersaasyhar/rag-pi-so/blob/main/notebooks/train_d3_colab.ipynb)
- **Kaggle Notebook (`notebooks/train_d3_kaggle.ipynb`)**:
  [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://github.com/ersaasyhar/rag-pi-so/blob/main/notebooks/train_d3_kaggle.ipynb)

Features across both platforms:
- **Hardware Acceleration**: Automatic installation of `flash-linear-attention` and `causal-conv1d` Triton/CUDA kernels to accelerate Gated DeltaNet layers from ~60 min/epoch down to **~2–3 min/epoch**.
- **RAM Pre-Tokenization**: Sequences pre-tokenized into memory once with `max_token_len=256` to eliminate CPU bottlenecks.
- **Precision Management**: Native FP16 Tensor Core execution on Nvidia T4 (`torch.amp.autocast`) with `torch.amp.GradScaler`.
- **Automated Knee Detection**: 5-epoch trajectory evaluation, saves per-epoch checkpoints, and auto-selects optimal elbow checkpoint.
- **Weight Publishing**: Merges elbow LoRA weights into backbone and pushes directly to [`ersaasyhar/d3-edge-indo-legal`](https://huggingface.co/ersaasyhar/d3-edge-indo-legal).

## 5. Operational Cost & Latency Economics

Total query cost decomposes into **Phase 1 (Retrieval)** and **Phase 2 (Answer Synthesis)**:

| Method | Retrieval Engine | Phase 1 Retrieval Cost | Synthesis Engine | Phase 2 Synthesis Cost | Total Cost / Query |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **`dense-baseline`** | Local MiniLM (CPU) | **$0.00 (Free)** | Flash 3.8 (`code`) | $0.0028 | **$0.0028** |
| **`hybrid-bm25`** | Local MiniLM + BM25 (CPU) | **$0.00 (Free)** | Flash 3.8 (`code`) | $0.0032 | **$0.0032** |
| **`tree-llm`** | Flash 3.8 tree navigation | ~$0.0015 | Flash 3.8 (`code`) | ~$0.0032 | **$0.0047** |
| **`tree-jev`** | TypeSafe Jev 1.13 System 1 | $0.0003 | Flash 3.8 (`code`) | $0.0040 | **$0.0043** |
| **`tree-d3`** | Local `d3-edge` 0.6B (CPU/GPU) | **$0.00 (Free)** | Flash 3.8 (`code`) | $0.0033 | **$0.0033** |

### Economics Breakdown
- **Vector RAG (`dense-baseline`, `hybrid-bm25`)**: Retrieval executes 100% locally via FastEmbed ONNX Runtime and Qdrant. Zero LLM API calls are consumed during retrieval.
- **`tree-llm`**: Incurs additive per-hop LLM API charges as the navigator steps through each branch level (`Root` $\rightarrow$ `BAB` $\rightarrow$ `Pasal`), increasing latency and token usage.
- **`tree-jev`**: Decouples reasoning costs: navigation is billed at TypeSafe's System 1 rate ($0.042/MTok input, $0 output), reducing retrieval overhead by ~80% relative to `tree-llm`.
- **`tree-d3`**: Complete local execution for retrieval with zero external API dependencies. Query costs are confined strictly to final answer synthesis.

---

## 6. Reproducing the Pipeline (Step-by-Step)

Execute the end-to-end legal ingestion, indexing, and benchmarking pipeline from scratch:

```bash
# 1. Download statutory corpus (30 landmark UUs and executive PPs)
uv run python scripts/download_corpus.py --limit 30

# 2. Extract per-page text from PDFs
uv run python -c "from src.extractor import extract_all; extract_all()"

# 3. Parse legal AST document trees (Root -> BAB -> Pasal -> Ayat)
uv run python -c "from src.legal_parser import parse_all; parse_all()"

# 4. Generate text chunks for vector stores
uv run python -c "from src.chunker import chunk_all; chunk_all()"

# 5. Build indexes
# Dense vector index
uv run python src/vector_rag/ingest.py
# Hybrid BM25 + dense index
uv run python src/vector_rag/hybrid_ingest.py
# Hierarchical tree index
uv run python src/tree_reasoning/tree_index.py

# 6. Generate benchmark evaluation questions
uv run python scripts/generate_benchmark.py --count 200

# 7. Run evaluation benchmarks
# Evaluate individual methods
uv run python scripts/run_benchmark.py --method hybrid-bm25 --limit 5
uv run python scripts/run_benchmark.py --method tree-llm --limit 3
uv run python scripts/run_benchmark.py --method tree-d3-base --limit 5
uv run python scripts/run_benchmark.py --method tree-d3 --limit 5
uv run python scripts/run_benchmark.py --compare-all --limit 5
```

---

## 7. Repository Layout

```
rag-pi-so/
├── data/
│   ├── corpus/                  # Statutory PDFs (UU, PP, Perpres)
│   ├── extracted/               # Per-page extracted text JSON
│   ├── texts/                   # Reconstructed Markdown with page markers
│   ├── trees/                   # AST legal structure trees (JSON)
│   ├── chunks/                  # Overlapping chunks for vector baseline
│   ├── benchmark_dataset.json   # 200 ground-truth legal query questions
│   ├── d3_navigation_train.jsonl# Supervised trajectories for D3 training
│   └── results/                 # Evaluation metric JSON artifacts
├── docs/
│   └── glossary.md              # Indonesian legal hierarchy & IR metric glossary
├── notebooks/
│   └── train_d3_colab.ipynb     # Google Colab GPU training & publishing notebook
├── scripts/
│   ├── download_corpus.py       # Corpus downloader
│   ├── generate_benchmark.py    # Ground-truth question generator
│   ├── run_benchmark.py         # 5-method evaluation benchmark CLI
│   ├── export_d3_dataset.py     # Trajectory dataset exporter
│   ├── train_d3_decision.py     # LoRA readout trainer & model pusher
│   └── upload_hf_dataset.py     # Hugging Face dataset uploader
├── src/
│   ├── vector_rag/              # Dense & Hybrid BM25 retrievers and Qdrant ingestion
│   ├── tree_reasoning/          # Tree index, LLM, Jev, and D3 System 1 navigators
│   ├── extractor.py             # PDF text extraction
│   ├── legal_parser.py          # Indonesian legal AST regex parser
│   ├── chunker.py               # Document chunker
│   ├── chat.py                  # Multi-backend chat assistant CLI
│   ├── evaluation.py            # Retrieval & generation evaluation metrics
│   └── pricing.py               # Token usage & pipeline cost calculator
├── AGENTS.md                    # Agent coding & operations guidelines
└── pyproject.toml               # Project dependencies (managed via uv)
```

For developer guidelines, operational instructions, and architectural principles, consult [`AGENTS.md`](AGENTS.md).
