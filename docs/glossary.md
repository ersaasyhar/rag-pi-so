# Legal & Retrieval Glossary

Comprehensive reference for Indonesian statutory terminology, information retrieval metrics, and System 1 tree reasoning concepts used across this repository.

---

## 1. Indonesian Statutory Hierarchy & Document Structure

Under Law No. 12/2011 (as amended by Law No. 13/2022) on the Formation of Legislation (*Pembentukan Peraturan Perundang-undangan*), Indonesian laws follow a strict constitutional hierarchy:

### Statutory Types

- **`Undang-Undang (UU)` (Statute / Act)**: Primary legislation enacted jointly by the House of Representatives (*Dewan Perwakilan Rakyat / DPR*) and the President. Establishes core rights, liabilities, criminal sanctions, and state institutional mandates.
- **`Peraturan Pemerintah (PP)` (Government Regulation)**: Executive regulations issued by the President to execute the provisions of an Act (*menjalankan Undang-Undang sebagaimana mestinya*). Cannot establish new criminal penalties or obligations absent statutory authorization.
- **`Peraturan Presiden (Perpres)` (Presidential Regulation)**: Executive orders issued by the President to govern administrative operational matters, institutional structures, or policy execution.
- **`Peraturan Menteri (Permen)` (Ministerial Regulation)**: Technical implementing rules promulgated by individual cabinet ministers within their portfolio jurisdiction.

### Structural Divisions within Legislation

- **`BAB` (Chapter)**: Top-level Roman-numeral division grouping related legal matters (e.g. `BAB I KETENTUAN UMUM`, `BAB II ASAS DAN TUJUAN`).
- **`Bagian` (Part)**: Intermediate subdivision within a `BAB` grouping specific subject areas.
- **`Paragraf` (Paragraph / Sub-part)**: Numbered sub-division within a `Bagian` organizing detailed provisions.
- **`Pasal` (Article)**: The fundamental operational unit of Indonesian legislation. Every legal duty, prohibition, or definition resides in a `Pasal`.
- **`Ayat` (Clause / Paragraph)**: Numbered sub-provisions within a `Pasal` (e.g., `Pasal 5 ayat (1)`), demarcating distinct conditions, exceptions, or procedural steps.
- **`Huruf` (Letter / Item)**: Alphabetical list items under an `Ayat` or `Pasal` (e.g., `huruf a`, `huruf b`) listing enumerations or conditions.
- **`Ketentuan Umum` (General Provisions)**: Standardized first chapter (`BAB I`) providing binding legal definitions of terms used throughout the regulation. Crucial anchor for statutory interpretation.
- **`Penjelasan` (Elucidation / Explanatory Memorandum)**: Official section-by-section legislative commentary attached to the statute. Serves as authentic interpretation (*penafsiran otentik*) for statutory ambiguity.

---

## 2. Information Retrieval & Evaluation Metrics

### Retrieval Engines & Fusion

- **Dense Vector Retrieval**: Semantic search encoding text passages into high-dimensional dense embeddings (`sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions) and computing cosine similarity in Qdrant (`indo_law_v1`). Captures paraphrase and semantic intent.
- **Sparse BM25 Retrieval**: Lexical search utilizing term frequency and inverse document frequency (TF-IDF family). Critical for statutory text to match exact statutory citations, article numbers, and formal legal phrasing without vector drift.
- **Reciprocal Rank Fusion (RRF)**: Rank-based score blending combining rankings from dense and sparse retrievers without score calibration:
  $$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
  where $k = 60$ (standard smoothing constant) and $r_m(d)$ is document rank in system $m$. Implemented via Qdrant multi-vector search (`indo_law_v2`).

### Accuracy in Information Retrieval vs Classification

- **Failure of Classical Classification Accuracy**: In classical classification, accuracy is defined as $\frac{TP + TN}{TP + TN + FP + FN}$. In statutory retrieval, an information need matches only $1 \dots 4$ relevant pages out of $2{,}609$ corpus pages ($\approx 0.1\%$, $< 0.2\%$ positive prevalence). A degenerate model returning an empty candidate set achieves $> 99.8\%$ classification accuracy while retrieving zero useful law.
- **`Top-1 Accuracy` (`Hit@1` / Exact Match@1)**: Canonical metric for first-rank retrieval accuracy in IR benchmarks (BEIR, MS MARCO, DPR). Binary indicator evaluating whether the primary candidate chunk placed at rank 1 contains the ground-truth legal authority:
  $$\text{Hit@1} = \mathbb{I}\left(\text{Pages}(c_1) \cap P_{\text{gold}} \neq \emptyset\right)$$
  For Vector RAG (`dense-baseline`, `hybrid-bm25`), evaluates whether chunk #1 contains the target statutory page. For Tree RAG (`tree-llm`, `tree-jev`, `tree-d3`), evaluates whether the primary navigated article hop lands on the target statutory page.
- **`Top-K Accuracy` (`Page-Hit@K`)**: Evaluates whether the retrieval candidate window of size $K$ surfaces at least one ground-truth page:
  $$\text{Hit@K} = \mathbb{I}\left(\exists i \le K : \text{Pages}(c_i) \cap P_{\text{gold}} \neq \emptyset\right)$$
- **`Page-MRR` (Mean Reciprocal Rank)**: Continuous rank-weighted harmonic accuracy penalizing lower-ranked hits:
  $$\text{MRR} = \frac{1}{\min \{i \ge 1 : \text{Pages}(c_i) \cap P_{\text{gold}} \neq \emptyset\}}$$
  If the target page is at rank 1, score is $1.0$; rank 2 yields $0.5$; rank 3 yields $0.333$; unretrieved within candidate pool yields $0.0$.
- **`Decision Routing Accuracy`**: In hierarchical tree navigation, literal multiclass classification accuracy ($\frac{\text{correct}}{\text{total}}$) across discrete branch routing candidates (e.g., selecting the correct `BAB` or `Pasal` from candidate menus). Evaluates System 1 reflex decision heads independently of downstream generation.

### Retrieval Quality & Coverage Metrics

- **`Page-Recall@K`**: Proportion of all relevant ground-truth document pages captured within top-$K$ retrieved results:
  $$\text{Page-Recall@K} = \frac{|\bigcup_{i=1}^K \text{Pages}(c_i) \cap P_{\text{gold}}|}{|P_{\text{gold}}|}$$
- **`Pasal-Recall@K`**: Proportion of ground-truth statutory articles (`Pasal`) captured in retrieved passages:
  $$\text{Pasal-Recall@K} = \frac{|\text{Pasals}(\text{Top-}K) \cap \text{Pasal}_{\text{gold}}|}{|\text{Pasal}_{\text{gold}}|}$$
  Evaluates whether the specific governing statutory article was located.
- **`NDCG@K` (Normalized Discounted Cumulative Gain)**: Rank-weighted metric measuring retrieval quality with logarithmic position discounting:
  $$\text{DCG@K} = \sum_{i=1}^K \frac{\text{rel}_i}{\log_2(i + 1)}, \quad \text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$
  where $\text{IDCG@K}$ is the ideal DCG obtained by sorting all relevant items to top positions.

### Generation & Correctness Metrics

- **`Faithfulness` (Hallucination Prevention)**: Evaluator scoring whether claims in the generated response are factually grounded in and entailed by the retrieved context ($0.0 \dots 1.0$).
- **`Answer Relevancy`**: Evaluator measuring semantic completeness and directness of the response relative to the user query ($0.0 \dots 1.0$).
- **`Citation Accuracy`**: Fraction of cited legal authorities (statute identifier, `Pasal`, page) in the generated response that actually exist in the retrieved context and support the answer claim.

---

## 3. Tree Reasoning & System 1 Decision Models

### Structural Tree Navigation

- **PageIndex Tree**: Hierarchical Document Object Model (DOM) representing Indonesian statutes as trees (`Root` $\rightarrow$ `BAB` $\rightarrow$ `Pasal` $\rightarrow$ `Ayat`). Built by AST-parsing statutory texts (`data/trees/*.json`). Eliminates chunk fragmentation and context loss.
- **Tree Reasoning Navigator**: Search paradigm traversing statutory trees from top to bottom. Instead of cosine similarity across flat chunks, the navigator evaluates candidate chapters (`BAB`), drills down to relevant articles (`Pasal`), and returns complete article nodes.

### Fast Decision Readout Models

- **System 1 Decision Model**: Neural network making discrete choices in a single forward pass without autoregressive token generation. Operates as a fast reflex classifier over candidate tree branches.
- **`vllm-sr/d3-edge`**: 0.6B open-weight foundation decision model developed by Stanford Research and vLLM. Couples a `Qwen3.5-0.6B` backbone with a 255-way linear decision readout matrix (`readout.safetensors` of shape `[255, 1024]`).
- **Decision Readout Projection**: Direct projection from the last hidden token state to choice logits:
  $$\text{logits} = h_{\text{last}} W_{\text{readout}}^T \quad \text{where } W_{\text{readout}} \in \mathbb{R}^{255 \times 1024}$$
  Invalid candidate choices are masked to $-\infty$ before computing softmax, ensuring exact choice selection without decoding loops.
- **PEFT LoRA Training**: Parameter-Efficient Fine-Tuning applying low-rank adapter decomposition ($r=16, \alpha=32$) to backbone attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`). Fine-tunes 1,081,344 parameters (0.1816% of total 595M model parameters).
- **TypeSafe AI Jev**: Cloud-hosted System 1 decision engine providing calibrated `Choice` (discrete path selection) and `Score` (relevance estimation) primitives with confidence threshold safeguards.
