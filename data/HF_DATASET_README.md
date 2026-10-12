---
annotations_creators:
- machine-generated
language:
- id
license: apache-2.0
task_categories:
- zero-shot-classification
- text-classification
tags:
- legal
- indonesian
- decision-model
- tree-navigation
- rag
- d3-edge
size_categories:
- 1K<n<10K
---

# Indonesian Legal Tree Navigation Dataset (`d3-indo-legal-navigation`)

Supervised tree navigation trajectory dataset extracted from Indonesian statutory legal documents (UU, PP, Perpres, Permen) to train System 1 decision models like `vllm-sr/d3-edge` (0.6B).

## Dataset Summary

Tree navigation formulates statutory retrieval as hierarchical decision paths instead of dense vector k-NN search:
1. `root_to_bab`: Navigate from document root to target chapter (`BAB`).
2. `bab_to_pasal`: Navigate from chapter node down to specific article (`Pasal`).

Each row represents an N-way discrete choice decision where the model picks the correct branch identifier given the user query and candidate sections.

## Data Splits & Statistics

- **Total records**: 7,436
- **`root_to_bab` decisions**: 3,718
- **`bab_to_pasal` decisions**: 3,718
- **Document coverage**: 30/30 Indonesian statutory regulations (UU, PP, Perpres)
- **Format**: JSON Lines (`train.jsonl`)
## Schema

Each JSON record contains:
- `query` (`str`): User legal question in Indonesian.
- `state` (`dict`): Current search state containing `query` and `current_section` title.
- `question_type` (`str`): `"choice"` (N-way discrete choice).
- `instructions` (`str`): Indonesian prompt instructing the decision model.
- `criteria` (`dict[str, str]`): Candidate options mapping option IDs (e.g. `bab_III`, `pasal_28`) to titles/summaries.
- `target_choice` (`str`): Ground-truth target option ID within `criteria`.
- `doc_id` (`str`): Legal document identifier (e.g. `uu_27_2022`).
- `level` (`str`): Navigation level (`root_to_bab` or `bab_to_pasal`).
- `benchmark_id` (`str`): Corresponding benchmark question ID.

## Intended Use

Fine-tuning decision readouts and causal backbones (`vllm-sr/d3-edge`, `Qwen/Qwen3.5-0.6B`) on structured decision-making for legal RAG navigation.
