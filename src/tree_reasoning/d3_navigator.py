"""PageIndex tree index with local vllm-sr/d3-edge (0.6B) System One decision navigator."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.context_formatter import format_context
from src.llm_client import TokenUsage, chat_completion, record_tokens, token_meter
from src.schema import Citation, RAGResponse, RetrievedChunk
from src.tree_reasoning.tree_index import build_tree_for_doc

logger = logging.getLogger(__name__)

DATA_DIR = BASE_DIR / "data"
MANIFEST_PATH = DATA_DIR / "corpus_manifest.json"
EXTRACTED_DIR = DATA_DIR / "extracted"

_GLOBAL_D3_MODELS: dict[tuple[str, str | None, str], Any] = {}


def get_d3_model(
    model_name: str | None = None,
    device: str = "cpu",
    adapter_path: str | Path | None = None,
) -> Any | None:
    """Singleton lazy loader keyed by model_name, adapter_path, and device."""
    if model_name is None:
        model_name = os.environ.get("D3_MODEL_NAME", "ersaasyhar/d3-edge-indo-legal")
    if adapter_path is None:
        adapter_path = os.environ.get("D3_ADAPTER_PATH")
    adapter_str = str(adapter_path) if adapter_path is not None else None
    cache_key = (model_name, adapter_str, device)

    if cache_key not in _GLOBAL_D3_MODELS:
        from transformers import AutoModel

        def try_local(name: str) -> Any | None:
            for kwargs in [
                {"local_files_only": True},
                {"revision": "18d61df78684675463af9ae732872478845c5a79", "local_files_only": True},
            ]:
                try:
                    return AutoModel.from_pretrained(
                        name, trust_remote_code=True, device=device, **kwargs
                    )
                except Exception:
                    pass
            return None

        # 1. Try local cache for requested model
        loaded_model = try_local(model_name)
        # 2. Try local cache for base fallback
        if loaded_model is None and model_name != "vllm-sr/d3-edge":
            loaded_model = try_local("vllm-sr/d3-edge")
        # 3. If still None and remote download explicitly allowed, try remote
        if loaded_model is None and os.environ.get("D3_DOWNLOAD_REMOTE") == "1":
            try:
                loaded_model = AutoModel.from_pretrained(
                    model_name, trust_remote_code=True, device=device
                )
            except Exception:
                try:
                    loaded_model = AutoModel.from_pretrained(
                        "vllm-sr/d3-edge", trust_remote_code=True, device=device
                    )
                except Exception:
                    pass

        if loaded_model is None:
            logger.warning("Failed to load D3 model checkpoint.")
            return None

        if loaded_model is not None and adapter_path is not None:
            adapter_p = Path(adapter_path)
            if adapter_p.exists():
                try:
                    from peft import PeftModel

                    logger.info("Loading LoRA adapter from %s...", adapter_path)
                    loaded_model.runtime.backbone = PeftModel.from_pretrained(
                        loaded_model.runtime.backbone, str(adapter_p)
                    )
                except Exception as lora_e:
                    logger.warning("Failed to load LoRA adapter from %s: %s", adapter_path, lora_e)
            else:
                logger.warning("LoRA adapter path does not exist: %s", adapter_path)

        _GLOBAL_D3_MODELS[cache_key] = loaded_model
    return _GLOBAL_D3_MODELS.get(cache_key)


class D3Navigator:
    def __init__(
        self,
        model_name: str | None = None,
        adapter_path: str | Path | None = None,
        device: str = "cpu",
        manifest_path: Path = MANIFEST_PATH,
        lazy: bool = False,
        confidence_threshold: float = 0.60,
    ):
        self.model_name = model_name or os.environ.get(
            "D3_MODEL_NAME", "ersaasyhar/d3-edge-indo-legal"
        )
        self.adapter_path = adapter_path or os.environ.get("D3_ADAPTER_PATH")
        self.device = device
        self.manifest_path = manifest_path
        self.confidence_threshold = confidence_threshold
        self._manifest: list[dict[str, Any]] | None = None
        self._model: Any | None = None
        if not lazy:
            self._ensure_model()

    def _ensure_model(self) -> Any | None:
        if self._model is None:
            self._model = get_d3_model(self.model_name, self.device, adapter_path=self.adapter_path)
        return self._model

    def is_available(self) -> bool:
        return self._ensure_model() is not None

    @property
    def manifest(self) -> list[dict[str, Any]]:
        if self._manifest is None:
            if self.manifest_path.exists():
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    self._manifest = json.load(f)
            else:
                self._manifest = []
        return self._manifest

    def _select_candidate_docs(self, query: str, max_docs: int = 2) -> list[dict[str, Any]]:
        """Select top candidate documents based on query keywords and landmark status."""
        q_lower = query.lower()
        scored: list[tuple[float, dict[str, Any]]] = []

        is_crime = any(
            w in q_lower for w in ["pidana", "pencurian", "pembunuhan", "kuhp", "penjara", "denda"]
        )
        is_pdp = any(
            w in q_lower for w in ["data pribadi", "pdp", "subjek data", "pengendali data"]
        )
        is_corp = any(w in q_lower for w in ["perseroan", "pt", "direksi", "saham", "umkm"])
        is_ite = any(
            w in q_lower for w in ["elektronik", "ite", "transaksi elektronik", "informasi"]
        )

        for d in self.manifest:
            score = 0.0
            doc_id = d["doc_id"]
            title = d["judul"].lower()

            if is_crime and "uu_1_2023" in doc_id:
                score += 10.0
            if is_pdp and "uu_27_2022" in doc_id:
                score += 10.0
            if is_corp and ("uu_40_2007" in doc_id or "uu_20_2008" in doc_id):
                score += 8.0
            if is_ite and ("uu_19_2016" in doc_id or "uu_11_2008" in doc_id):
                score += 8.0

            for w in q_lower.split():
                if len(w) > 3 and w in title:
                    score += 2.0

            scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:max_docs]]

    def navigate_node(self, query: str, node: dict[str, Any]) -> list[int]:
        """Navigate down a node using D3 Choice and Score with confidence fallback."""
        children = node.get("nodes", [])
        if not children:
            p_start = node.get("start_index", 1)
            p_end = node.get("end_index", p_start)
            return list(range(p_start, p_end + 1))

        model = self._ensure_model()
        if not model:
            p_start = node.get("start_index", 1)
            p_end = node.get("end_index", p_start)
            return list(range(p_start, p_end + 1))

        options = {
            c["node_id"]: f"{c['title']}. {c.get('summary', '')[:100]}" for c in children[:200]
        }

        try:
            resp = model.system_one(
                state={"query": query, "current_section": node.get("title", "")},
                questions={
                    "next_branch": {
                        "type": "choice",
                        "instructions": "Pilih bagian dokumen yang paling mungkin memuat jawaban pertanyaan.",
                        "criteria": options,
                    },
                    "relevance": {
                        "type": "score",
                        "instructions": "Nilai relevansi bagian dokumen ini terhadap pertanyaan.",
                        "criteria": [
                            "tidak relevan",
                            "sedikit relevan",
                            "cukup relevan",
                            "relevan",
                            "sangat tepat",
                        ],
                    },
                },
            )
            usage = resp.get("usage", {})
            if usage:
                in_tok = usage.get("input_tokens", 0)
                out_tok = usage.get("output_tokens", 0)
                record_tokens(in_tok, out_tok, model="vllm-sr/d3-edge")

            answers = resp.get("answers", {})
            conf = answers.get("relevance", {}).get("confidence", 1.0)
            if conf < self.confidence_threshold:
                logger.warning(
                    "D3 confidence below threshold (%.2f < %.2f) at node %s. Falling back to full subtree.",
                    conf,
                    self.confidence_threshold,
                    node.get("node_id"),
                )
                p_start = node.get("start_index", 1)
                p_end = node.get("end_index", p_start)
                return list(range(p_start, p_end + 1))

            selected_id = answers.get("next_branch", {}).get("choice")
            next_node = next((c for c in children if c["node_id"] == selected_id), children[0])
            return self.navigate_node(query, next_node)
        except Exception as e:
            logger.warning("D3 system_one call failed: %s. Using subtree fallback.", e)
            p_start = node.get("start_index", 1)
            p_end = node.get("end_index", p_start)
            return list(range(p_start, p_end + 1))

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Retrieve chunks across corpus documents using D3 graph navigation."""
        model = self._ensure_model()
        if not model:
            logger.warning("vllm-sr/d3-edge model not available.")
            return []

        candidate_docs = self._select_candidate_docs(query, max_docs=1)
        if not candidate_docs:
            candidate_docs = self.manifest[:1]

        retrieved: list[RetrievedChunk] = []

        for doc in candidate_docs:
            doc_id = doc["doc_id"]
            tree = build_tree_for_doc(doc_id)
            root_nodes = [n for n in tree.get("structure", []) if n.get("nodes")]
            if not root_nodes:
                continue

            ext_path = EXTRACTED_DIR / f"{doc_id}.json"
            p_map: dict[int, str] = {}
            if ext_path.exists():
                with open(ext_path, "r", encoding="utf-8") as fp:
                    ext = json.load(fp)
                p_map = {p["page_number"]: p["text"] for p in ext.get("pages", [])}

            stopwords = {
                "apa",
                "saja",
                "yang",
                "dan",
                "ini",
                "itu",
                "pada",
                "oleh",
                "dari",
                "untuk",
                "menurut",
                "bagaimana",
                "adalah",
                "tentang",
                "baru",
                "tahun",
            }
            meta_words = {
                "definisi",
                "sanksi",
                "pengertian",
                "syarat",
                "aturan",
                "undang",
                "pidana",
                "hukum",
            }
            raw_words = [w for w in query.lower().split() if len(w) >= 3 and w not in stopwords]
            q_words = [w.strip(".,()?!") for w in raw_words if len(w.strip(".,()?!")) >= 3]

            scored_babs = []
            for b in root_nodes:
                child_nodes = b.get("nodes", [])
                if not child_nodes:
                    continue

                def p_score(p):
                    txt = (p.get("title", "") + " " + p.get("summary", "")).lower()
                    return sum(1.0 if w in meta_words else 5.0 for w in q_words if w in txt)

                best_score = max((p_score(p) for p in child_nodes), default=0.0)
                scored_babs.append((best_score, b))

            scored_babs.sort(key=lambda x: x[0], reverse=True)
            top_babs = [b for s, b in scored_babs if s > 0][:2]
            if not top_babs:
                top_babs = root_nodes[:2]

            for bab_node in top_babs:
                child_pasals = bab_node.get("nodes", [])
                if not child_pasals:
                    continue

                def pasal_match(p):
                    txt = (p.get("title", "") + " " + p.get("summary", "")).lower()
                    return sum(1.0 if w in meta_words else 5.0 for w in q_words if w in txt)

                sorted_pasals = sorted(child_pasals, key=pasal_match, reverse=True)
                candidate_pasals = sorted_pasals[:8]

                p_options = {
                    p["node_id"]: f"{p['title']}: {p.get('summary', '')[:80]}"
                    for p in candidate_pasals
                }

                try:
                    resp = model.system_one(
                        state={"query": query, "bab": bab_node.get("title", "")},
                        questions={
                            "pasal": {
                                "type": "choice",
                                "instructions": "Pilih Pasal yang paling tepat memuat jawaban.",
                                "criteria": p_options,
                            },
                            "relevance": {
                                "type": "score",
                                "instructions": "Nilai relevansi pasal ini terhadap pertanyaan.",
                                "criteria": [
                                    "tidak relevan",
                                    "sedikit relevan",
                                    "cukup relevan",
                                    "relevan",
                                    "sangat tepat",
                                ],
                            },
                        },
                    )
                    usage = resp.get("usage", {})
                    if usage:
                        in_tok = usage.get("input_tokens", 0)
                        out_tok = usage.get("output_tokens", 0)
                        record_tokens(in_tok, out_tok, model="vllm-sr/d3-edge")

                    answers = resp.get("answers", {})
                    pasal_ans = answers.get("pasal", {})
                    chosen_id = pasal_ans.get("choice")
                    chosen_pasal = next(
                        (p for p in child_pasals if p["node_id"] == chosen_id), None
                    )
                    probs = pasal_ans.get("probabilities", {})
                    top_prob = probs.get(chosen_id, 0.5)

                    # Add best choice as primary chunk
                    if chosen_pasal:
                        p_start = chosen_pasal.get("start_index", 1)
                        p_end = chosen_pasal.get("end_index", p_start)
                        pages = list(range(p_start, p_end + 1))
                        txt_parts = [p_map[p] for p in pages if p in p_map]
                        retrieved.append(
                            RetrievedChunk(
                                doc_id=doc_id,
                                pages=pages,
                                pasal=chosen_pasal.get("title", ""),
                                text="\n\n".join(txt_parts[:3]),
                                score=top_prob,
                                bab=bab_node.get("title", ""),
                                chunk_id=chosen_pasal.get("node_id", ""),
                            )
                        )

                    # Add other pasals ranked by model probability to fill top_k
                    other_pasals = [p for p in child_pasals if p["node_id"] != chosen_id]
                    other_pasals.sort(key=lambda p: probs.get(p["node_id"], 0.0), reverse=True)
                    for op in other_pasals:
                        p_start = op.get("start_index", 1)
                        p_end = op.get("end_index", p_start)
                        pages = list(range(p_start, p_end + 1))
                        txt_parts = [p_map[p] for p in pages if p in p_map]
                        retrieved.append(
                            RetrievedChunk(
                                doc_id=doc_id,
                                pages=pages,
                                pasal=op.get("title", ""),
                                text="\n\n".join(txt_parts[:3]),
                                score=probs.get(op["node_id"], 0.1),
                                bab=bab_node.get("title", ""),
                                chunk_id=op.get("node_id", ""),
                            )
                        )

                except Exception as e:
                    logger.warning("D3 system_one failed: %s", e)
                    p_start = bab_node.get("start_index", 1)
                    p_end = bab_node.get("end_index", p_start)
                    pages = list(range(p_start, min(p_start + 4, p_end + 1)))
                    txt_parts = [p_map[p] for p in pages if p in p_map]
                    retrieved.append(
                        RetrievedChunk(
                            doc_id=doc_id,
                            pages=pages,
                            pasal=bab_node.get("title", ""),
                            text="\n\n".join(txt_parts[:3]),
                            score=0.5,
                            bab=bab_node.get("title", ""),
                            chunk_id=bab_node.get("node_id", ""),
                        )
                    )

        # Sort by score descending and deduplicate by chunk_id
        seen_ids = set()
        deduped: list[RetrievedChunk] = []
        for c in sorted(retrieved, key=lambda x: x.score, reverse=True):
            if c.chunk_id not in seen_ids:
                seen_ids.add(c.chunk_id)
                deduped.append(c)

        return deduped[:top_k]

    def answer(self, query: str, top_k: int = 5) -> RAGResponse:
        meter = TokenUsage()
        with token_meter() as m:
            chunks = self.retrieve(query, top_k=top_k)
            context = format_context(chunks)

            system_prompt = (
                "Anda adalah konsultan hukum Indonesia.\n"
                "Jawab pertanyaan berdasarkan konteks pohon peraturan berikut.\n\n"
                f"KONTEKS:\n{context}"
            )
            response = chat_completion(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
            )

        meter.input_tokens = m.input_tokens
        meter.output_tokens = m.output_tokens
        meter.total_tokens = m.total_tokens
        meter.latency_seconds = m.latency_seconds
        meter.model_usage = dict(m.model_usage)
        citations = [
            Citation(
                doc_id=c.doc_id,
                pasal=c.pasal,
                page_number=c.pages[0] if c.pages else 1,
                relevance_score=c.score,
            )
            for c in chunks
        ]
        return RAGResponse(
            answer=response.content.strip(),
            citations=citations,
            usage=meter,
            retrieved_chunks=chunks,
        )


def retrieve(
    query: str,
    top_k: int = 5,
    model_name: str | None = None,
    adapter_path: str | Path | None = None,
) -> list[RetrievedChunk]:
    return D3Navigator(model_name=model_name, adapter_path=adapter_path).retrieve(
        query, top_k=top_k
    )


def answer(
    query: str,
    top_k: int = 5,
    model_name: str | None = None,
    adapter_path: str | Path | None = None,
) -> RAGResponse:
    return D3Navigator(model_name=model_name, adapter_path=adapter_path).answer(query, top_k=top_k)
