"""PageIndex tree index with TypeSafe AI Jev (System One) navigator."""

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

from typesafe_sdk import Choice, Score, TypeSafeClient

from src.context_formatter import format_context
from src.llm_client import TokenUsage, chat_completion, record_tokens, token_meter
from src.schema import Citation, RAGResponse, RetrievedChunk
from src.tree_reasoning.tree_index import build_tree_for_doc

logger = logging.getLogger(__name__)

DATA_DIR = BASE_DIR / "data"
MANIFEST_PATH = DATA_DIR / "corpus_manifest.json"
EXTRACTED_DIR = DATA_DIR / "extracted"

TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY", "").strip()


class JevNavigator:
    def __init__(self, api_key: str | None = None, manifest_path: Path = MANIFEST_PATH):
        self.api_key = api_key or TYPESAFE_API_KEY
        self.manifest_path = manifest_path
        self._manifest: list[dict[str, Any]] | None = None
        self.client: TypeSafeClient | None = None
        if self.api_key:
            try:
                self.client = TypeSafeClient(api_key=self.api_key)
            except Exception as e:
                logger.warning("Failed to initialize TypeSafeClient: %s", e)

    def is_available(self) -> bool:
        return self.client is not None

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
        """Navigate down a node using Jev Choice and Score with confidence-based fallback."""
        children = node.get("nodes", [])
        if not children:
            # Leaf node
            p_start = node.get("start_index", 1)
            p_end = node.get("end_index", p_start)
            return list(range(p_start, p_end + 1))

        if not self.client:
            # Fallback if no Jev API: return all pages in subtree
            p_start = node.get("start_index", 1)
            p_end = node.get("end_index", p_start)
            return list(range(p_start, p_end + 1))

        # Format options for Choice (max 255 options)
        options = {
            c["node_id"]: f"{c['title']}. {c.get('summary', '')[:100]}" for c in children[:200]
        }

        try:
            resp = self.client.system_one(
                state={"query": query, "current_section": node.get("title", "")},
                questions={
                    "next_branch": Choice(
                        instructions="Pilih bagian dokumen yang paling mungkin memuat jawaban pertanyaan.",
                        criteria=options,
                    ),
                    "relevance": Score(
                        instructions="Nilai relevansi bagian dokumen ini terhadap pertanyaan.",
                        criteria=[
                            "tidak relevan",
                            "sedikit relevan",
                            "cukup relevan",
                            "relevan",
                            "sangat tepat",
                        ],
                    ),
                },
            )
            # Record Jev tokens
            if hasattr(resp, "usage") and resp.usage:
                in_tok = getattr(resp.usage, "input_tokens", 0)
                out_tok = getattr(resp.usage, "output_tokens", 0)
                record_tokens(in_tok, out_tok, model="typesafe/jev-1.13")

            # Check confidence score
            conf = 1.0
            if hasattr(resp, "scores") and "relevance" in resp.scores:
                conf = getattr(resp.scores["relevance"], "confidence", 1.0)

            if conf < 0.60:
                logger.warning(
                    "Jev confidence below threshold (%.2f < 0.60) at node %s. Falling back to full subtree.",
                    conf,
                    node.get("node_id"),
                )
                p_start = node.get("start_index", 1)
                p_end = node.get("end_index", p_start)
                return list(range(p_start, p_end + 1))

            selected_id = resp.choices["next_branch"].choice
            next_node = next((c for c in children if c["node_id"] == selected_id), children[0])
            return self.navigate_node(query, next_node)

        except Exception as e:
            logger.warning("Jev system_one call failed: %s. Using subtree fallback.", e)
            p_start = node.get("start_index", 1)
            p_end = node.get("end_index", p_start)
            return list(range(p_start, p_end + 1))

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Retrieve chunks across corpus documents using adjusted Jev graph navigation."""
        if not self.is_available():
            logger.warning("TYPESAFE_API_KEY not set. Jev navigator unavailable.")
            return []

        candidate_docs = self._select_candidate_docs(query, max_docs=2)
        if not candidate_docs:
            candidate_docs = self.manifest[:2]

        retrieved: list[RetrievedChunk] = []

        for doc in candidate_docs:
            doc_id = doc["doc_id"]
            tree = build_tree_for_doc(doc_id)
            root_nodes = [n for n in tree.get("structure", []) if n.get("nodes")]
            if not root_nodes:
                continue

            # Load extracted text map
            ext_path = EXTRACTED_DIR / f"{doc_id}.json"
            p_map: dict[int, str] = {}
            if ext_path.exists():
                with open(ext_path, "r", encoding="utf-8") as fp:
                    ext = json.load(fp)
                p_map = {p["page_number"]: p["text"] for p in ext.get("pages", [])}

            # Filter candidate BABs by query term matches
            q_words = [w for w in query.lower().split() if len(w) > 3]
            scored_babs = []
            for b in root_nodes:
                txt = (
                    b.get("title", "")
                    + " "
                    + " ".join(c.get("summary", "") for c in b.get("nodes", []))
                ).lower()
                matches = sum(1 for w in q_words if w in txt)
                scored_babs.append((matches, b))

            scored_babs.sort(key=lambda x: x[0], reverse=True)
            top_babs = [b for s, b in scored_babs if s > 0][:2]
            if not top_babs:
                top_babs = root_nodes[:2]

            for bab_node in top_babs:
                child_pasals = bab_node.get("nodes", [])
                if not child_pasals:
                    continue

                p_options = {
                    p["node_id"]: f"{p['title']}: {p.get('summary', '')[:80]}"
                    for p in child_pasals[:30]
                }

                try:
                    assert self.client is not None
                    resp = self.client.system_one(
                        state={"query": query, "bab": bab_node.get("title", "")},
                        questions={
                            "pasal": Choice(
                                instructions="Pilih Pasal yang paling tepat memuat jawaban.",
                                criteria=p_options,
                            ),
                            "relevance": Score(
                                instructions="Nilai tingkat kepastian relevansi pasal ini.",
                                criteria=["rendah", "sedang", "tinggi"],
                            ),
                        },
                    )

                    # Record Jev tokens
                    if hasattr(resp, "usage") and resp.usage:
                        in_tok = getattr(resp.usage, "input_tokens", 0)
                        out_tok = getattr(resp.usage, "output_tokens", 0)
                        record_tokens(in_tok, out_tok, model="typesafe/jev-1.13")

                    conf = 1.0
                    if hasattr(resp, "choices") and "pasal" in resp.choices:
                        conf = getattr(resp.choices["pasal"], "confidence", 1.0)

                    chosen_id = resp.choices["pasal"].choice if "pasal" in resp.choices else None
                    chosen_pasal = next(
                        (p for p in child_pasals if p["node_id"] == chosen_id), None
                    )

                    if conf >= 0.60 and chosen_pasal:
                        p_start = chosen_pasal.get("start_index", 1)
                        p_end = chosen_pasal.get("end_index", p_start)
                        pages = list(range(p_start, p_end + 1))
                        pasal_title = chosen_pasal.get("title", "")
                        chunk_id = chosen_pasal.get("node_id", "")
                    else:
                        p_start = bab_node.get("start_index", 1)
                        p_end = bab_node.get("end_index", p_start)
                        pages = list(range(p_start, min(p_start + 4, p_end + 1)))
                        pasal_title = bab_node.get("title", "")
                        chunk_id = bab_node.get("node_id", "")

                    txt_parts = [p_map[p] for p in pages if p in p_map]
                    retrieved.append(
                        RetrievedChunk(
                            doc_id=doc_id,
                            pages=pages,
                            pasal=pasal_title,
                            text="\n\n".join(txt_parts[:3]),
                            score=conf,
                            bab=bab_node.get("title", ""),
                            chunk_id=chunk_id,
                        )
                    )
                except Exception as e:
                    logger.warning("Jev system_one failed: %s", e)

        return retrieved[:top_k]

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


def retrieve(query: str, top_k: int = 5) -> list[RetrievedChunk]:
    return JevNavigator().retrieve(query, top_k=top_k)


def answer(query: str, top_k: int = 5) -> RAGResponse:
    return JevNavigator().answer(query, top_k=top_k)
