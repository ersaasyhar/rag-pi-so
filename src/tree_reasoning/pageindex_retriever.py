"""Tree index retrieval with LLM reasoning navigator."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.context_formatter import format_context
from src.llm_client import TokenUsage, chat_completion, retrieval_completion, token_meter
from src.schema import Citation, RAGResponse, RetrievedChunk
from src.tree_reasoning.tree_index import build_tree_for_doc

DATA_DIR = BASE_DIR / "data"
MANIFEST_PATH = DATA_DIR / "corpus_manifest.json"
EXTRACTED_DIR = DATA_DIR / "extracted"


class PageIndexLLMRetriever:
    def __init__(self, manifest_path: Path = MANIFEST_PATH):
        self.manifest_path = manifest_path
        self._manifest: list[dict[str, Any]] | None = None

    @property
    def manifest(self) -> list[dict[str, Any]]:
        if self._manifest is None:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                self._manifest = json.load(f)
        return self._manifest

    def _select_candidate_docs(self, query: str, max_docs: int = 3) -> list[dict[str, Any]]:
        """Select top candidate documents based on query keywords and landmark status."""
        q_lower = query.lower()
        scored: list[tuple[float, dict[str, Any]]] = []

        # Domain term hints
        is_crime = any(
            w in q_lower for w in ["pidana", "pencurian", "pembunuhan", "kuhp", "penjara", "denda"]
        )
        is_pdp = any(
            w in q_lower for w in ["pribadi", "data pribadi", "kebocoran", "pemrosesan data"]
        )
        is_ciptaker = any(
            w in q_lower
            for w in ["kerja", "cipta kerja", "pesangon", "upah", "phk", "tenaga kerja"]
        )
        is_pt = any(
            w in q_lower for w in ["perseroan", "direksi", "komisaris", "rups", "saham", "pt "]
        )
        is_p3 = any(
            w in q_lower
            for w in ["peraturan", "undang-undang", "hierarki", "materi muatan", "pengundangan"]
        )
        is_kpu = any(
            w in q_lower for w in ["pemilu", "kampanye", "dana kampanye", "pemilihan umum", "kpu"]
        )

        for doc in self.manifest:
            score = 0.0
            d_id = doc.get("doc_id", "")
            title = doc.get("judul", "").lower()
            subject = doc.get("tentang", "").lower()

            # Landmark boost
            if doc.get("is_landmark"):
                score += 2.0

            # Direct title/subject overlap
            for word in q_lower.split():
                if len(word) > 3:
                    if word in title:
                        score += 3.0
                    if word in subject:
                        score += 2.0

            # Domain heuristics
            if is_crime and "uu_1_2023" in d_id:
                score += 15.0
            if is_pdp and "uu_27_2022" in d_id:
                score += 15.0
            if is_ciptaker and "uu_6_2023" in d_id:
                score += 15.0
            if is_pt and "uu_40_2007" in d_id:
                score += 15.0
            if is_p3 and "uu_12_2011" in d_id:
                score += 15.0
            if is_kpu and "pkpu" in d_id:
                score += 10.0

            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:max_docs]]

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Navigate PageIndex trees using single-pass LLM reasoning down branches."""
        candidate_docs = self._select_candidate_docs(query, max_docs=1)
        retrieved_chunks: list[RetrievedChunk] = []

        for doc in candidate_docs:
            doc_id = doc["doc_id"]
            tree = build_tree_for_doc(doc_id)
            root_nodes = tree.get("structure", [])
            if not root_nodes:
                continue

            # Build concise tree index representation
            tree_index_lines = []
            all_leaves: dict[str, dict[str, Any]] = {}
            for b_node in root_nodes:
                if b_node.get("type") in ("penutup", "lampiran"):
                    continue
                b_title = b_node["title"]
                child_pasals = b_node.get("nodes", [])
                p_strs = []
                for cp in child_pasals:
                    p_id = cp["node_id"]
                    all_leaves[p_id] = cp
                    p_strs.append(f"{cp['title']}(ID:{p_id})")
                tree_index_lines.append(f"[{b_title}] -> {', '.join(p_strs[:10])}")

            prompt = (
                f"Dokumen: {doc['judul']}\n"
                f"Pertanyaan: '{query}'\n\n"
                "Struktur Bab & Pasal:\n" + "\n".join(tree_index_lines[:25]) + "\n\n"
                "Pilih 1-3 ID Pasal yang paling relevan menjawab pertanyaan.\n"
                'Kembalikan HANYA JSON array string, contoh: ["bab_1_pasal_1"]'
            )

            try:
                resp = retrieval_completion([{"role": "user", "content": prompt}], temperature=0.0)
                m = re.search(r"\[.*?\]", resp.content, re.S)
                chosen_ids = json.loads(m.group(0)) if m else []
            except Exception:
                chosen_ids = []

            leaf_nodes = [all_leaves[pid] for pid in chosen_ids if pid in all_leaves]
            if not leaf_nodes:
                leaf_nodes = list(all_leaves.values())[:top_k]
            # Read text from extracted JSON for selected leaves
            extracted_path = EXTRACTED_DIR / f"{doc_id}.json"
            page_text_map: dict[int, str] = {}
            if extracted_path.exists():
                with open(extracted_path, "r", encoding="utf-8") as fp:
                    ext_data = json.load(fp)
                page_text_map = {p["page_number"]: p["text"] for p in ext_data.get("pages", [])}

            for idx, leaf in enumerate(leaf_nodes):
                p_start = leaf.get("start_index", 1)
                p_end = leaf.get("end_index", p_start)
                pages = list(range(p_start, p_end + 1))
                body_parts = [page_text_map.get(pg, "") for pg in pages if pg in page_text_map]
                full_text = "\n\n".join(body_parts) if body_parts else leaf.get("summary", "")

                retrieved_chunks.append(
                    RetrievedChunk(
                        doc_id=doc_id,
                        pages=pages,
                        pasal=leaf.get("title", ""),
                        text=full_text,
                        score=1.0 - (idx * 0.05),
                        bab=doc["judul"],
                        chunk_id=leaf.get("node_id", ""),
                        chunk_type="pasal",
                    )
                )

        return retrieved_chunks[:top_k]

    def answer(self, query: str, top_k: int = 5) -> RAGResponse:
        meter = TokenUsage()
        with token_meter() as m:
            chunks = self.retrieve(query, top_k=top_k)
            context = format_context(chunks)

            system_prompt = (
                "Anda adalah konsultan hukum Indonesia ahli peraturan perundang-undangan.\n"
                "Jawab pertanyaan berdasarkan konteks pohon peraturan berikut.\n"
                "Sertakan rujukan nama dokumen, pasal, dan halaman sumber.\n\n"
                f"KONTEKS:\n{context}"
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]
            response = chat_completion(messages, temperature=0.0)

        meter.input_tokens = m.input_tokens
        meter.output_tokens = m.output_tokens
        meter.total_tokens = m.total_tokens
        meter.latency_seconds = m.latency_seconds
        meter.model_usage = dict(m.model_usage)
        citations: list[Citation] = [
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
    return PageIndexLLMRetriever().retrieve(query, top_k=top_k)


def answer(query: str, top_k: int = 5) -> RAGResponse:
    return PageIndexLLMRetriever().answer(query, top_k=top_k)
