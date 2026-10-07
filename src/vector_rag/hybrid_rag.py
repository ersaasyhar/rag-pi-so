"""Hybrid RAG using Dense + BM25 Qdrant retriever."""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.context_formatter import format_context
from src.llm_client import TokenUsage, chat_completion, token_meter
from src.schema import Citation, RAGResponse
from src.vector_rag.hybrid_retriever import Phase2QdrantRetriever


class Phase2QdrantRAG:
    def __init__(self, retriever: Phase2QdrantRetriever | None = None):
        self.retriever = retriever or Phase2QdrantRetriever()

    def answer(self, query: str, top_k: int = 5) -> RAGResponse:
        meter = TokenUsage()
        with token_meter() as m:
            chunks = self.retriever.retrieve(query, top_k=top_k)
            context = format_context(chunks)

            system_prompt = (
                "Anda adalah konsultan hukum Indonesia ahli peraturan perundang-undangan.\n"
                "Jawab pertanyaan pengguna dengan ketat hanya berdasarkan KONTEKS peraturan di bawah.\n"
                "Format jawaban:\n"
                "1. Jawaban Langsung (ringkas, padat).\n"
                "2. Dasar Hukum (cantumkan nama peraturan, nomor, tahun, pasal, dan halaman).\n\n"
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
        citations: list[Citation] = []
        for c in chunks:
            citations.append(
                Citation(
                    doc_id=c.doc_id,
                    pasal=c.pasal,
                    page_number=c.pages[0] if c.pages else 1,
                    relevance_score=c.score,
                )
            )

        return RAGResponse(
            answer=response.content.strip(),
            citations=citations,
            usage=meter,
            retrieved_chunks=chunks,
        )

    def close(self) -> None:
        self.retriever.close()


def answer(query: str, top_k: int = 5) -> RAGResponse:
    rag = Phase2QdrantRAG()
    try:
        return rag.answer(query, top_k=top_k)
    finally:
        rag.close()
