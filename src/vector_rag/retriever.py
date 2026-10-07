"""Dense vector retriever."""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from qdrant_client import QdrantClient

from src.llm_client import embed_texts
from src.schema import RetrievedChunk

QDRANT_DIR = BASE_DIR / "data" / "qdrant_v1"
COLLECTION_NAME = "indo_law_v1"


class Phase1Retriever:
    def __init__(self, storage_path: Path = QDRANT_DIR):
        self.storage_path = storage_path
        self._client: QdrantClient | None = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(path=str(self.storage_path))
        return self._client

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Perform dense vector search against indo_law_v1."""
        query_vector = embed_texts([query])[0]
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        chunks: list[RetrievedChunk] = []
        for pt in results.points:
            p = pt.payload or {}
            pages = p.get("pages") or [p.get("page_start", 1)]
            chunks.append(
                RetrievedChunk(
                    doc_id=p.get("doc_id", ""),
                    pages=pages,
                    pasal=p.get("pasal", ""),
                    text=p.get("text", ""),
                    score=float(pt.score or 0.0),
                    bab=p.get("bab", ""),
                    chunk_id=p.get("chunk_id", ""),
                    chunk_type=p.get("chunk_type", "pasal"),
                    metadata=p,
                )
            )
        return chunks

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


def retrieve(query: str, top_k: int = 5) -> list[RetrievedChunk]:
    retriever = Phase1Retriever()
    try:
        return retriever.retrieve(query, top_k=top_k)
    finally:
        retriever.close()
