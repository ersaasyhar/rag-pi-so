"""Hybrid retriever (Dense + Sparse BM25 via Qdrant RRF fusion)."""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.llm_client import embed_texts
from src.schema import RetrievedChunk
from src.sparse_bm25 import BM25Encoder

QDRANT_DIR = BASE_DIR / "data" / "qdrant_v2"
COLLECTION_NAME = "indo_law_v2"


class Phase2QdrantRetriever:
    def __init__(self, storage_path: Path = QDRANT_DIR):
        self.storage_path = storage_path
        self._client: QdrantClient | None = None
        self._bm25: BM25Encoder | None = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(path=str(self.storage_path))
        return self._client

    @property
    def bm25(self) -> BM25Encoder:
        if self._bm25 is None:
            self._bm25 = BM25Encoder.load()
        return self._bm25

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Perform hybrid search (dense + BM25 sparse) with Reciprocal Rank Fusion."""
        query_dense = embed_texts([query])[0]
        sp_indices, sp_values = self.bm25.encode(query, is_query=True)

        prefetch_queries = [
            models.Prefetch(query=query_dense, using="dense", limit=20),
        ]
        if sp_indices:
            prefetch_queries.append(
                models.Prefetch(
                    query=models.SparseVector(indices=sp_indices, values=sp_values),
                    using="sparse",
                    limit=20,
                )
            )

        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=prefetch_queries,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
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
    retriever = Phase2QdrantRetriever()
    try:
        return retriever.retrieve(query, top_k=top_k)
    finally:
        retriever.close()
