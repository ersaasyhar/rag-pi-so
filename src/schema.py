"""Shared schema definitions for retrieval, RAG, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.llm_client import TokenUsage


@dataclass
class RetrievedChunk:
    doc_id: str
    pages: list[int]
    pasal: str
    text: str
    score: float
    bab: str = ""
    chunk_id: str = ""
    chunk_type: str = "pasal"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Citation:
    doc_id: str
    pasal: str
    page_number: int
    relevance_score: float


@dataclass
class RAGResponse:
    answer: str
    citations: list[Citation]
    usage: TokenUsage
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
