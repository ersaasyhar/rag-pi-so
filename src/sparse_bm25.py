"""BM25 sparse vector tokenizer and weight calculator for Qdrant hybrid search."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "data" / "bm25_model.json"


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization for legal terms."""
    return re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text.lower())


class BM25Encoder:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vocab: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.avg_dl: float = 1.0
        self.num_docs: int = 0

    def fit(self, documents: list[str]) -> BM25Encoder:
        self.num_docs = len(documents)
        df: Counter[str] = Counter()
        total_len = 0

        doc_token_sets: list[list[str]] = []
        for doc in documents:
            tokens = tokenize(doc)
            total_len += len(tokens)
            doc_token_sets.append(tokens)
            for t in set(tokens):
                df[t] += 1

        self.avg_dl = total_len / max(1, self.num_docs)

        # Build vocabulary sorted by frequency
        self.vocab = {}
        self.idf = {}
        for token, count in df.items():
            token_id = len(self.vocab)
            self.vocab[token] = token_id
            # BM25 standard IDF with floor 0.1
            val = math.log((self.num_docs - count + 0.5) / (count + 0.5) + 1.0)
            self.idf[token] = max(0.1, val)

        return self

    def encode(self, text: str, is_query: bool = False) -> tuple[list[int], list[float]]:
        tokens = tokenize(text)
        if not tokens:
            return [], []

        tf = Counter(tokens)
        dl = len(tokens)

        indices: list[int] = []
        values: list[float] = []

        for token, count in tf.items():
            if token not in self.vocab:
                continue
            idx = self.vocab[token]
            token_idf = self.idf[token]

            if is_query:
                # Query term weight
                weight = token_idf
            else:
                # Document term weight
                num = count * (self.k1 + 1.0)
                denom = count + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_dl))
                weight = token_idf * (num / denom)

            indices.append(idx)
            values.append(round(weight, 4))

        return indices, values

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "k1": self.k1,
            "b": self.b,
            "avg_dl": self.avg_dl,
            "num_docs": self.num_docs,
            "vocab": self.vocab,
            "idf": self.idf,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> BM25Encoder:
        if not path.exists():
            raise FileNotFoundError(f"BM25 model not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        encoder = cls(k1=data.get("k1", 1.5), b=data.get("b", 0.75))
        encoder.avg_dl = data["avg_dl"]
        encoder.num_docs = data["num_docs"]
        encoder.vocab = data["vocab"]
        encoder.idf = data["idf"]
        return encoder
