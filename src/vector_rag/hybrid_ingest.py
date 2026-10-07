"""Hybrid Qdrant ingestion (Dense + Sparse BM25 named vectors)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.llm_client import embed_texts, get_embedding_dimension
from src.sparse_bm25 import BM25Encoder

DATA_DIR = BASE_DIR / "data"
CHUNKS_DIR = DATA_DIR / "chunks"
QDRANT_DIR = DATA_DIR / "qdrant_v2"
COLLECTION_NAME = "indo_law_v2"


def get_qdrant_client(storage_path: Path = QDRANT_DIR) -> QdrantClient:
    storage_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(storage_path))


def ingest_corpus(force_recreate: bool = False) -> dict[str, Any]:
    client = get_qdrant_client()
    exists = client.collection_exists(COLLECTION_NAME)

    if exists:
        if force_recreate:
            print(f"Collection {COLLECTION_NAME} exists. Deleting for re-ingest...")
            client.delete_collection(COLLECTION_NAME)
        else:
            count = client.count(COLLECTION_NAME).count
            print(f"Collection {COLLECTION_NAME} already exists with {count} points.")
            client.close()
            return {"total_indexed": count, "time_seconds": 0.0}

    # Load all chunks
    chunk_files = sorted(CHUNKS_DIR.glob("*.jsonl"))
    all_chunks: list[dict[str, Any]] = []
    for cf in chunk_files:
        with open(cf, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    all_chunks.append(json.loads(line))

    # Fit and save BM25 encoder
    print("Fitting BM25 encoder on chunk corpus...")
    bm25 = BM25Encoder().fit([c["text"] for c in all_chunks])
    bm25.save()
    print(f"BM25 vocabulary size: {len(bm25.vocab)} terms.")

    # Create hybrid collection
    dim = get_embedding_dimension()
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )

    print(f"Indexing {len(all_chunks)} chunks into {COLLECTION_NAME} (dense + sparse)...")
    start_time = time.perf_counter()

    batch_size = 64
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]

        dense_vecs = embed_texts(texts)
        sparse_vecs = [bm25.encode(t) for t in texts]

        points = []
        for idx_in_batch, c in enumerate(batch):
            pt_id = i + idx_in_batch
            sp_indices, sp_values = sparse_vecs[idx_in_batch]
            vector_dict: dict[str, Any] = {
                "dense": dense_vecs[idx_in_batch],
            }
            if sp_indices:
                vector_dict["sparse"] = models.SparseVector(indices=sp_indices, values=sp_values)

            points.append(
                models.PointStruct(
                    id=pt_id,
                    vector=vector_dict,
                    payload=c,
                )
            )

        client.upsert(collection_name=COLLECTION_NAME, points=points)

    elapsed = time.perf_counter() - start_time
    total = client.count(COLLECTION_NAME).count
    client.close()

    print(f"Successfully indexed {total} hybrid points into {COLLECTION_NAME} in {elapsed:.2f}s.")
    return {"total_indexed": total, "time_seconds": elapsed}


if __name__ == "__main__":
    ingest_corpus(force_recreate=True)
