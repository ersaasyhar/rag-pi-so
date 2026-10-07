"""Dense vector Qdrant ingestion."""

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

DATA_DIR = BASE_DIR / "data"
CHUNKS_DIR = DATA_DIR / "chunks"
QDRANT_DIR = DATA_DIR / "qdrant_v1"
COLLECTION_NAME = "indo_law_v1"


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

    dim = get_embedding_dimension()
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )

    # Create payload indexes
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="doc_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="chunk_type",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

    # Read all chunks
    chunk_files = sorted(CHUNKS_DIR.glob("*.jsonl"))
    all_chunks: list[dict[str, Any]] = []
    for cf in chunk_files:
        with open(cf, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    all_chunks.append(json.loads(line))

    print(f"Embedding and indexing {len(all_chunks)} chunks into {COLLECTION_NAME}...")
    start_time = time.perf_counter()

    batch_size = 64
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = embed_texts(texts)

        points = []
        for idx_in_batch, c in enumerate(batch):
            pt_id = i + idx_in_batch
            points.append(
                models.PointStruct(
                    id=pt_id,
                    vector=embeddings[idx_in_batch],
                    payload=c,
                )
            )

        client.upsert(collection_name=COLLECTION_NAME, points=points)

    elapsed = time.perf_counter() - start_time
    total = client.count(COLLECTION_NAME).count
    client.close()

    print(f"Successfully indexed {total} points into {COLLECTION_NAME} in {elapsed:.2f}s.")
    return {"total_indexed": total, "time_seconds": elapsed}


if __name__ == "__main__":
    ingest_corpus(force_recreate=True)
