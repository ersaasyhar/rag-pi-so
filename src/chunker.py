"""Chunker for Indonesian legal documents.

Produces Pasal-level chunks with BAB hierarchy prefix, handles long Pasal splits at Ayat boundaries,
and structures Penjelasan/Lampiran as distinct chunk types.
Outputs: data/chunks/{doc_id}.jsonl
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PARSED_DIR = DATA_DIR / "parsed"
CHUNKS_DIR = DATA_DIR / "chunks"

MAX_CHUNK_CHARS = 3500


def format_chunk_text(
    jenis: str,
    nomor: str,
    tahun: str,
    judul: str,
    bab: str,
    pasal: str,
    body_text: str,
) -> str:
    """Format chunk text with official legal metadata header."""
    header = (
        f"DOKUMEN: {jenis.upper()} Nomor {nomor} Tahun {tahun} tentang {judul}\n"
        f"LOKASI: {bab} - {pasal}\n\n"
    )
    return header + body_text.strip()


def chunk_document(doc_id: str) -> list[dict[str, Any]]:
    parsed_file = PARSED_DIR / f"{doc_id}.json"
    if not parsed_file.exists():
        raise FileNotFoundError(f"Parsed file not found: {parsed_file}")

    with open(parsed_file, "r", encoding="utf-8") as f:
        doc = json.load(f)

    meta = doc.get("metadata", {})
    jenis = meta.get("jenis_peraturan", "PERATURAN")
    nomor = meta.get("nomor", "0")
    tahun = meta.get("tahun", "0000")
    judul = meta.get("judul", doc_id)

    chunks: list[dict[str, Any]] = []

    # 1. Pembukaan chunk
    pembukaan_nodes = [n for n in doc.get("structure", []) if n.get("type") == "pembukaan"]
    if pembukaan_nodes and pembukaan_nodes[0].get("text"):
        p_node = pembukaan_nodes[0]
        text_pembukaan = p_node["text"][:3000]
        chunk_text = (
            f"DOKUMEN: {jenis.upper()} Nomor {nomor} Tahun {tahun} tentang {judul}\n"
            f"LOKASI: PEMBUKAAN\n\n{text_pembukaan}"
        )
        chunks.append(
            {
                "chunk_id": f"{doc_id}_pembukaan",
                "doc_id": doc_id,
                "jenis": jenis,
                "nomor": nomor,
                "tahun": tahun,
                "judul": judul,
                "bab": "PEMBUKAAN",
                "pasal": "Pembukaan",
                "page_start": p_node.get("page_start", 1),
                "page_end": p_node.get("page_end", 1),
                "pages": list(range(p_node.get("page_start", 1), p_node.get("page_end", 1) + 1)),
                "chunk_type": "pembukaan",
                "text": chunk_text,
            }
        )

    # 2. Batang Tubuh (Pasal-level chunks)
    pasal_map = doc.get("pasal_map", {})
    for p_id, p_info in pasal_map.items():
        pasal_name = p_info.get("pasal", p_id)
        bab_name = p_info.get("bab", "BATANG TUBUH")
        body = p_info.get("text", "").strip()
        p_start = p_info.get("page_start", 1)
        p_end = p_info.get("page_end", p_start)
        pages = p_info.get("pages", [p_start])

        if not body:
            continue

        safe_p_id = re.sub(r"[^a-zA-Z0-9_]", "_", p_id.lower())

        if len(body) <= MAX_CHUNK_CHARS:
            formatted = format_chunk_text(jenis, nomor, tahun, judul, bab_name, pasal_name, body)
            chunks.append(
                {
                    "chunk_id": f"{doc_id}_{safe_p_id}",
                    "doc_id": doc_id,
                    "jenis": jenis,
                    "nomor": nomor,
                    "tahun": tahun,
                    "judul": judul,
                    "bab": bab_name,
                    "pasal": pasal_name,
                    "page_start": p_start,
                    "page_end": p_end,
                    "pages": pages,
                    "chunk_type": "pasal",
                    "text": formatted,
                }
            )
        else:
            # Long pasal: split at Ayat boundaries
            ayats = re.split(r"(?=\(\d+\))", body)
            ayats = [a.strip() for a in ayats if a.strip()]
            if len(ayats) <= 1:
                # Fallback: split by character limit
                step = 2500
                for part_idx, i in enumerate(range(0, len(body), step)):
                    sub_body = body[i : i + step]
                    formatted = format_chunk_text(
                        jenis,
                        nomor,
                        tahun,
                        judul,
                        bab_name,
                        f"{pasal_name} (Bagian {part_idx + 1})",
                        sub_body,
                    )
                    chunks.append(
                        {
                            "chunk_id": f"{doc_id}_{safe_p_id}_part{part_idx + 1}",
                            "doc_id": doc_id,
                            "jenis": jenis,
                            "nomor": nomor,
                            "tahun": tahun,
                            "judul": judul,
                            "bab": bab_name,
                            "pasal": pasal_name,
                            "page_start": p_start,
                            "page_end": p_end,
                            "pages": pages,
                            "chunk_type": "pasal",
                            "text": formatted,
                        }
                    )
            else:
                curr_group: list[str] = []
                curr_len = 0
                part_idx = 1
                for ayat in ayats:
                    if curr_len + len(ayat) > MAX_CHUNK_CHARS and curr_group:
                        sub_body = "\n".join(curr_group)
                        formatted = format_chunk_text(
                            jenis,
                            nomor,
                            tahun,
                            judul,
                            bab_name,
                            f"{pasal_name} (Bagian {part_idx})",
                            sub_body,
                        )
                        chunks.append(
                            {
                                "chunk_id": f"{doc_id}_{safe_p_id}_part{part_idx}",
                                "doc_id": doc_id,
                                "jenis": jenis,
                                "nomor": nomor,
                                "tahun": tahun,
                                "judul": judul,
                                "bab": bab_name,
                                "pasal": pasal_name,
                                "page_start": p_start,
                                "page_end": p_end,
                                "pages": pages,
                                "chunk_type": "pasal",
                                "text": formatted,
                            }
                        )
                        part_idx += 1
                        curr_group = [ayat]
                        curr_len = len(ayat)
                    else:
                        curr_group.append(ayat)
                        curr_len += len(ayat)

                if curr_group:
                    sub_body = "\n".join(curr_group)
                    formatted = format_chunk_text(
                        jenis,
                        nomor,
                        tahun,
                        judul,
                        bab_name,
                        f"{pasal_name} (Bagian {part_idx})",
                        sub_body,
                    )
                    chunks.append(
                        {
                            "chunk_id": f"{doc_id}_{safe_p_id}_part{part_idx}",
                            "doc_id": doc_id,
                            "jenis": jenis,
                            "nomor": nomor,
                            "tahun": tahun,
                            "judul": judul,
                            "bab": bab_name,
                            "pasal": pasal_name,
                            "page_start": p_start,
                            "page_end": p_end,
                            "pages": pages,
                            "chunk_type": "pasal",
                            "text": formatted,
                        }
                    )

    # 3. Penjelasan / Lampiran chunks
    for node in doc.get("structure", []):
        n_type = node.get("type")
        if n_type in ("penjelasan", "lampiran") and node.get("text"):
            text = node["text"]
            step = 2500
            for part_idx, i in enumerate(range(0, min(len(text), 15000), step)):
                sub_text = text[i : i + step]
                header = (
                    f"DOKUMEN: {jenis.upper()} Nomor {nomor} Tahun {tahun} tentang {judul}\n"
                    f"LOKASI: {node['title'].upper()} (Bagian {part_idx + 1})\n\n"
                )
                chunks.append(
                    {
                        "chunk_id": f"{doc_id}_{n_type}_part{part_idx + 1}",
                        "doc_id": doc_id,
                        "jenis": jenis,
                        "nomor": nomor,
                        "tahun": tahun,
                        "judul": judul,
                        "bab": node["title"].upper(),
                        "pasal": node["title"],
                        "page_start": node.get("page_start", 1),
                        "page_end": node.get("page_end", 1),
                        "pages": list(
                            range(node.get("page_start", 1), node.get("page_end", 1) + 1)
                        ),
                        "chunk_type": n_type,
                        "text": header + sub_text,
                    }
                )

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = CHUNKS_DIR / f"{doc_id}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    return chunks


def chunk_all() -> int:
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    parsed_files = list(PARSED_DIR.glob("*.json"))
    print(f"Chunking {len(parsed_files)} documents...")
    total_chunks = 0
    for pf in sorted(parsed_files):
        doc_id = pf.stem
        c = chunk_document(doc_id)
        total_chunks += len(c)
    print(f"Total chunks created: {total_chunks} across {len(parsed_files)} documents.")
    return total_chunks


if __name__ == "__main__":
    chunk_all()
