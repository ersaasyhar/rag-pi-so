"""Dual-format extractor: PDF to structured JSON and Markdown with page markers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pymupdf

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
TEXTS_DIR = DATA_DIR / "texts"
MANIFEST_PATH = DATA_DIR / "corpus_manifest.json"


def extract_pdf(
    pdf_path: str | Path,
    doc_id: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract pages from PDF with 1-based indexing, emit JSON and Markdown."""
    pdf_path = Path(pdf_path)
    if not pdf_path.is_absolute():
        pdf_path = BASE_DIR / pdf_path

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = pymupdf.open(pdf_path)
    pages_data: list[dict[str, Any]] = []
    md_sections: list[str] = []

    for idx, page in enumerate(doc):
        page_num = idx + 1  # 1-based indexing
        text = page.get_text("text").strip()

        if len(text) < 50:
            logger.warning(
                "Low text density doc=%s page=%d (length=%d)", doc_id, page_num, len(text)
            )

        pages_data.append({"page_number": page_num, "text": text})
        md_sections.append(f"<!-- PAGE {page_num} -->\n\n{text}")

    doc.close()

    result = {
        "doc_id": doc_id,
        "metadata": metadata or {},
        "num_pages": len(pages_data),
        "pages": pages_data,
    }

    # Save structured JSON
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    out_json = EXTRACTED_DIR / f"{doc_id}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Save Markdown with page markers
    TEXTS_DIR.mkdir(parents=True, exist_ok=True)
    out_md = TEXTS_DIR / f"{doc_id}.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n\n".join(md_sections) + "\n")

    return result


def extract_all(manifest_path: str | Path = MANIFEST_PATH) -> list[dict[str, Any]]:
    """Extract all documents listed in corpus manifest."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    results: list[dict[str, Any]] = []
    print(f"Extracting {len(items)} documents...")
    for item in items:
        doc_id = item["doc_id"]
        pdf_path = item["pdf_path"]
        print(f"  Extracting {doc_id} ({item.get('num_pages', 0)} pages)...")
        res = extract_pdf(pdf_path, doc_id=doc_id, metadata=item)
        results.append(res)

    print(f"Successfully extracted {len(results)} documents.")
    return results


if __name__ == "__main__":
    extract_all()
