"""Tree index builder for PageIndex navigation.

Builds and caches hierarchical document trees (BAB -> Pasal -> Ayat) in data/pageindex/{doc_id}_tree.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
PARSED_DIR = DATA_DIR / "parsed"
PAGEINDEX_DIR = DATA_DIR / "pageindex"
PDF_DIR = DATA_DIR / "pdfs"


def build_tree_for_doc(doc_id: str, force: bool = False) -> dict[str, Any]:
    PAGEINDEX_DIR.mkdir(parents=True, exist_ok=True)
    tree_file = PAGEINDEX_DIR / f"{doc_id}_tree.json"

    if tree_file.exists() and not force:
        with open(tree_file, "r", encoding="utf-8") as f:
            return json.load(f)

    parsed_file = PARSED_DIR / f"{doc_id}.json"
    if not parsed_file.exists():
        raise FileNotFoundError(f"Parsed document not found: {parsed_file}")

    with open(parsed_file, "r", encoding="utf-8") as f:
        doc = json.load(f)

    meta = doc.get("metadata", {})
    title = meta.get("judul", doc_id)

    # Convert parsed structure to PageIndex standard tree format
    tree_nodes: list[dict[str, Any]] = []

    for node in doc.get("structure", []):
        _ = node.get("type")
        n_id = node.get("node_id", "")
        n_title = node.get("title", "")
        p_start = node.get("page_start", 1)
        p_end = node.get("page_end", p_start)
        raw_text = node.get("text", "")

        children: list[dict[str, Any]] = []
        for child in node.get("nodes", []):
            c_start = child.get("page_start", p_start)
            c_end = child.get("page_end", c_start)
            c_text = child.get("text", "")
            summary = c_text[:250].replace("\n", " ") if c_text else child.get("title", "")
            children.append(
                {
                    "node_id": child.get("node_id", ""),
                    "title": child.get("title", ""),
                    "start_index": c_start,
                    "end_index": c_end,
                    "summary": summary,
                    "nodes": [],
                }
            )

        summary = raw_text[:250].replace("\n", " ") if raw_text else n_title
        tree_nodes.append(
            {
                "node_id": n_id,
                "title": n_title,
                "start_index": p_start,
                "end_index": p_end,
                "summary": summary,
                "nodes": children,
            }
        )

    tree_data = {
        "doc_id": doc_id,
        "title": title,
        "num_pages": doc.get("num_pages", 1),
        "structure": tree_nodes,
    }

    with open(tree_file, "w", encoding="utf-8") as f:
        json.dump(tree_data, f, indent=2, ensure_ascii=False)

    return tree_data


def build_all_trees() -> int:
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    files = list(PARSED_DIR.glob("*.json"))
    print(f"Building PageIndex trees for {len(files)} documents...")
    for f in sorted(files):
        build_tree_for_doc(f.stem)
    print("All trees built and cached in data/pageindex/.")
    return len(files)


if __name__ == "__main__":
    build_all_trees()
