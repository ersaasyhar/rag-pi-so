"""Legal structure parser for Indonesian regulations (UU 12/2011 hierarchy).

Parses documents into:
Judul -> Pembukaan -> Batang Tubuh (BAB -> Bagian -> Pasal -> Ayat) -> Penutup -> Penjelasan -> Lampiran.
Emits structural tree and article-to-page index in data/parsed/{doc_id}.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
PARSED_DIR = DATA_DIR / "parsed"

ROMAN_MAP = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_to_int(raw: str) -> int:
    val = 0
    s = raw.strip().upper()
    for i, c in enumerate(s):
        if c not in ROMAN_MAP:
            return 0
        v = ROMAN_MAP[c]
        if i + 1 < len(s) and ROMAN_MAP.get(s[i + 1], 0) > v:
            val -= v
        else:
            val += v
    return val


def normalize_pasal_id(raw: str) -> str:
    """Normalize 'Pasal 1', 'Pasal I', 'pasal 01' -> 'Pasal 1'."""
    raw = raw.strip()
    m = re.match(r"^Pasal\s+([0-9]+|[IVXLCDM]+)", raw, re.I)
    if not m:
        return raw
    val = m.group(1)
    if val.isdigit():
        return f"Pasal {int(val)}"
    r_int = roman_to_int(val)
    if r_int > 0:
        return f"Pasal {r_int}"
    return f"Pasal {val}"


def parse_legal_structure(doc_data: dict[str, Any]) -> dict[str, Any]:
    doc_id = doc_data["doc_id"]
    metadata = doc_data.get("metadata", {})
    pages = doc_data.get("pages", [])

    # Identify boundary of Penutup / Penjelasan / Lampiran
    penjelasan_start = None
    lampiran_start = None
    batang_tubuh_end = None

    for p in pages:
        pg = p["page_number"]
        txt = p["text"]
        if pg > 5 and not penjelasan_start:
            if re.search(r"PENJE\S*ASAN\s+ATAS", txt, re.I) or re.search(
                r"^PENJELASAN\s*$", txt, re.M | re.I
            ):
                penjelasan_start = pg
        if pg > 5 and not lampiran_start:
            if re.search(r"^LAMPIRAN\s*", txt, re.M | re.I):
                lampiran_start = pg

    if penjelasan_start:
        batang_tubuh_end = penjelasan_start - 1
    elif lampiran_start:
        batang_tubuh_end = lampiran_start - 1
    else:
        batang_tubuh_end = len(pages)

    # Collect Batang Tubuh lines with page numbers
    lines_with_page: list[tuple[int, str]] = []
    pembukaan_lines: list[tuple[int, str]] = []
    penjelasan_lines: list[tuple[int, str]] = []
    lampiran_lines: list[tuple[int, str]] = []

    for p in pages:
        pg = p["page_number"]
        for line in p["text"].split("\n"):
            sline = line.strip()
            if not sline:
                continue
            if lampiran_start and pg >= lampiran_start:
                lampiran_lines.append((pg, sline))
            elif penjelasan_start and pg >= penjelasan_start:
                penjelasan_lines.append((pg, sline))
            elif pg <= batang_tubuh_end:
                lines_with_page.append((pg, sline))

    # Parse Batang Tubuh into BAB and Pasal
    bab_nodes: list[dict[str, Any]] = []
    pasal_map: dict[str, dict[str, Any]] = {}

    current_bab = {
        "node_id": "bab_0",
        "type": "bab",
        "identifier": "KETENTUAN AWAL",
        "title": "Ketentuan Awal",
        "page_start": 1,
        "page_end": 1,
        "nodes": [],
    }
    current_pasal: dict[str, Any] | None = None

    for pg, line in lines_with_page:
        # Check BAB header
        m_bab = re.match(r"^BAB\s+([IVXLCDM]+)(.*)", line, re.I)
        if m_bab:
            # Finalize current pasal
            if current_pasal:
                current_pasal["text"] = "\n".join(current_pasal.pop("_lines"))
                current_bab["nodes"].append(current_pasal)
                current_pasal = None

            if current_bab["nodes"]:
                current_bab["page_end"] = (
                    current_bab["nodes"][-1]["page_end"] if current_bab["nodes"] else pg
                )
                bab_nodes.append(current_bab)

            bab_id = f"bab_{m_bab.group(1).upper()}"
            current_bab = {
                "node_id": bab_id,
                "type": "bab",
                "identifier": f"BAB {m_bab.group(1).upper()}",
                "title": m_bab.group(2).strip() or f"BAB {m_bab.group(1).upper()}",
                "page_start": pg,
                "page_end": pg,
                "nodes": [],
            }
            continue

        # Check Pasal header
        m_pasal = re.match(r"^Pasal\s+([0-9]+|[IVXLCDM]+)\s*$", line, re.I)
        if m_pasal:
            if current_pasal:
                current_pasal["text"] = "\n".join(current_pasal.pop("_lines"))
                current_bab["nodes"].append(current_pasal)

            norm_id = normalize_pasal_id(line)
            current_pasal = {
                "node_id": f"{current_bab['node_id']}_{norm_id.lower().replace(' ', '_')}",
                "type": "pasal",
                "identifier": norm_id,
                "title": norm_id,
                "bab": current_bab["identifier"],
                "page_start": pg,
                "page_end": pg,
                "pages": [pg],
                "ayats": [],
                "_lines": [line],
            }
            if norm_id not in pasal_map:
                pasal_map[norm_id] = current_pasal
            continue

        # Accumulate lines
        if current_pasal:
            current_pasal["page_end"] = pg
            if pg not in current_pasal["pages"]:
                current_pasal["pages"].append(pg)
            current_pasal["_lines"].append(line)

            # Check for Ayat marker "(1)", "(2)"
            m_ayat = re.match(r"^\((\d+)\)\s*(.*)", line)
            if m_ayat:
                current_pasal["ayats"].append(line)
        else:
            if not current_bab["nodes"]:
                pembukaan_lines.append((pg, line))

    # Close remaining
    if current_pasal:
        current_pasal["text"] = "\n".join(current_pasal.pop("_lines"))
        current_bab["nodes"].append(current_pasal)
    if current_bab["nodes"]:
        current_bab["page_end"] = current_bab["nodes"][-1]["page_end"]
        bab_nodes.append(current_bab)

    # Clean up pasal_map references
    for p_id, p_info in list(pasal_map.items()):
        p_info.pop("_lines", None)

    # Top-level structure
    structure = [
        {
            "node_id": "pembukaan",
            "type": "pembukaan",
            "title": "Pembukaan",
            "page_start": 1,
            "page_end": bab_nodes[0]["page_start"] if bab_nodes else 1,
            "text": "\n".join(line for _, line in pembukaan_lines),
        },
        *bab_nodes,
    ]

    if penjelasan_lines:
        structure.append(
            {
                "node_id": "penjelasan",
                "type": "penjelasan",
                "title": "Penjelasan",
                "page_start": penjelasan_start or 1,
                "page_end": (lampiran_start - 1) if lampiran_start else len(pages),
                "text": "\n".join(line for _, line in penjelasan_lines[:500]),
            }
        )

    if lampiran_lines:
        structure.append(
            {
                "node_id": "lampiran",
                "type": "lampiran",
                "title": "Lampiran",
                "page_start": lampiran_start or len(pages),
                "page_end": len(pages),
                "text": "\n".join(line for _, line in lampiran_lines[:500]),
            }
        )

    result = {
        "doc_id": doc_id,
        "metadata": metadata,
        "num_pages": len(pages),
        "total_articles": len(pasal_map),
        "structure": structure,
        "pasal_map": {
            k: {
                "pasal": v["identifier"],
                "bab": v["bab"],
                "page_start": v["page_start"],
                "page_end": v["page_end"],
                "pages": v["pages"],
                "ayats_count": len(v.get("ayats", [])),
                "text": v["text"],
            }
            for k, v in pasal_map.items()
        },
    }

    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PARSED_DIR / f"{doc_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def parse_all() -> list[dict[str, Any]]:
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    files = list(EXTRACTED_DIR.glob("*.json"))
    print(f"Parsing legal structure for {len(files)} extracted files...")
    results: list[dict[str, Any]] = []
    for f in sorted(files):
        with open(f, "r", encoding="utf-8") as fp:
            doc_data = json.load(fp)
        res = parse_legal_structure(doc_data)
        results.append(res)
    print(f"Parsed {len(results)} documents successfully.")
    return results


if __name__ == "__main__":
    parse_all()
