"""Convert Markdown / Text back to PDF preserving page boundaries."""

from __future__ import annotations

import re
from pathlib import Path

from fpdf import FPDF


def clean_text_for_fpdf(text: str) -> str:
    """Normalize typography symbols to latin-1 compatible ASCII."""
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "--",
        "\u2026": "...",
        "\u00a0": " ",
        "\ufeff": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Filter out characters outside latin-1
    return text.encode("latin-1", "replace").decode("latin-1")


def md_to_pdf(
    md_path: str | Path,
    output_pdf_path: str | Path,
    words_per_page: int = 500,
) -> int:
    """Convert Markdown file to PDF. Preserves <!-- PAGE n --> markers if present."""
    md_path = Path(md_path)
    output_pdf_path = Path(output_pdf_path)
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check for page markers
    marker_pattern = r"<!--\s*PAGE\s+\d+\s*-->"
    has_markers = bool(re.search(marker_pattern, content, re.I))

    if has_markers:
        raw_pages = re.split(marker_pattern, content)
        pages = [p.strip() for p in raw_pages if p.strip()]
    else:
        # Fallback word-count chunking
        words = content.split()
        pages = []
        for i in range(0, len(words), words_per_page):
            pages.append(" ".join(words[i : i + words_per_page]))

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=False)
    pdf.set_font("helvetica", size=9)

    for page_text in pages:
        pdf.add_page()
        cleaned = clean_text_for_fpdf(page_text)
        # 190mm usable width, 5mm line height
        pdf.multi_cell(w=190, h=5, text=cleaned)

    pdf.output(str(output_pdf_path))
    return len(pages)
