from src.schema import RetrievedChunk


def format_context(chunks: list[RetrievedChunk]) -> str:
    formatted: list[str] = []
    for idx, c in enumerate(chunks, 1):
        pages_str = ", ".join(map(str, c.pages[:3]))
        header = f"[{idx}] Dokumen: {c.doc_id.upper()} | {c.bab} - {c.pasal} (Halaman: {pages_str})"
        formatted.append(f"{header}\n{c.text}\n")
    return "\n---\n".join(formatted)
