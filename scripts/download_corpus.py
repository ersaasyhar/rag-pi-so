"""Corpus downloader for Indonesian legal documents.

Downloads landmark UUs from BPK and extracts PPs from HF Azzindani/ID_REG.
Builds data/corpus_manifest.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import requests
from huggingface_hub import hf_hub_download

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
MANIFEST_PATH = DATA_DIR / "corpus_manifest.json"

BPK_LANDMARK_UUS = [
    {
        "doc_id": "uu_1_2023",
        "jenis_peraturan": "UU",
        "nomor": "1",
        "tahun": "2023",
        "judul": "Kitab Undang-Undang Hukum Pidana (KUHP)",
        "url": "https://peraturan.bpk.go.id/Download/287456/UU%20Nomor%201%20Tahun%202023.pdf",
    },
    {
        "doc_id": "uu_27_2022",
        "jenis_peraturan": "UU",
        "nomor": "27",
        "tahun": "2022",
        "judul": "Pelindungan Data Pribadi",
        "url": "https://peraturan.bpk.go.id/Download/224884/UU%20Nomor%2027%20Tahun%202022.pdf",
    },
    {
        "doc_id": "uu_6_2023",
        "jenis_peraturan": "UU",
        "nomor": "6",
        "tahun": "2023",
        "judul": "Penetapan Perppu Cipta Kerja Menjadi Undang-Undang",
        "url": "https://peraturan.bpk.go.id/Download/302681/UU%20Nomor%206%20Tahun%202023.pdf",
    },
    {
        "doc_id": "uu_19_2016",
        "jenis_peraturan": "UU",
        "nomor": "19",
        "tahun": "2016",
        "judul": "Perubahan Atas UU Informasi dan Transaksi Elektronik",
        "url": "https://peraturan.bpk.go.id/Download/26676/UU%20Nomor%20%2019%20Tahun%202016.pdf",
    },
    {
        "doc_id": "uu_40_2007",
        "jenis_peraturan": "UU",
        "nomor": "40",
        "tahun": "2007",
        "judul": "Perseroan Terbatas",
        "url": "https://peraturan.bpk.go.id/Download/29563/UU%20Nomor%2040%20Tahun%202007.pdf",
    },
    {
        "doc_id": "uu_12_2011",
        "jenis_peraturan": "UU",
        "nomor": "12",
        "tahun": "2011",
        "judul": "Pembentukan Peraturan Perundang-undangan",
        "url": "https://peraturan.bpk.go.id/Download/28610/UU%2012%20Tahun%202011.pdf",
    },
    {
        "doc_id": "uu_11_2008",
        "jenis_peraturan": "UU",
        "nomor": "11",
        "tahun": "2008",
        "judul": "Informasi dan Transaksi Elektronik",
        "url": "https://peraturan.bpk.go.id/Download/26683/UU%20Nomor%2011%20Tahun%202008.pdf",
    },
    {
        "doc_id": "uu_20_2008",
        "jenis_peraturan": "UU",
        "nomor": "20",
        "tahun": "2008",
        "judul": "Usaha Mikro, Kecil, dan Menengah",
        "url": "https://peraturan.bpk.go.id/Download/29221/UU%20Nomor%2020%20Tahun%202008.pdf",
    },
    {
        "doc_id": "uu_30_2014",
        "jenis_peraturan": "UU",
        "nomor": "30",
        "tahun": "2014",
        "judul": "Administrasi Pemerintahan",
        "url": "https://peraturan.bpk.go.id/Download/28023/UU%20Nomor%2030%20Tahun%202014.pdf",
    },
]


def download_file(url: str, dest_path: Path) -> bool:
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        return True
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        r = requests.get(url, headers=headers, stream=True, timeout=30)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}", file=sys.stderr)
        if dest_path.exists():
            dest_path.unlink()
        return False


def parse_pdf_header(pdf_path: Path) -> tuple[str, str, str, str, int]:
    """Extract (jenis, nomor, tahun, judul, num_pages) from PDF first page."""
    doc = fitz.open(pdf_path)
    num_pages = doc.page_count
    first_text = ""
    if num_pages > 0:
        first_text = doc[0].get_text("text")[:2000]
    doc.close()

    filename = pdf_path.stem
    # Defaults from filename
    m = re.search(
        r"(UU|PP|PERPRES|PERMEN)[_\s]+(?:No(?:mor)?[_\s]+)?(\d+)[_\s]+(?:Tahun[_\s]+)?(\d{4})",
        filename,
        re.I,
    )
    jenis = "PERATURAN"
    nomor = "0"
    tahun = "0000"
    if m:
        jenis = m.group(1).upper()
        nomor = m.group(2)
        tahun = m.group(3)

    # Refine from first page text
    if "UNDANG-UNDANG" in first_text.upper():
        jenis = "UU"
    elif "PERATURAN PEMERINTAH" in first_text.upper():
        jenis = "PP"
    elif "PERATURAN PRESIDEN" in first_text.upper():
        jenis = "PERPRES"
    elif "PERATURAN MENTERI" in first_text.upper():
        jenis = "PERMEN"

    m_no = re.search(r"NOMOR\s+(\d+)\s+TAHUN\s+(\d{4})", first_text, re.I)
    if m_no:
        nomor = m_no.group(1)
        tahun = m_no.group(2)

    judul = ""
    m_tentang = re.search(
        r"TENTANG\s+([\s\S]+?)(?=\bDENGAN RAHMAT|\bBAB I|\bMenimbang)", first_text, re.I
    )
    if m_tentang:
        judul = " ".join(m_tentang.group(1).split())
    if not judul:
        judul = f"{jenis} Nomor {nomor} Tahun {tahun}"

    return jenis, nomor, tahun, judul, num_pages


def extract_hf_corpus(limit_pp: int = 40) -> list[Path]:
    """Download Azzindani/ID_REG batch-1-chunk-1.zip and extract up to limit_pp PDFs."""
    print("Fetching Azzindani/ID_REG batch-1-chunk-1.zip from Hugging Face...")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    zip_path = hf_hub_download(
        repo_id="Azzindani/ID_REG",
        filename="batch-1-chunk-1.zip",
        repo_type="dataset",
    )
    dest_dir = PDF_DIR / "pp"
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    with zipfile.ZipFile(zip_path, "r") as z:
        pdf_names = [n for n in z.namelist() if n.lower().endswith(".pdf")]
        for name in pdf_names[:limit_pp]:
            out_file = dest_dir / Path(name).name
            if not out_file.exists() or out_file.stat().st_size == 0:
                with z.open(name) as src, open(out_file, "wb") as dst:
                    dst.write(src.read())
            extracted.append(out_file)
    print(f"Extracted {len(extracted)} PPs from Hugging Face.")
    return extracted


def build_corpus(limit: int = 50) -> list[dict[str, Any]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []

    # 1. Download landmark UUs
    print(f"Downloading {len(BPK_LANDMARK_UUS)} landmark UUs from BPK...")
    uu_dir = PDF_DIR / "uu"
    uu_dir.mkdir(parents=True, exist_ok=True)
    for uu in BPK_LANDMARK_UUS:
        pdf_path = uu_dir / f"{uu['doc_id']}.pdf"
        print(f"  Downloading {uu['doc_id']}: {uu['judul']}...")
        if download_file(uu["url"], pdf_path):
            doc = fitz.open(pdf_path)
            pages = doc.page_count
            doc.close()
            manifest.append(
                {
                    "doc_id": uu["doc_id"],
                    "jenis_peraturan": uu["jenis_peraturan"],
                    "nomor": uu["nomor"],
                    "tahun": uu["tahun"],
                    "judul": uu["judul"],
                    "pdf_path": str(pdf_path.relative_to(BASE_DIR)).replace("\\", "/"),
                    "num_pages": pages,
                }
            )

    # 2. Extract PPs from Hugging Face
    needed_pp = max(0, limit - len(manifest))
    if needed_pp > 0:
        pp_files = extract_hf_corpus(limit_pp=needed_pp)
        for pdf_path in pp_files:
            try:
                jenis, nomor, tahun, judul, num_pages = parse_pdf_header(pdf_path)
                doc_id = f"{jenis.lower()}_{nomor}_{tahun}"
                # Deduplicate doc_id if collision
                existing_ids = {m["doc_id"] for m in manifest}
                if doc_id in existing_ids:
                    doc_id = f"{doc_id}_{pdf_path.stem}"

                manifest.append(
                    {
                        "doc_id": doc_id,
                        "jenis_peraturan": jenis,
                        "nomor": nomor,
                        "tahun": tahun,
                        "judul": judul,
                        "pdf_path": str(pdf_path.relative_to(BASE_DIR)).replace("\\", "/"),
                        "num_pages": num_pages,
                    }
                )
            except Exception as e:
                print(f"Error parsing {pdf_path}: {e}", file=sys.stderr)

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nCorpus manifest written to {MANIFEST_PATH}")
    print(f"Total documents: {len(manifest)}")
    print(f"Total pages: {sum(m['num_pages'] for m in manifest)}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Indonesian legal corpus")
    parser.add_argument("--limit", type=int, default=50, help="Target number of documents")
    args = parser.parse_args()
    build_corpus(limit=args.limit)
