"""Rubric templates for LLM-as-judge evaluation."""

from __future__ import annotations

FAITHFULNESS_PROMPT = """Anda adalah evaluator RAG hukum yang objektif dan ketat.
Tugas Anda adalah menilai apakah jawaban yang diberikan HANYA didasarkan pada konteks yang disediakan (tanpa halusinasi).

KONTEKS:
{context}

PERTANYAAN:
{question}

JAWABAN:
{answer}

Kriteria Penilaian:
- Skor 1.0 jika seluruh pernyataan fakta dalam jawaban dapat dibuktikan langsung dari konteks.
- Skor 0.5 jika sebagian besar fakta dari konteks namun ada klaim minor yang tidak didukung.
- Skor 0.0 jika jawaban mengandung klaim hukum bertentangan atau sepenuhnya di luar konteks.

Kembalikan HANYA format JSON valid berikut tanpa markdown backtick:
{{"score": float, "reason": "penjelasan singkat 1 kalimat"}}
"""

ANSWER_RELEVANCY_PROMPT = """Anda adalah evaluator RAG hukum.
Tugas Anda adalah menilai seberapa tepat dan relevan jawaban terhadap pertanyaan yang diajukan.

PERTANYAAN:
{question}

JAWABAN:
{answer}

Kriteria Penilaian:
- Skor 1.0 jika jawaban secara langsung dan lengkap menjawab pertanyaan.
- Skor 0.5 jika jawaban menjawab sebagian pertanyaan atau bertele-tele namun masih relevan.
- Skor 0.0 jika jawaban tidak relevan, menolak menjawab tanpa dasar, atau menyimpang jauh.

Kembalikan HANYA format JSON valid berikut tanpa markdown backtick:
{{"score": float, "reason": "penjelasan singkat 1 kalimat"}}
"""
