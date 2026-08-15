"""
Stage 1: Ingestion.
Loads every PDF in the database folder, extracts text per page with
PyMuPDF, and splits into overlapping chunks. Each chunk keeps its source
file and page number for citation-grade traceability all the way through
retrieval.
"""
import os
import glob
from dataclasses import dataclass, field

import fitz  # PyMuPDF


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_file: str
    page: int
    doc_id: str  # short id derived from filename, used as the node/citation key


def _chunk_text(text, chunk_size=800, overlap=150):
    """Simple sliding-window chunker over raw characters, snapped to sentence
    boundaries where possible so chunks don't cut mid-sentence."""
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            last_period = text.rfind(". ", start, end)
            if last_period != -1 and last_period > start + chunk_size // 2:
                end = last_period + 1
        chunks.append(text[start:end].strip())
        if end >= n:
            break  # reached the end of the text; stop instead of recomputing start
        next_start = end - overlap
        start = next_start if next_start > start else end  # guarantee forward progress
    return [c for c in chunks if c]


def load_and_chunk(database_dir="database", chunk_size=800, overlap=150):
    """Returns a list of Chunk objects across every PDF in database_dir."""
    all_chunks = []
    pdf_paths = sorted(glob.glob(os.path.join(database_dir, "*.pdf")))

    for pdf_path in pdf_paths:
        doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
        try:
            pdf = fitz.open(pdf_path)
        except Exception as e:
            print(f"[ingest] Skipping {pdf_path}: {e}")
            continue

        for page_num in range(len(pdf)):
            page_text = pdf[page_num].get_text("text")
            if not page_text.strip():
                continue
            for i, chunk_text in enumerate(_chunk_text(page_text, chunk_size, overlap)):
                chunk_id = f"{doc_id}::p{page_num + 1}::c{i}"
                all_chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source_file=os.path.basename(pdf_path),
                    page=page_num + 1,
                    doc_id=doc_id,
                ))
        pdf.close()

    return all_chunks, [os.path.basename(p) for p in pdf_paths]
