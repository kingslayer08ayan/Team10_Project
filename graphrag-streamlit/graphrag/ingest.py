"""
Stage 1: Ingestion (layout-aware, section-hierarchical).

Replaces plain per-page text extraction with:
  1. Layout-aware span extraction (PyMuPDF get_text("dict")) -- reads font
     size and bold flags per line, not just raw text, so we can tell a
     header from a body paragraph without a trained layout model.
  2. High-confidence top-level section detection -- a line is only treated
     as a section break if it has BOTH a strong layout signal (larger
     font or bold) AND either matches a known section name or has an
     unusually strong font-size jump. Weak signals are ignored as body
     text, to avoid false positives from bolded phrases mid-paragraph.
  3. Cross-page assembly -- sections accumulate text across page
     boundaries; a section only ends when the next confident header
     appears, possibly several pages later.
  4. Semantic child chunking inside each section -- sentence-boundary
     merging up to a target size with slight overlap (same idea as
     Docling's recursive split-then-merge), scoped within one section so
     a child chunk never straddles two different sections.
  5. Reference/bibliography/acknowledgment sections are detected and
     excluded from chunking entirely -- pure noise for retrieval and gap
     analysis.

Each child Chunk carries both a parent_id (pointing to the full section
text, stored separately) and a short extractive section_summary (first
sentence of its parent section) that gets prepended at embedding time
only -- so embeddings have topic context even though the displayed text
stays exactly what was in the PDF.
"""
import os
import re
import glob
from dataclasses import dataclass, field

import fitz  # PyMuPDF

from . import sections as sec_module

SKIP_SECTION_NAMES = {
    "references", "bibliography", "acknowledgments", "acknowledgements",
    "appendix", "supplementary material",
}

HEADER_MAX_CHARS = 80
FONT_SIZE_RATIO_STRONG = 1.30   # font this much bigger than body = strong signal alone
FONT_SIZE_RATIO_WEAK = 1.15     # font this much bigger + bold, or matches alias = ok
NUMBERING_RE = re.compile(r"^\s*(\d+(\.\d+)*\.?|[IVXLC]+\.)\s+")


@dataclass
class Section:
    section_id: str
    doc_id: str
    canonical_section: str
    original_heading: str
    confidence: float
    tier: int
    text: str
    start_page: int
    end_page: int
    skipped: bool = False


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_file: str
    page: int
    doc_id: str
    canonical_section: str = "Unknown"
    section_heading: str = ""
    section_summary: str = ""   # extractive, prepended at embed time only
    parent_id: str = ""         # -> Section.section_id


# ------------------------------------------------------------------
# Layout-aware span extraction
# ------------------------------------------------------------------

def _page_lines_with_layout(page):
    """
    Yields (text, max_font_size, is_bold) per line on the page, using
    PyMuPDF's structured dict output instead of plain text.
    """
    d = page.get_text("dict")
    for block in d.get("blocks", []):
        if block.get("type") != 0:  # skip image blocks
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = "".join(s["text"] for s in spans).strip()
            if not text:
                continue
            max_size = max(s["size"] for s in spans)
            is_bold = any((s["flags"] & 2 ** 4) or "bold" in s.get("font", "").lower()
                         for s in spans)
            yield text, max_size, is_bold


def _estimate_body_font_size(pdf):
    """Most common span font size across the doc = body text size."""
    from collections import Counter
    sizes = Counter()
    for page in pdf:
        for _, size, _ in _page_lines_with_layout(page):
            sizes[round(size)] += 1
    if not sizes:
        return 10.0
    return float(sizes.most_common(1)[0][0])


def _is_header_candidate(text, font_size, is_bold, body_size):
    if len(text) > HEADER_MAX_CHARS or len(text) < 3:
        return False
    if text.endswith("."):  # headers rarely end in a period
        return False
    ratio = font_size / body_size if body_size else 1.0
    stripped = NUMBERING_RE.sub("", text).strip()
    known_alias, conf = sec_module.classify_tier1(stripped)

    if ratio >= FONT_SIZE_RATIO_STRONG:
        return True
    if ratio >= FONT_SIZE_RATIO_WEAK and (is_bold or known_alias):
        return True
    if known_alias and conf == 1.0 and is_bold:
        return True
    return False


# ------------------------------------------------------------------
# Chunking within a section
# ------------------------------------------------------------------

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _chunk_text(text, chunk_size=500, overlap=60):
    """Sentence-boundary sliding window, sized for child chunks within a
    single section (smaller than the old page-level 800 char default,
    since sections are now the coarse unit and children should be finer).
    This is the fallback used when no embedder is available -- fixed
    size, no notion of where the topic actually shifts."""
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
            break
        next_start = end - overlap
        start = next_start if next_start > start else end
    return [c for c in chunks if c]


def _semantic_chunk_text(text, embedder, chunk_size=500, min_chunk_size=120,
                         similarity_drop_threshold=0.35):
    """
    True semantic chunking: splits into sentences, embeds each one, and
    cuts a new chunk boundary where consecutive-sentence similarity drops
    sharply (a topic shift) OR the accumulated chunk hits chunk_size,
    whichever comes first. This is what makes chunking "semantic" rather
    than just "sentence-boundary-snapped" -- the cut points are chosen by
    where the content's meaning actually changes, not by a fixed
    character count.

    similarity_drop_threshold is a judgment call like MERGE_THRESHOLD in
    entity_resolution.py: 0.35 is a reasonable starting point for
    consecutive-sentence cosine similarity on academic prose (which tends
    to run more diffuse than pairs of standalone entity names), but
    recalibrate against your own corpus rather than trusting this blindly.

    Falls back to _chunk_text if the section has fewer than 2 sentences
    (nothing to detect a shift between) -- and the caller falls back
    entirely to _chunk_text if no embedder is available at all.
    """
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if len(sentences) < 2:
        return _chunk_text(text, chunk_size)

    import numpy as np
    vecs = np.array(list(embedder.embed(sentences)))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    unit_vecs = vecs / norms

    chunks = []
    current = [sentences[0]]
    current_len = len(sentences[0])

    for i in range(1, len(sentences)):
        sim = float(unit_vecs[i] @ unit_vecs[i - 1])
        sentence_len = len(sentences[i])
        topic_shift = sim < (1 - similarity_drop_threshold)
        would_overflow = current_len + sentence_len > chunk_size

        if (topic_shift or would_overflow) and current_len >= min_chunk_size:
            chunks.append(" ".join(current))
            current = [sentences[i]]
            current_len = sentence_len
        else:
            current.append(sentences[i])
            current_len += sentence_len

    if current:
        chunks.append(" ".join(current))

    return chunks


def _extractive_summary(text, max_chars=160):
    """First sentence of the section, truncated -- the free version of a
    chunk-summary prepend, no LLM call needed."""
    first_period = text.find(". ")
    summary = text[:first_period + 1] if first_period != -1 else text[:max_chars]
    return summary[:max_chars].strip()


# ------------------------------------------------------------------
# Section assembly (per document)
# ------------------------------------------------------------------

def _assemble_sections(pdf, doc_id, embedder=None):
    """
    Walks every page in order, detects high-confidence section headers
    via layout, and assembles cross-page section text buffers. Returns
    a list of Section objects in document order.
    """
    body_size = _estimate_body_font_size(pdf)

    raw_sections = []  # list of dicts: heading, page_start, segments=[(page,text)]
    current = {"heading": "", "page_start": 1, "segments": []}

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        for text, font_size, is_bold in _page_lines_with_layout(page):
            if _is_header_candidate(text, font_size, is_bold, body_size):
                if current["segments"]:
                    raw_sections.append(current)
                heading = NUMBERING_RE.sub("", text).strip()
                current = {"heading": heading, "page_start": page_num + 1, "segments": []}
            else:
                current["segments"].append((page_num + 1, text))
    if current["segments"]:
        raw_sections.append(current)

    if not raw_sections:
        # No confident headers detected anywhere in the doc -- fall back
        # to treating the whole document as one unheaded section; child
        # chunks within it will get position-based canonical tags later.
        full_text = " ".join(
            t for p in range(len(pdf)) for t, _, _ in _page_lines_with_layout(pdf[p])
        )
        raw_sections = [{"heading": "", "page_start": 1,
                         "segments": [(1, full_text)]}]

    # Classify each raw section and build Section objects
    total_chars = sum(len(t) for r in raw_sections for _, t in r["segments"])
    sections = []
    chars_so_far = 0

    for idx, raw in enumerate(raw_sections):
        text = " ".join(t for _, t in raw["segments"])
        end_page = raw["segments"][-1][0] if raw["segments"] else raw["page_start"]
        position_fraction = chars_so_far / total_chars if total_chars else 0.0
        chars_so_far += len(text)

        canonical, original, confidence, tier = sec_module.classify_section(
            raw["heading"], text[:300], position_fraction, embedder=embedder
        )
        skip = canonical.lower() in SKIP_SECTION_NAMES or \
               original.lower().strip(":") in SKIP_SECTION_NAMES

        sections.append(Section(
            section_id=f"{doc_id}::sec{idx}",
            doc_id=doc_id,
            canonical_section=canonical,
            original_heading=original or "(untitled)",
            confidence=confidence,
            tier=tier,
            text=text,
            start_page=raw["page_start"],
            end_page=end_page,
            skipped=skip,
        ))

    return sections


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------

def load_and_chunk(database_dir="database", chunk_size=500, overlap=60, embedder=None,
                  progress_callback=None):
    """
    Returns (chunks, sections, pdf_files):
      chunks   - list of Chunk (child-level, for embedding/retrieval)
      sections - list of Section (parent-level, for context expansion)
      pdf_files - list of processed PDF filenames

    `embedder` is optional -- if a fastembed model instance is passed:
      - Tier 2 section classification (embedding-based) is available
      - Child chunking within each section uses true semantic chunking
        (topic-shift detection between consecutive sentences) instead of
        the fixed sentence-window fallback
    Without it, both fall back to their non-embedding versions -- the
    pipeline still runs, just with coarser section tags and fixed-size
    chunks rather than topic-boundary-aware ones.
    """
    all_chunks = []
    all_sections = []
    pdf_paths = sorted(glob.glob(os.path.join(database_dir, "*.pdf")))

    for idx, pdf_path in enumerate(pdf_paths, start=1):
        doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
        source_file = os.path.basename(pdf_path)
        if progress_callback:
            try:
                progress_callback(idx, len(pdf_paths), stage="ingestion",
                                  message=f"Ingesting PDF {idx}/{len(pdf_paths)}: {source_file}")
            except TypeError:
                progress_callback(idx, len(pdf_paths))

        try:
            pdf = fitz.open(pdf_path)
        except Exception as e:
            print(f"[ingest] Skipping {pdf_path}: {e}")
            continue

        doc_sections = _assemble_sections(pdf, doc_id, embedder=embedder)
        all_sections.extend(doc_sections)

        for section in doc_sections:
            if section.skipped:
                continue  # references/appendix etc. -- excluded from chunking
            summary = _extractive_summary(section.text)

            if embedder is not None:
                child_texts = _semantic_chunk_text(section.text, embedder, chunk_size)
            else:
                child_texts = _chunk_text(section.text, chunk_size, overlap)

            for i, child_text in enumerate(child_texts):
                all_chunks.append(Chunk(
                    chunk_id=f"{section.section_id}::c{i}",
                    text=child_text,
                    source_file=source_file,
                    page=section.start_page,
                    doc_id=doc_id,
                    canonical_section=section.canonical_section,
                    section_heading=section.original_heading,
                    section_summary=summary,
                    parent_id=section.section_id,
                ))

        pdf.close()

    return all_chunks, all_sections, [os.path.basename(p) for p in pdf_paths]
