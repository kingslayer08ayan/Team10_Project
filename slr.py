"""
SLR Orchestrator.

Coordinates:

    Retrieved papers
          ↓
    Paper Analysis
          ↓
       Comparison
          ↓
        Writer
          ↓
    Literature Review
"""

from collections import defaultdict

import Paper_Analysis
import Comparison
import Writer


def _collect_selected_papers(results, all_chunks):
    """
    Find all chunks belonging to the papers selected by GraphRAG.
    """

    selected_doc_ids = []

    for result in results:
        doc_id = result["chunk"].doc_id

        if doc_id not in selected_doc_ids:
            selected_doc_ids.append(doc_id)

    chunks_by_doc = defaultdict(list)

    for chunk in all_chunks:
        chunks_by_doc[chunk.doc_id].append(chunk)

    selected_papers = {}

    for doc_id in selected_doc_ids:
        selected_papers[doc_id] = chunks_by_doc.get(doc_id, [])

    return selected_papers


def _format_paper(chunks):
    """
    Combine all chunks belonging to one paper into one text string.
    """

    if not chunks:
        return ""

    chunks = sorted(
        chunks,
        key=lambda c: (c.page, c.chunk_id)
    )

    source_file = chunks[0].source_file

    text = "\n".join(
        f"[Page {chunk.page}]\n{chunk.text}"
        for chunk in chunks
    )

    return f"""
Source file: {source_file}

{text}
"""


def generate_review(results, all_chunks):
    """
    Generate a literature review from the papers retrieved by GraphRAG.

    Args:
        results:
            Top-K results returned by GraphRAG.

        all_chunks:
            All chunks available in the GraphRAG bundle.

    Returns:
        str:
            Generated literature review.
    """

    if not results:
        raise ValueError(
            "No papers were selected for the literature review."
        )

    # ---------------------------------------------------------------
    # Stage 1: Get the selected papers
    # ---------------------------------------------------------------

    selected_papers = _collect_selected_papers(
        results,
        all_chunks,
    )

    print(
        f"[SLR] Selected {len(selected_papers)} papers."
    )

    # ---------------------------------------------------------------
    # Stage 2: Analyze each paper
    # ---------------------------------------------------------------

    analyses = []

    for number, (doc_id, chunks) in enumerate(
        selected_papers.items(),
        start=1,
    ):
        print(
            f"[SLR] Analyzing paper "
            f"{number}/{len(selected_papers)}: {doc_id}"
        )

        paper_text = _format_paper(chunks)

        if not paper_text.strip():
            print(
                f"[SLR] Skipping {doc_id}: no readable text."
            )
            continue

        analysis = Paper_Analysis.analyze(
            paper_text=paper_text,
            paper_id=doc_id,
        )

        analyses.append(analysis)

    if not analyses:
        raise ValueError(
            "None of the selected papers could be analyzed."
        )

    # ---------------------------------------------------------------
    # Stage 3: Compare papers
    # ---------------------------------------------------------------

    print(
        f"[SLR] Comparing {len(analyses)} papers..."
    )

    comparison = Comparison.compare(
        analyses
    )

    # ---------------------------------------------------------------
    # Stage 4: Write literature review
    # ---------------------------------------------------------------

    print(
        "[SLR] Writing literature review..."
    )

    review = Writer.write(
        analyses=analyses,
        comparison=comparison,
    )

    print(
        "[SLR] Literature review complete."
    )

    return review