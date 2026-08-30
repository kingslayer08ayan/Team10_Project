"""
Build pipeline: ingest (section-aware, semantic chunking) -> entity
extraction + resolution -> typed graph + community summaries ->
ChunkIndex (Chroma).

Caching strategy (two separate stores):
  - Graph + sections + communities + summaries -> .cache/graph.pkl
  - Chunk vectors                              -> .cache/chroma/

One shared embedder instance is created once and threaded through every
stage that can use it (section Tier-2 classification, semantic chunking,
entity resolution, Chroma indexing) -- if fastembed isn't ready or the
network is blocked, every one of those stages falls back to its
non-embedding version rather than the whole build failing.
"""
import os
import pickle

from . import ingest, graph_builder
from .embeddings import ChunkIndex

GRAPH_CACHE = ".cache/graph.pkl"


def _get_shared_embedder():
    """Best-effort fastembed instance, reused across every pipeline stage.
    Returns None (fast, bounded by a timeout) if fastembed isn't
    installed, the model isn't downloaded yet, or the network is
    blocked/slow -- every stage that uses it has a non-embedding
    fallback, so the app still runs, just with coarser results."""
    from .embeddings import create_embedder_with_timeout
    return create_embedder_with_timeout()


def build(database_dir="database", progress_callback=None):
    embedder = _get_shared_embedder()

    chunks, doc_sections, pdf_files = ingest.load_and_chunk(
        database_dir, embedder=embedder, progress_callback=progress_callback
    )
    if not chunks:
        raise RuntimeError(f"No extractable text found in PDFs under '{database_dir}/'.")

    graph, communities, community_summaries = graph_builder.build_graph(
        chunks, progress_callback=progress_callback, embedder=embedder
    )

    # ChunkIndex handles Chroma persistence internally; the same embedder
    # instance is reused inside it rather than loading the model twice.
    index = ChunkIndex(chunks, embedder=embedder)

    sections_by_id = {s.section_id: s for s in doc_sections}

    bundle = {
        "chunks": chunks,
        "sections": doc_sections,
        "sections_by_id": sections_by_id,
        "pdf_files": pdf_files,
        "graph": graph,
        "communities": communities,
        "community_summaries": community_summaries,
        "index": index,
    }

    os.makedirs(os.path.dirname(GRAPH_CACHE), exist_ok=True)
    with open(GRAPH_CACHE, "wb") as f:
        pickle.dump(
            {"chunks": chunks, "sections": doc_sections, "pdf_files": pdf_files,
             "graph": graph, "communities": communities,
             "community_summaries": community_summaries},
            f,
        )

    return bundle


def load_cached():
    """
    Loads graph + sections + community summaries from pickle and
    re-attaches a ChunkIndex pointed at the existing Chroma collection
    (no re-embedding, no re-extraction, no re-resolution on load).
    """
    if not os.path.exists(GRAPH_CACHE):
        return None
    try:
        with open(GRAPH_CACHE, "rb") as f:
            cached = pickle.load(f)
    except Exception:
        return None

    chunks = cached["chunks"]
    doc_sections = cached.get("sections", [])
    index = ChunkIndex(chunks)

    return {
        "chunks": chunks,
        "sections": doc_sections,
        "sections_by_id": {s.section_id: s for s in doc_sections},
        "pdf_files": cached["pdf_files"],
        "graph": cached["graph"],
        "communities": cached["communities"],
        "community_summaries": cached.get("community_summaries", {}),
        "index": index,
    }
