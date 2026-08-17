"""
Build pipeline: ingest -> entity extraction -> graph -> ChunkIndex (Chroma).

Caching strategy (two separate stores):
  - Graph + communities -> .cache/graph.pkl  (networkx, fast to rebuild)
  - Chunk vectors       -> .cache/chroma/    (ChromaDB persistent dir)

This means:
  - Adding new PDFs: graph rebuilds from scratch (fast, ~0.1s for 20 PDFs)
    but Chroma only embeds the NEW chunks (incremental upsert in
    embeddings.py), saving fastembed time on chunks already indexed.
  - Sklearn version mismatch warnings are gone: no TfidfVectorizer pickled.
  - The bundle returned is the same shape as before so app.py is unchanged.
"""
import os
import pickle

from . import ingest, graph_builder
from .embeddings import ChunkIndex

GRAPH_CACHE = ".cache/graph.pkl"


def build(database_dir="database", progress_callback=None):
    chunks, pdf_files = ingest.load_and_chunk(database_dir)
    if not chunks:
        raise RuntimeError(f"No extractable text found in PDFs under '{database_dir}/'.")

    graph, communities = graph_builder.build_graph(
        chunks, progress_callback=progress_callback
    )

    # ChunkIndex handles Chroma persistence internally;
    # passing chunks triggers incremental upsert for any new ones.
    index = ChunkIndex(chunks)

    bundle = {
        "chunks": chunks,
        "pdf_files": pdf_files,
        "graph": graph,
        "communities": communities,
        "index": index,
    }

    os.makedirs(os.path.dirname(GRAPH_CACHE), exist_ok=True)
    with open(GRAPH_CACHE, "wb") as f:
        pickle.dump(
            {"chunks": chunks, "pdf_files": pdf_files,
             "graph": graph, "communities": communities},
            f,
        )

    return bundle


def load_cached():
    """
    Loads graph from pickle and re-attaches a ChunkIndex pointed at the
    existing Chroma collection (no re-embedding on load).
    """
    if not os.path.exists(GRAPH_CACHE):
        return None
    try:
        with open(GRAPH_CACHE, "rb") as f:
            cached = pickle.load(f)
    except Exception:
        return None

    chunks = cached["chunks"]
    # Re-create ChunkIndex: Chroma sees all chunk_ids already exist,
    # skips re-embedding, just reconnects to the persistent collection.
    index = ChunkIndex(chunks)

    return {
        "chunks": chunks,
        "pdf_files": cached["pdf_files"],
        "graph": cached["graph"],
        "communities": cached["communities"],
        "index": index,
    }