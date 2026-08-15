"""
Ties ingest -> entity extraction -> graph build -> vector index together,
and caches the result to disk (pickle) so the Streamlit app only rebuilds
when the user explicitly asks it to, not on every UI interaction.
"""
import os
import pickle

from . import _1_ingest as ingest, _3_graph_builder as graph_builder
from ._4_embeddings import ChunkIndex

CACHE_PATH = ".cache/graph_index.pkl"


def build(database_dir="database", progress_callback=None):
    chunks, pdf_files = ingest.load_and_chunk(database_dir)
    if not chunks:
        raise RuntimeError(f"No extractable text found in PDFs under '{database_dir}/'.")

    graph, communities = graph_builder.build_graph(chunks, progress_callback=progress_callback)
    index = ChunkIndex(chunks)

    bundle = {
        "chunks": chunks,
        "pdf_files": pdf_files,
        "graph": graph,
        "communities": communities,
        "index": index,
    }
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(bundle, f)
    return bundle


def load_cached():
    if not os.path.exists(CACHE_PATH):
        return None
    with open(CACHE_PATH, "rb") as f:
        return pickle.load(f)
