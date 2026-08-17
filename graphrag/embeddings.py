"""
Stage 4: Embedding + vector store.

Replaces the TF-IDF matrix with:
  - fastembed (BAAI/bge-small-en-v1.5, ONNX, no PyTorch) for encoding
  - ChromaDB in persistent local mode as the vector store

Why this pair:
  - fastembed downloads the model once (~90 MB) on first run, then caches
    it locally -- no network call on subsequent runs.
  - ChromaDB persists its collection to disk (.cache/chroma/) so adding
    new PDFs only embeds the new chunks, not the whole corpus again
    (incremental upsert via chunk_id as the document ID).
  - The public interface (embed_query / similarity_scores) is kept
    identical to the old TF-IDF ChunkIndex so retriever.py only changes
    where it matters (Chroma query instead of matrix multiply).

Fallback: if fastembed isn't importable or the model download hasn't
happened yet, the class transparently falls back to TF-IDF so the app
stays runnable on first launch before the model is cached.
"""
import os
import numpy as np

CHROMA_DIR = ".cache/chroma"
COLLECTION_NAME = "graphrag_chunks"
FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"


def _try_import_fastembed():
    try:
        from fastembed import TextEmbedding
        return TextEmbedding
    except Exception:
        return None


def _try_import_chroma():
    try:
        import chromadb
        return chromadb
    except Exception:
        return None


class ChunkIndex:
    """
    Unified interface over either:
      A) fastembed + ChromaDB  (preferred, persistent, incremental)
      B) TF-IDF in-memory      (fallback if fastembed model not ready)

    Call `.backend` to see which is active: "chroma" or "tfidf".
    """

    def __init__(self, chunks):
        self.chunks = chunks
        self._chunk_id_to_pos = {c.chunk_id: i for i, c in enumerate(chunks)}
        self.backend = self._build(chunks)

    def _build(self, chunks):
        TextEmbedding = _try_import_fastembed()
        chromadb = _try_import_chroma()

        if TextEmbedding and chromadb:
            try:
                return self._build_chroma(chunks, TextEmbedding, chromadb)
            except Exception as e:
                print(f"[embeddings] Chroma/fastembed build failed ({e}); "
                      f"falling back to TF-IDF.")

        return self._build_tfidf(chunks)

    # ------------------------------------------------------------------
    # Path A: fastembed + ChromaDB
    # ------------------------------------------------------------------
    def _build_chroma(self, chunks, TextEmbedding, chromadb):
        self._embedder = TextEmbedding(FASTEMBED_MODEL)
        os.makedirs(CHROMA_DIR, exist_ok=True)
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        self._collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # Only embed chunks not already in the collection (incremental add)
        existing_ids = set(self._collection.get(include=[])["ids"])
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

        if new_chunks:
            texts = [c.text for c in new_chunks]
            ids = [c.chunk_id for c in new_chunks]
            metadatas = [
                {
                    "source_file": c.source_file,
                    "page": c.page,
                    "doc_id": c.doc_id,
                }
                for c in new_chunks
            ]
            # fastembed returns a generator; list() forces evaluation
            embeddings = list(self._embedder.embed(texts))
            # Chroma expects plain Python lists, not numpy arrays
            embeddings_list = [e.tolist() for e in embeddings]
            self._collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings_list,
                metadatas=metadatas,
            )
            print(f"[embeddings] Added {len(new_chunks)} new chunks to Chroma "
                  f"({len(existing_ids)} already indexed).")
        else:
            print(f"[embeddings] All {len(chunks)} chunks already in Chroma; "
                  f"skipping re-embed.")

        return "chroma"

    def embed_query(self, text):
        """Returns query embedding as numpy array (chroma path) or sparse matrix (tfidf)."""
        if self.backend == "chroma":
            vecs = list(self._embedder.embed([text]))
            return np.array(vecs[0])
        else:
            return self._tfidf_vec.transform([text])

    def similarity_scores(self, query_vec):
        """
        Returns a numpy array of similarity scores, one per chunk,
        in the same order as self.chunks.
        """
        if self.backend == "chroma":
            return self._chroma_scores(query_vec)
        else:
            from sklearn.metrics.pairwise import cosine_similarity
            return cosine_similarity(query_vec, self._tfidf_matrix).flatten()

    def _chroma_scores(self, query_vec):
        """
        Query Chroma for all chunks, convert distances to similarities,
        and return a score array aligned with self.chunks order.
        """
        n = len(self.chunks)
        if n == 0:
            return np.zeros(0)

        results = self._collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=min(n, self._collection.count()),
            include=["distances"],
        )
        ids = results["ids"][0]
        # Chroma cosine space returns distance = 1 - similarity
        distances = results["distances"][0]

        scores = np.zeros(n)
        for chunk_id, dist in zip(ids, distances):
            pos = self._chunk_id_to_pos.get(chunk_id)
            if pos is not None:
                scores[pos] = max(0.0, 1.0 - dist)  # convert to similarity

        return scores

    # ------------------------------------------------------------------
    # Path B: TF-IDF fallback
    # ------------------------------------------------------------------
    def _build_tfidf(self, chunks):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf_vec = TfidfVectorizer(stop_words="english", max_features=5000)
        self._tfidf_matrix = self._tfidf_vec.fit_transform([c.text for c in chunks])
        print("[embeddings] Using TF-IDF fallback (fastembed model not available yet).")
        return "tfidf"