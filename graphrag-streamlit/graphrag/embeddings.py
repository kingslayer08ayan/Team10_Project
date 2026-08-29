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
import threading
import numpy as np

CHROMA_DIR = ".cache/chroma"
COLLECTION_NAME = "graphrag_chunks"
FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDER_INIT_TIMEOUT_S = 8  # fail fast instead of riding fastembed's ~90s retry backoff
EMBED_BATCH_SIZE = 32
USE_FASTEMBED = os.environ.get("GRAPHRAG_USE_FASTEMBED", "0") == "1"


_embedder_cache = {"value": "unset"}  # sentinel distinguishes "not tried" from "tried, got None"


def create_embedder_with_timeout(timeout=EMBEDDER_INIT_TIMEOUT_S):
    """
    Tries to construct a fastembed TextEmbedding instance, bounded by a
    hard timeout. fastembed retries model downloads with exponential
    backoff (3s, 9s, 27s -- twice, from two mirror sources) when the
    network is blocked or Hugging Face is unreachable, which can hang
    for 90+ seconds before giving up on its own. We don't want that: if
    it's not ready within `timeout` seconds, treat it as unavailable and
    let the caller fall back to TF-IDF / Tier-1+3 sections immediately.

    Uses a daemon thread rather than ThreadPoolExecutor -- the executor's
    `with` block calls shutdown(wait=True) on exit, which blocks until
    the abandoned retry thread finishes anyway, silently defeating the
    timeout. A plain daemon thread lets us walk away from a still-running
    attempt without waiting for it or blocking process exit.

    Result is cached per-process (including a None/failure result) so
    callers in different modules (pipeline.py's section classifier,
    embeddings.py's ChunkIndex) don't each pay the timeout separately --
    the first attempt's outcome is reused for the rest of this run. If
    you fix the network mid-session, restart the app to retry.
    """
    if _embedder_cache["value"] != "unset":
        return _embedder_cache["value"]

    try:
        from fastembed import TextEmbedding
    except Exception:
        _embedder_cache["value"] = None
        return None

    result_holder = {}

    def _init():
        try:
            result_holder["model"] = TextEmbedding(FASTEMBED_MODEL)
        except Exception as e:
            result_holder["error"] = e

    thread = threading.Thread(target=_init, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        print(f"[embeddings] fastembed model init exceeded {timeout}s "
              f"(likely blocked/slow network) -- falling back to TF-IDF for now. "
              f"(the attempt keeps running in the background in case it still "
              f"succeeds, but we're not waiting on it)")
        result = None
    elif "model" in result_holder:
        result = result_holder["model"]
    else:
        print(f"[embeddings] fastembed model init failed "
              f"({result_holder.get('error')}); falling back to TF-IDF.")
        result = None

    _embedder_cache["value"] = result
    return result


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


class _NoOpEmbeddingFunction:
    """
    Satisfies Chroma's EmbeddingFunction interface (a callable taking
    Documents -> Embeddings) without ever loading or downloading a model.
    We always pass embeddings explicitly to upsert()/query(), so this
    should never actually be called -- it exists purely to stop Chroma
    from lazily constructing its own default ONNXMiniLM_L6_V2 embedding
    function, which is a separate model download from a different source
    than fastembed's, with no timeout protection of our own around it.
    """
    def __call__(self, input):
        raise RuntimeError(
            "ChunkIndex always supplies embeddings explicitly -- if this "
            "was called, something upstream stopped passing embeddings= "
            "to a Chroma add/upsert/query call."
        )

    def name(self):
        return "external-fastembed"


class ChunkIndex:
    """
    Unified interface over either:
      A) fastembed + ChromaDB  (preferred, persistent, incremental)
      B) TF-IDF in-memory      (fallback if fastembed model not ready)

    Call `.backend` to see which is active: "chroma" or "tfidf".
    """

    def __init__(self, chunks, embedder=None):
        self.chunks = chunks
        self._chunk_id_to_pos = {c.chunk_id: i for i, c in enumerate(chunks)}
        self._shared_embedder = embedder  # reuse a model instance if the caller has one
        self.backend = self._build(chunks)

    def _build(self, chunks):
        if not USE_FASTEMBED:
            return self._build_tfidf(chunks)

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
        self._embedder = self._shared_embedder or create_embedder_with_timeout()
        if self._embedder is None:
            raise RuntimeError("fastembed model unavailable within timeout")

        os.makedirs(CHROMA_DIR, exist_ok=True)
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        self._collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            # IMPORTANT: without this, Chroma may lazily construct its OWN
            # default embedding function (ONNXMiniLM_L6_V2) the first time
            # the collection is touched -- a second, completely separate
            # model download from a different source than fastembed's,
            # with no timeout guard of ours around it. Since we always
            # supply embeddings explicitly (both here and in queries),
            # Chroma never actually needs to call this -- passing a no-op
            # that errors loudly if it's ever invoked prevents the silent
            # background download entirely rather than just hoping it's
            # never triggered.
            embedding_function=_NoOpEmbeddingFunction(),
        )

        # Only embed chunks not already in the collection (incremental add)
        existing_ids = set(self._collection.get(include=[])["ids"])
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

        if new_chunks:
            # Embed with the section summary prepended for topic context,
            # but store the raw chunk text as the document -- retrieval
            # quality benefits from the context, display should stay clean.
            embed_texts = [
                f"{c.section_summary} {c.text}".strip() if c.section_summary else c.text
                for c in new_chunks
            ]
            display_texts = [c.text for c in new_chunks]
            ids = [c.chunk_id for c in new_chunks]
            metadatas = [
                {
                    "source_file": c.source_file,
                    "page": c.page,
                    "doc_id": c.doc_id,
                    "canonical_section": c.canonical_section,
                    "section_heading": c.section_heading,
                    "parent_id": c.parent_id,
                }
                for c in new_chunks
            ]
            for start in range(0, len(new_chunks), EMBED_BATCH_SIZE):
                end = start + EMBED_BATCH_SIZE
                embeddings = list(self._embedder.embed(embed_texts[start:end]))
                self._collection.upsert(
                    ids=ids[start:end],
                    documents=display_texts[start:end],
                    embeddings=[e.tolist() for e in embeddings],
                    metadatas=metadatas[start:end],
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

    def similarity_scores(self, query_vec, top_n=None):
        """
        Returns a numpy array of similarity scores, one per chunk,
        in the same order as self.chunks.
        """
        if self.backend == "chroma":
            return self._chroma_scores(query_vec, top_n=top_n)
        else:
            from sklearn.metrics.pairwise import cosine_similarity
            return cosine_similarity(query_vec, self._tfidf_matrix).flatten()

    def _chroma_scores(self, query_vec, top_n=None):
        """
        Query Chroma for all chunks, convert distances to similarities,
        and return a score array aligned with self.chunks order.
        """
        n = len(self.chunks)
        if n == 0:
            return np.zeros(0)

        result_count = min(n, self._collection.count())
        if top_n is not None:
            result_count = min(result_count, max(1, top_n))
        results = self._collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=result_count,
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