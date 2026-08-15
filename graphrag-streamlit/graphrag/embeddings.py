"""
Stage 4: Vector index.
Uses TF-IDF as an offline, dependency-free embedding proxy so retrieval
works with zero API keys and zero network calls to a model host. This is
the one piece worth swapping first if you later add real embeddings
(e.g. a local sentence-transformers model or an embeddings API) -- the
rest of the pipeline (HyDE, graph boosting, retriever) only depends on
`vectorize([text])` and `similarity(matrix, vector)`, so the swap is
localized to this file.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ChunkIndex:
    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def embed_query(self, text):
        return self.vectorizer.transform([text])

    def similarity_scores(self, query_vec):
        """Returns a numpy array of cosine similarities, one per chunk."""
        return cosine_similarity(query_vec, self.matrix).flatten()
