from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np


class SimpleCache:

    def __init__(self, max_size=50, similarity_threshold=0.90):

        self.cache = {}
        self.order = []

        self.max_size = max_size
        self.similarity_threshold = similarity_threshold

        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en"
        )

        self.query_vectors = {}

    # -------------------------
    # Exact cache
    # -------------------------
    def get(self, query):
        return self.cache.get(query)

    # -------------------------
    # Semantic cache
    # -------------------------
    def semantic_get(self, query):

        if not self.query_vectors:
            return None

        query_vec = self.embeddings.embed_query(query)

        best_match = None
        best_score = 0

        for cached_query, vec in self.query_vectors.items():

            score = np.dot(query_vec, vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(vec)
            )

            if score > best_score:
                best_score = score
                best_match = cached_query

        if best_score >= self.similarity_threshold:
            return self.cache.get(best_match)

        return None

    # -------------------------
    # Set cache
    # -------------------------
    def set(self, query, value):

        if query not in self.cache:

            if len(self.order) >= self.max_size:

                oldest = self.order.pop(0)

                del self.cache[oldest]
                del self.query_vectors[oldest]

            self.order.append(query)

        self.cache[query] = value

        self.query_vectors[query] = self.embeddings.embed_query(query)