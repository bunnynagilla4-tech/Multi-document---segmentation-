"""Article Similarity Module for NewsLens.
Calculates TF-IDF vector representations and pairwise cosine similarity.
Supports top-K similar article retrieval and distance matrix generation.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import config


class ArticleSimilarity:
    """Computes TF-IDF vector representations and pairwise cosine similarities."""

    def __init__(self, max_features: int = 2000):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_features,
            token_pattern=r"(?u)\b[a-zA-Z0-9-]{3,}\b"
        )
        self.tfidf_matrix = None
        self.similarity_matrix = None

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Compute TF-IDF vectors and symmetric cosine similarity matrix."""
        if not texts or len(texts) == 0:
            raise ValueError("Input texts list cannot be empty.")

        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        return self.similarity_matrix

    def get_most_similar(
        self,
        article_idx: int,
        top_k: int = 3,
        titles: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve the top-K most similar articles for a given article index."""
        if self.similarity_matrix is None:
            raise ValueError("Must run fit_transform before querying similarity.")

        n_docs = self.similarity_matrix.shape[0]
        if article_idx < 0 or article_idx >= n_docs:
            raise IndexError(f"Article index {article_idx} out of range [0, {n_docs-1}].")

        scores = self.similarity_matrix[article_idx].copy()
        scores[article_idx] = -1.0  # Exclude self-similarity

        # Get sorted indices descending
        best_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in best_indices:
            score = float(scores[idx])
            if score < 0:
                continue
            item = {
                "article_index": int(idx),
                "similarity_score": round(score, 3),
                "title": titles[idx] if titles and idx < len(titles) else f"Article {idx}"
            }
            results.append(item)

        return results

    def get_distance_matrix(self) -> np.ndarray:
        """Compute symmetric distance matrix D = 1 - S, bounded in [0, 1] with 0 on diagonal."""
        if self.similarity_matrix is None:
            raise ValueError("Similarity matrix is not yet computed.")

        dist = 1.0 - self.similarity_matrix
        dist = np.clip(dist, 0.0, 1.0)
        # Symmetrize to guard against floating point inaccuracies
        dist = (dist + dist.T) / 2.0
        np.fill_diagonal(dist, 0.0)
        return dist

    def get_similarity_matrix(self) -> np.ndarray:
        """Return the computed symmetric cosine similarity matrix."""
        return self.similarity_matrix

