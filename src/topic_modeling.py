"""Topic Modeling Module for NewsLens.
Implements Non-Negative Matrix Factorization (NMF) with TF-IDF and
Latent Dirichlet Allocation (LDA) for topic discovery and keyword extraction.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import config


class TopicModeler:
    """Unsupervised topic modeler using NMF or LDA.

    Note: Topics discovered are statistical approximations based on word co-occurrence
    and do not guarantee one-to-one correspondence with distinct real-world news events.
    """

    def __init__(
        self,
        n_topics: int = config.DEFAULT_NUM_TOPICS,
        method: str = "nmf",
        max_features: int = 1000,
        random_state: int = config.RANDOM_STATE
    ):
        self.n_topics = n_topics
        self.method = method.lower()
        self.max_features = max_features
        self.random_state = random_state

        self.vectorizer = None
        self.model = None
        self.feature_names = []
        self.topic_matrix = None

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit the vectorizer and decomposition model, returning document-topic matrix."""
        if not texts or len(texts) == 0:
            raise ValueError("Input texts list cannot be empty.")

        # Adjust n_topics if fewer documents than requested topics
        effective_topics = min(self.n_topics, len(texts))
        if effective_topics < 1:
            effective_topics = 1

        if self.method == "lda":
            self.vectorizer = CountVectorizer(
                stop_words="english",
                max_features=self.max_features,
                token_pattern=r"(?u)\b[a-zA-Z]{3,}\b"
            )
            count_matrix = self.vectorizer.fit_transform(texts)
            self.feature_names = self.vectorizer.get_feature_names_out()

            self.model = LatentDirichletAllocation(
                n_components=effective_topics,
                random_state=self.random_state,
                max_iter=50
            )
            self.topic_matrix = self.model.fit_transform(count_matrix)
        else:  # Default: NMF with TF-IDF
            self.vectorizer = TfidfVectorizer(
                stop_words="english",
                max_features=self.max_features,
                token_pattern=r"(?u)\b[a-zA-Z]{3,}\b"
            )
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            self.feature_names = self.vectorizer.get_feature_names_out()

            self.model = NMF(
                n_components=effective_topics,
                random_state=self.random_state,
                max_iter=config.NMF_MAX_ITER
            )
            self.topic_matrix = self.model.fit_transform(tfidf_matrix)

        return self.topic_matrix

    def get_topic_keywords(self, top_n: int = config.MAX_TOPIC_WORDS) -> Dict[int, List[str]]:
        """Return the top representative keywords for each discovered topic."""
        if self.model is None or self.feature_names is None or len(self.feature_names) == 0:
            return {}

        topics_keywords = {}
        for topic_idx, topic in enumerate(self.model.components_):
            # Sort component weights descending
            top_word_indices = topic.argsort()[-top_n:][::-1]
            keywords = [self.feature_names[i] for i in top_word_indices]
            topics_keywords[topic_idx] = keywords

        return topics_keywords

    def assign_topics(self, texts: List[str]) -> Tuple[List[int], List[float], List[str]]:
        """Assign dominant topic index, confidence weight, and topic label to each document."""
        if self.topic_matrix is None:
            self.fit_transform(texts)

        dominant_topics = []
        confidence_scores = []
        topic_keywords = self.get_topic_keywords(top_n=3)

        for doc_dist in self.topic_matrix:
            if doc_dist.sum() > 0:
                norm_dist = doc_dist / doc_dist.sum()
                top_idx = int(np.argmax(norm_dist))
                top_score = float(norm_dist[top_idx])
            else:
                top_idx = 0
                top_score = 0.0

            dominant_topics.append(top_idx)
            confidence_scores.append(round(top_score, 3))

        topic_labels = [
            f"Topic {idx + 1}: {', '.join(topic_keywords.get(idx, ['General']))}"
            for idx in dominant_topics
        ]

        return dominant_topics, confidence_scores, topic_labels
