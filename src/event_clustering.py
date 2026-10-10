"""Event Discovery and Article Clustering Module for NewsLens.
Groups related news articles into candidate events using Agglomerative Hierarchical Clustering
with precomputed cosine distance matrices.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from src.similarity import ArticleSimilarity
import config


class EventClusterer:
    """Discovers candidate news events by clustering articles based on semantic similarity."""

    def __init__(
        self,
        distance_threshold: float = config.DEFAULT_DISTANCE_THRESHOLD,
        linkage: str = config.CLUSTERING_LINKAGE
    ):
        self.distance_threshold = distance_threshold
        self.linkage = linkage
        self.similarity_engine = ArticleSimilarity()
        self.labels_ = None
        self.cluster_summary_ = []

    def fit_predict(
        self,
        texts: List[str],
        titles: Optional[List[str]] = None,
        article_ids: Optional[List[str]] = None
    ) -> List[int]:
        """Cluster input texts into candidate event groups.

        Returns list of integer cluster IDs (0-indexed).
        """
        n_samples = len(texts)
        if n_samples == 0:
            return []

        # If only 1 document, single cluster
        if n_samples == 1:
            self.labels_ = [0]
            self.cluster_summary_ = [{
                "event_id": 1,
                "label": "Event Group 1",
                "article_count": 1,
                "article_indices": [0],
                "article_ids": [article_ids[0]] if article_ids else ["art_001"],
                "representative_title": titles[0] if titles else "Article 1"
            }]
            return self.labels_

        # Calculate TF-IDF cosine distance matrix
        self.similarity_engine.fit_transform(texts)
        distance_matrix = self.similarity_engine.get_distance_matrix()

        # Perform Agglomerative Clustering
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage=self.linkage,
            distance_threshold=self.distance_threshold
        )
        self.labels_ = clusterer.fit_predict(distance_matrix).tolist()

        # Build detailed cluster summaries
        self._build_cluster_summary(texts, titles, article_ids)
        return self.labels_

    def _build_cluster_summary(
        self,
        texts: List[str],
        titles: Optional[List[str]],
        article_ids: Optional[List[str]]
    ):
        """Construct inspectable event cluster summaries with titles and statistics."""
        clusters_dict: Dict[int, List[int]] = {}
        for idx, cluster_label in enumerate(self.labels_):
            clusters_dict.setdefault(cluster_label, []).append(idx)

        # Sort clusters by size (largest first)
        sorted_clusters = sorted(clusters_dict.items(), key=lambda kv: len(kv[1]), reverse=True)

        self.cluster_summary_ = []
        for rank, (c_id, member_indices) in enumerate(sorted_clusters, start=1):
            c_titles = [titles[i] if titles and i < len(titles) else f"Article {i}" for i in member_indices]
            c_art_ids = [article_ids[i] if article_ids and i < len(article_ids) else f"art_{i+1:03d}" for i in member_indices]

            # Representative title: choose shortest title or first title
            rep_title = c_titles[0] if c_titles else "Untitled Event"

            summary = {
                "cluster_internal_id": int(c_id),
                "event_id": rank,
                "label": f"Event {rank}: {rep_title[:50]}...",
                "article_count": len(member_indices),
                "article_indices": member_indices,
                "article_ids": c_art_ids,
                "titles": c_titles,
                "representative_title": rep_title
            }
            self.cluster_summary_.append(summary)

    def get_cluster_summary(self) -> List[Dict[str, Any]]:
        """Return structured summary of all candidate events."""
        return self.cluster_summary_
