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
        article_ids: Optional[List[str]] = None,
        source_datasets: Optional[List[str]] = None,
        sources: Optional[List[str]] = None
    ) -> List[int]:
        """Cluster input texts into candidate event groups.

        Returns list of integer cluster IDs (0-indexed).
        """
        n_samples = len(texts)
        if n_samples == 0:
            return []

        # Calculate TF-IDF cosine distance matrix
        self.similarity_engine.fit_transform(texts)
        distance_matrix = self.similarity_engine.get_distance_matrix()

        # If only 1 document, single cluster
        if n_samples == 1:
            self.labels_ = [0]
            self.cluster_summary_ = [{
                "cluster_internal_id": 0,
                "event_id": 1,
                "label": "Event 1: " + ((titles[0][:50] + "...") if titles else "Article 1"),
                "article_count": 1,
                "article_indices": [0],
                "article_ids": [article_ids[0]] if article_ids else ["art_001"],
                "titles": [titles[0]] if titles else ["Article 1"],
                "representative_title": titles[0] if titles else "Article 1",
                "source_datasets": [source_datasets[0]] if source_datasets else ["Primary Dataset"],
                "sources": [sources[0]] if sources else ["Unknown Source"],
                "is_cross_dataset": False,
                "avg_similarity": 1.0,
                "min_similarity": 1.0,
                "cohesion_level": "Single Article",
                "is_unrelated": False,
                "cohesion_note": "📌 Single isolated article; no cross-source matches found.",
                "extractive_summary": "",
                "abstractive_summary": "",
                "summary_status": "pending"
            }]
            return self.labels_

        # Perform Agglomerative Clustering
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage=self.linkage,
            distance_threshold=self.distance_threshold
        )
        self.labels_ = clusterer.fit_predict(distance_matrix).tolist()

        # Build detailed cluster summaries with cohesion analysis
        self._build_cluster_summary(texts, titles, article_ids, source_datasets, sources)
        return self.labels_

    def _build_cluster_summary(
        self,
        texts: List[str],
        titles: Optional[List[str]],
        article_ids: Optional[List[str]],
        source_datasets: Optional[List[str]],
        sources: Optional[List[str]]
    ):
        """Construct inspectable event cluster summaries with titles, cohesion, and dataset provenance."""
        clusters_dict: Dict[int, List[int]] = {}
        for idx, cluster_label in enumerate(self.labels_):
            clusters_dict.setdefault(cluster_label, []).append(idx)

        # Sort clusters by size (largest first)
        sorted_clusters = sorted(clusters_dict.items(), key=lambda kv: len(kv[1]), reverse=True)
        sim_matrix = self.similarity_engine.get_similarity_matrix()

        self.cluster_summary_ = []
        for rank, (c_id, member_indices) in enumerate(sorted_clusters, start=1):
            c_titles = [titles[i] if titles and i < len(titles) else f"Article {i}" for i in member_indices]
            c_art_ids = [article_ids[i] if article_ids and i < len(article_ids) else f"art_{i+1:03d}" for i in member_indices]
            c_datasets = [source_datasets[i] if source_datasets and i < len(source_datasets) else "Primary Dataset" for i in member_indices]
            c_sources = [sources[i] if sources and i < len(sources) else "Unknown Source" for i in member_indices]

            unique_datasets = sorted(list(set(c_datasets)))
            is_cross_dataset = len(unique_datasets) > 1

            # Intra-cluster cohesion calculation
            if len(member_indices) > 1 and sim_matrix is not None:
                sub_sim = sim_matrix[np.ix_(member_indices, member_indices)]
                n_m = len(member_indices)
                mask = ~np.eye(n_m, dtype=bool)
                pairwise_sims = sub_sim[mask]
                avg_sim = round(float(np.mean(pairwise_sims)), 3) if len(pairwise_sims) > 0 else 1.0
                min_sim = round(float(np.min(pairwise_sims)), 3) if len(pairwise_sims) > 0 else 1.0
            else:
                avg_sim = 1.0
                min_sim = 1.0

            # Cohesion and unrelatedness detection
            if len(member_indices) == 1:
                cohesion_level = "Single Article"
                is_unrelated = False
                cohesion_note = "📌 Single article cluster; no related articles discovered."
            elif avg_sim < 0.20 or min_sim < 0.10:
                cohesion_level = "Low Cohesion"
                is_unrelated = True
                cohesion_note = "⚠️ Low Cohesion: Articles in this cluster have weak semantic similarity and may describe unrelated events."
            elif avg_sim < 0.35:
                cohesion_level = "Moderate Cohesion"
                is_unrelated = False
                cohesion_note = "ℹ️ Moderate Cohesion: Loosely connected news stories."
            else:
                cohesion_level = "High Cohesion"
                is_unrelated = False
                cohesion_note = "✅ High Cohesion: Strong cross-source agreement on event details."

            rep_title = c_titles[0] if c_titles else "Untitled Event"

            summary = {
                "cluster_internal_id": int(c_id),
                "event_id": rank,
                "label": f"Event {rank}: {rep_title[:50]}...",
                "article_count": len(member_indices),
                "article_indices": member_indices,
                "article_ids": c_art_ids,
                "titles": c_titles,
                "representative_title": rep_title,
                "source_datasets": c_datasets,
                "unique_datasets": unique_datasets,
                "sources": c_sources,
                "is_cross_dataset": is_cross_dataset,
                "avg_similarity": avg_sim,
                "min_similarity": min_sim,
                "cohesion_level": cohesion_level,
                "is_unrelated": is_unrelated,
                "cohesion_note": cohesion_note,
                "extractive_summary": "",
                "abstractive_summary": "",
                "summary_status": "pending"
            }
            self.cluster_summary_.append(summary)

    def generate_all_summaries(
        self,
        articles_df: pd.DataFrame,
        extractive_engine=None,
        abstractive_engine=None,
        num_sentences: int = 3
    ):
        """Generate extractive (and abstractive if active) summaries for all clusters.

        Handles summarization failures gracefully.
        """
        from src.extractive_summarizer import ExtractiveSummarizer
        if extractive_engine is None:
            extractive_engine = ExtractiveSummarizer()

        for c in self.cluster_summary_:
            member_indices = c["article_indices"]
            cluster_articles = articles_df.iloc[member_indices].to_dict("records")

            # Extractive Summarization
            try:
                ext_res = extractive_engine.summarize_articles(cluster_articles, num_sentences=num_sentences)
                summary_text = ext_res.get("summary_text", "").strip()
                if not summary_text:
                    c["extractive_summary"] = "No summary generated (Text body empty or unsegmentable)."
                    c["summary_status"] = "failed: empty extractive output"
                else:
                    c["extractive_summary"] = summary_text
                    c["summary_status"] = "success"
            except Exception as e:
                c["extractive_summary"] = f"Summarization failed: {str(e)}"
                c["summary_status"] = f"failed: {str(e)}"

            # Abstractive Summarization (applied only if model is working/available)
            if abstractive_engine and getattr(abstractive_engine, "is_available", lambda: False)():
                try:
                    combined_text = " ".join(str(a.get("text", "")) for a in cluster_articles)
                    abs_res = abstractive_engine.summarize(combined_text)
                    c["abstractive_summary"] = abs_res.get("summary_text", "")
                except Exception as e:
                    c["abstractive_summary"] = f"Abstractive generation unavailable: {str(e)}"
            else:
                c["abstractive_summary"] = "N/A (Pretrained neural transformer not active)"

    def export_to_dataframe(self) -> pd.DataFrame:
        """Export clustered results and summaries to a clean pandas DataFrame for CSV download."""
        rows = []
        for c in self.cluster_summary_:
            rows.append({
                "Cluster_ID": c.get("event_id"),
                "Event_Label": c.get("label"),
                "Representative_Title": c.get("representative_title"),
                "Article_Count": c.get("article_count"),
                "Source_Datasets": ", ".join(c.get("unique_datasets", [])),
                "Is_Cross_Dataset": "Yes" if c.get("is_cross_dataset") else "No",
                "Average_Cosine_Similarity": c.get("avg_similarity"),
                "Cohesion_Level": c.get("cohesion_level"),
                "Is_Unrelated_Flag": "Unrelated / Outlier" if c.get("is_unrelated") else "Coherent Event",
                "Cohesion_Note": c.get("cohesion_note"),
                "Extractive_Summary": c.get("extractive_summary"),
                "Abstractive_Summary": c.get("abstractive_summary"),
                "Summary_Status": c.get("summary_status"),
                "Article_Titles": " | ".join(c.get("titles", []))
            })
        return pd.DataFrame(rows)

    def export_article_mapping_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Export article-level mapping with cluster assignment, source dataset, and summary."""
        records = []
        cluster_map = {c["event_id"]: c for c in self.cluster_summary_}

        for idx, row in df.iterrows():
            c_label = self.labels_[idx] if self.labels_ and idx < len(self.labels_) else 0
            # Find matching rank event_id
            matched_event = None
            for c in self.cluster_summary_:
                if idx in c["article_indices"]:
                    matched_event = c
                    break

            ev_id = matched_event.get("event_id") if matched_event else c_label + 1
            ev_title = matched_event.get("representative_title") if matched_event else "Unknown"
            ev_summary = matched_event.get("extractive_summary") if matched_event else ""
            ev_cohesion = matched_event.get("cohesion_level") if matched_event else "Unknown"

            records.append({
                "Article_ID": row.get("article_id", f"art_{idx+1:03d}"),
                "Title": row.get("title", ""),
                "Source_Publication": row.get("source", "Unknown"),
                "Source_Dataset": row.get("source_dataset", "Primary Dataset"),
                "Published_Date": row.get("published_date", "Unknown"),
                "Assigned_Event_ID": ev_id,
                "Event_Headline": ev_title,
                "Event_Cohesion": ev_cohesion,
                "Cluster_Extractive_Summary": ev_summary
            })
        return pd.DataFrame(records)

    def get_cluster_summary(self) -> List[Dict[str, Any]]:
        """Return structured summary of all candidate events."""
        return self.cluster_summary_

