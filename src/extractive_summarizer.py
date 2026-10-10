"""Extractive Summarization Module for NewsLens.
Selects salient sentences across multiple news articles using TF-IDF centrality,
lead-sentence weighting, and Maximal Marginal Relevance (MMR) redundancy filtering.
Maintains full traceability back to source articles and sentence indices.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing import split_sentences, clean_text_basic
import config


class ExtractiveSummarizer:
    """Multi-document extractive summarizer with MMR diversity and source attribution."""

    def __init__(
        self,
        redundancy_threshold: float = 0.65,
        lead_bias_weight: float = 0.15
    ):
        self.redundancy_threshold = redundancy_threshold
        self.lead_bias_weight = lead_bias_weight

    def summarize_articles(
        self,
        articles: List[Dict[str, Any]],
        num_sentences: int = config.DEFAULT_EXTRACTIVE_SENTENCES
    ) -> Dict[str, Any]:
        """Generate extractive summary from multiple articles of an event group.

        Each item in `articles` must have at least:
        `text` (str), and optionally `title` (str), `article_id` (str), `source` (str).

        Returns:
            {
                "summary_text": str,
                "sentences": List[Dict[str, Any]],  # Traceable sentence records
                "compression_ratio": float,
                "original_word_count": int,
                "summary_word_count": int
            }
        """
        all_sentences = []

        # 1. Gather all sentences across all articles with source tracking
        for art_idx, art in enumerate(articles):
            raw_text = art.get("text", "")
            title = art.get("title", f"Article {art_idx + 1}")
            art_id = art.get("article_id", f"art_{art_idx + 1}")
            source = art.get("source", "Unknown Source")

            sentences = split_sentences(raw_text)
            total_art_sents = len(sentences)

            for sent_idx, sent in enumerate(sentences):
                clean_sent = clean_text_basic(sent)
                if len(clean_sent.split()) < 5:  # Skip very short fragments
                    continue

                # Position bias: earlier sentences in news articles carry higher lead value
                pos_weight = 1.0 - (sent_idx / max(total_art_sents, 1)) * self.lead_bias_weight

                all_sentences.append({
                    "text": clean_sent,
                    "article_id": art_id,
                    "title": title,
                    "source": source,
                    "sentence_index": sent_idx,
                    "pos_weight": pos_weight
                })

        if not all_sentences:
            return {
                "summary_text": "",
                "sentences": [],
                "compression_ratio": 0.0,
                "original_word_count": 0,
                "summary_word_count": 0
            }

        # If total candidate sentences is <= requested length, return all
        if len(all_sentences) <= num_sentences:
            selected_sents = all_sentences
            summary_text = " ".join(s["text"] for s in selected_sents)
            orig_words = sum(len(a.get("text", "").split()) for a in articles)
            sum_words = len(summary_text.split())
            ratio = round(sum_words / max(orig_words, 1), 3)
            return {
                "summary_text": summary_text,
                "sentences": selected_sents,
                "compression_ratio": ratio,
                "original_word_count": orig_words,
                "summary_word_count": sum_words
            }

        # 2. Vectorize sentences using TF-IDF
        candidate_texts = [s["text"] for s in all_sentences]
        vectorizer = TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b[a-zA-Z0-9-]{3,}\b")
        try:
            sent_tfidf = vectorizer.fit_transform(candidate_texts)
        except ValueError:
            # Fallback if no valid tokens
            selected_sents = all_sentences[:num_sentences]
            summary_text = " ".join(s["text"] for s in selected_sents)
            return {
                "summary_text": summary_text,
                "sentences": selected_sents,
                "compression_ratio": 0.5,
                "original_word_count": 100,
                "summary_word_count": len(summary_text.split())
            }

        # Document centroid (average TF-IDF vector of all sentences in the event)
        doc_centroid = sent_tfidf.mean(axis=0)
        # Convert matrix to 2D numpy array for cosine similarity
        centroid_arr = np.asarray(doc_centroid)

        # Base relevance scores: cosine similarity of each sentence to the event centroid
        base_relevance = cosine_similarity(sent_tfidf, centroid_arr).ravel()

        # Apply position bias
        for i, sent_meta in enumerate(all_sentences):
            base_relevance[i] *= sent_meta["pos_weight"]

        # 3. Maximal Marginal Relevance (MMR) Selection
        pairwise_sim = cosine_similarity(sent_tfidf)
        selected_indices: List[int] = []
        unselected_indices = list(range(len(all_sentences)))

        # Pick the highest scoring sentence first
        first_idx = int(np.argmax(base_relevance))
        selected_indices.append(first_idx)
        unselected_indices.remove(first_idx)

        # Iteratively pick remaining sentences
        while len(selected_indices) < num_sentences and unselected_indices:
            best_mmr = -float("inf")
            best_cand = None

            for cand_idx in unselected_indices:
                rel = base_relevance[cand_idx]
                max_redundancy = max(pairwise_sim[cand_idx, sel_idx] for sel_idx in selected_indices)

                # Penalize candidates that exceed redundancy threshold
                if max_redundancy >= self.redundancy_threshold:
                    penalty = 0.8 * max_redundancy
                else:
                    penalty = 0.3 * max_redundancy

                mmr_score = rel - penalty

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_cand = cand_idx

            if best_cand is not None:
                selected_indices.append(best_cand)
                unselected_indices.remove(best_cand)
            else:
                break

        # Sort selected sentences to preserve logical narrative ordering (by article then sentence_index)
        selected_indices = sorted(selected_indices)

        selected_sents = []
        for idx in selected_indices:
            item = dict(all_sentences[idx])
            item["importance_score"] = round(float(base_relevance[idx]), 3)
            selected_sents.append(item)

        summary_text = " ".join(s["text"] for s in selected_sents)
        orig_words = sum(len(a.get("text", "").split()) for a in articles)
        sum_words = len(summary_text.split())
        ratio = round(sum_words / max(orig_words, 1), 3)

        return {
            "summary_text": summary_text,
            "sentences": selected_sents,
            "compression_ratio": ratio,
            "original_word_count": orig_words,
            "summary_word_count": sum_words
        }
