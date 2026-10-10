"""Hybrid Summarization Module for NewsLens.
Combines extractive salience filtering with abstractive synthesis to produce
concise, non-redundant event summaries while preserving source provenance.
"""

from typing import List, Dict, Any
from src.extractive_summarizer import ExtractiveSummarizer
from src.abstractive_summarizer import AbstractiveSummarizer
import config


class HybridSummarizer:
    """Two-stage hybrid summarization pipeline:

    Stage 1: Multi-document extractive filtering distills key facts from all articles
             and filters out cross-source redundancies.
    Stage 2: Abstractive synthesis condenses and smooths the extracted factual sentences
             into a coherent summary.
    Provenance: Maintains direct trace links to supporting source passages.
    """

    def __init__(
        self,
        extractive_engine: ExtractiveSummarizer = None,
        abstractive_engine: AbstractiveSummarizer = None
    ):
        self.extractive_engine = extractive_engine or ExtractiveSummarizer()
        self.abstractive_engine = abstractive_engine or AbstractiveSummarizer()

    def summarize_event(
        self,
        articles: List[Dict[str, Any]],
        num_extracted_sentences: int = 4
    ) -> Dict[str, Any]:
        """Generate a hybrid summary for a candidate news event."""
        if not articles:
            return {
                "hybrid_summary": "",
                "extractive_summary": "",
                "supporting_passages": [],
                "compression_ratio": 0.0,
                "provenance": []
            }

        # Stage 1: Extractive Filtering
        extractive_result = self.extractive_engine.summarize_articles(
            articles,
            num_sentences=num_extracted_sentences
        )
        extracted_text = extractive_result.get("summary_text", "")
        supporting_sentences = extractive_result.get("sentences", [])

        # Stage 2: Abstractive Condensation
        abstractive_result = self.abstractive_engine.summarize(extracted_text)
        final_summary = abstractive_result.get("summary_text", extracted_text)

        # Build provenance mapping
        provenance = []
        for s in supporting_sentences:
            provenance.append({
                "article_id": s.get("article_id", ""),
                "title": s.get("title", ""),
                "source": s.get("source", ""),
                "passage": s.get("text", "")
            })

        total_orig_words = sum(len(a.get("text", "").split()) for a in articles)
        summary_words = len(final_summary.split())
        ratio = round(summary_words / max(total_orig_words, 1), 3)

        return {
            "hybrid_summary": final_summary,
            "extractive_intermediate": extracted_text,
            "supporting_passages": provenance,
            "compression_ratio": ratio,
            "original_word_count": total_orig_words,
            "hybrid_word_count": summary_words,
            "abstractive_status": abstractive_result.get("status"),
            "method_notes": "Extractive salience filtering + Abstractive synthesis with full source attribution."
        }
