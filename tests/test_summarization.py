"""Unit tests for summarization (extractive, abstractive fallback, hybrid) and evaluation."""

import unittest
from src.extractive_summarizer import ExtractiveSummarizer
from src.abstractive_summarizer import AbstractiveSummarizer
from src.hybrid_summarizer import HybridSummarizer
from src.evaluation import evaluate_summary, evaluate_clustering


class TestSummarizationAndEvaluation(unittest.TestCase):
    def setUp(self):
        self.sample_articles = [
            {
                "article_id": "art_01",
                "title": "Severe Rains Hit City",
                "source": "News Wire",
                "text": "Torrential monsoon rains hit the metropolitan area on Monday morning. Major railway lines were submerged under three feet of water. Municipal authorities issued an urgent red alert across low-lying districts."
            },
            {
                "article_id": "art_02",
                "title": "Floods Halt Transit",
                "source": "Daily Press",
                "text": "Commuter train operations ground to a halt as relentless rains submerged railway tracks. Rescue teams evacuated stranded commuters from suburban stations. Schools and offices remained shut following the municipal red alert."
            }
        ]

    def test_extractive_summarizer(self):
        """Test sentence selection and source attribution."""
        summarizer = ExtractiveSummarizer()
        result = summarizer.summarize_articles(self.sample_articles, num_sentences=2)

        self.assertIn("summary_text", result)
        self.assertIn("sentences", result)
        self.assertEqual(len(result["sentences"]), 2)

        # Check provenance tracking
        first_sent = result["sentences"][0]
        self.assertIn("article_id", first_sent)
        self.assertIn("title", first_sent)
        self.assertIn("source", first_sent)
        self.assertIn("importance_score", first_sent)
        self.assertGreater(result["summary_word_count"], 0)

    def test_abstractive_fallback(self):
        """Test abstractive fallback behaves deterministically without crashing."""
        summarizer = AbstractiveSummarizer()
        result = summarizer.summarize("This is a lead sentence about news. This is a supporting detail.")

        self.assertIn("summary_text", result)
        self.assertIn("status", result)
        self.assertTrue(len(result["summary_text"]) > 0)

    def test_hybrid_summarizer(self):
        """Test hybrid two-stage summarization."""
        hybrid = HybridSummarizer()
        res = hybrid.summarize_event(self.sample_articles, num_extracted_sentences=2)

        self.assertIn("hybrid_summary", res)
        self.assertIn("supporting_passages", res)
        self.assertGreater(len(res["supporting_passages"]), 0)
        self.assertGreater(res["hybrid_word_count"], 0)

    def test_evaluation_with_reference(self):
        """Test ROUGE evaluation when reference summary is provided."""
        gen = "Torrential rains caused floods and halted railway transit in the city."
        ref = "Severe rains caused flooding and suspended city railway operations."
        eval_res = evaluate_summary(gen, ref)

        self.assertTrue(eval_res["has_reference"])
        self.assertIn("ROUGE-1", eval_res["metrics"])
        self.assertGreater(eval_res["metrics"]["ROUGE-1"]["f1"], 0.0)

    def test_evaluation_without_reference(self):
        """Test evaluation explains reference requirement when reference is omitted."""
        gen = "Some generated summary without any reference provided."
        eval_res = evaluate_summary(gen, reference_summary=None)

        self.assertFalse(eval_res["has_reference"])
        self.assertIn("note", eval_res)

    def test_clustering_evaluation(self):
        """Test Adjusted Rand Index on ground truth labels."""
        true_labels = [0, 0, 1, 1]
        pred_labels = [1, 1, 0, 0]  # Perfect clustering with permuted cluster labels
        eval_res = evaluate_clustering(true_labels, pred_labels)

        self.assertTrue(eval_res["has_ground_truth"])
        self.assertEqual(eval_res["metrics"]["Adjusted_Rand_Index_ARI"], 1.0)


if __name__ == "__main__":
    unittest.main()
