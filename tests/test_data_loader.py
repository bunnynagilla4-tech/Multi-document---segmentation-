"""Unit tests for the NewsLens data loader and validation module."""

import unittest
from pathlib import Path
import pandas as pd
import io

from src.data_loader import (
    load_dataset,
    validate_dataset,
    clean_dataset,
    get_dataset_summary,
    combine_multiple_datasets,
    auto_map_columns
)
import config


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.sample_csv_path = config.SAMPLE_DATA_PATH

    def test_sample_csv_exists_and_loads(self):
        """Test that sample_news.csv exists and loads cleanly."""
        self.assertTrue(self.sample_csv_path.exists(), "Sample dataset file does not exist.")
        df = load_dataset(self.sample_csv_path, file_type="csv")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0, "Sample dataset is empty.")

    def test_validation_success(self):
        """Test validation on valid dataset."""
        df = pd.DataFrame({
            "title": ["Headline 1", "Headline 2"],
            "text": ["Article body text 1.", "Article body text 2."]
        })
        is_valid, errors = validate_dataset(df)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_validation_missing_columns(self):
        """Test validation failure when mandatory column is missing."""
        df = pd.DataFrame({"title": ["Only title"]})
        is_valid, errors = validate_dataset(df)
        self.assertFalse(is_valid)
        self.assertTrue(any("Missing required columns" in err for err in errors))

    def test_clean_dataset_removes_empty_and_duplicates(self):
        """Test missing row removal and duplicate suppression."""
        raw_df = pd.DataFrame({
            "title": ["Valid Title", "Empty Text", "", "Duplicate Title 1", "Duplicate Title 2"],
            "text": [
                "Valid article content.",
                "",
                "Has text but empty title.",
                "Identical article content.",
                "Identical article content."
            ]
        })
        cleaned, stats = clean_dataset(raw_df)

        # Expected:
        # - Row 0: kept
        # - Row 1: dropped (empty text)
        # - Row 2: dropped (empty title)
        # - Row 3: kept
        # - Row 4: dropped (duplicate text)
        self.assertEqual(stats["initial_rows"], 5)
        self.assertEqual(stats["missing_filtered"], 2)
        self.assertEqual(stats["duplicates_removed"], 1)
        self.assertEqual(stats["final_rows"], 2)
        self.assertEqual(len(cleaned), 2)
        self.assertIn("article_id", cleaned.columns)

    def test_summary_metrics(self):
        """Test dataset summary calculations."""
        df = pd.DataFrame({
            "title": ["Title A", "Title B"],
            "text": ["Ten words in this sentence for testing calculating average length properly.", "Short text here."],
            "source": ["Source 1", "Source 2"],
            "event_id": ["ev_1", "ev_1"]
        })
        summary = get_dataset_summary(df)
        self.assertEqual(summary["total_articles"], 2)
        self.assertEqual(summary["unique_sources"], 2)
        self.assertTrue(summary["has_event_labels"])
        self.assertGreater(summary["avg_word_count"], 0)

    def test_combine_multiple_datasets_success(self):
        """Test reading and combining multiple CSV buffers with cross-dataset deduplication."""
        csv1 = io.StringIO("title,text,source\nFlood in City,Water levels rose rapidly in suburbs.,News Daily\nQuantum Advance,Scientists built 1000 qubits.,Tech Wire\n")
        csv2 = io.StringIO("headline,content,source\nFlood in City,Water levels rose rapidly in suburbs.,City Press\nCricket Victory,India won the finals in Melbourne.,Sports News\n")

        # csv2 has a duplicate text with csv1 ("Water levels rose rapidly in suburbs.")
        # and has synonyms 'headline' and 'content' instead of 'title' and 'text'
        combined, reports, stats = combine_multiple_datasets([csv1, csv2], file_names=["reuters.csv", "bbc.csv"])

        self.assertEqual(len(reports), 2)
        self.assertEqual(reports[0]["status"], "valid")
        self.assertEqual(reports[1]["status"], "valid")
        self.assertEqual(stats["total_files"], 2)
        self.assertEqual(stats["valid_files"], 2)

        # 4 total rows in raw -> 1 cross-dataset duplicate removed -> 3 unique articles
        self.assertEqual(stats["duplicates_removed"], 1)
        self.assertEqual(len(combined), 3)
        self.assertIn("source_dataset", combined.columns)
        self.assertEqual(set(combined["source_dataset"]), {"reuters.csv", "bbc.csv"})

    def test_combine_multiple_datasets_with_invalid_file(self):
        """Test that invalid CSV files are caught, reported, and skipped while valid files succeed."""
        valid_csv = io.StringIO("title,text\nValid Article,This is a valid news article text.\n")
        invalid_csv = io.StringIO("id,random_column\n1,No required text columns here\n")

        combined, reports, stats = combine_multiple_datasets([valid_csv, invalid_csv], file_names=["valid.csv", "invalid.csv"])

        self.assertEqual(len(reports), 2)
        self.assertEqual(reports[0]["status"], "valid")
        self.assertEqual(reports[1]["status"], "invalid")
        self.assertIsNotNone(reports[1]["error"])
        self.assertEqual(len(combined), 1)
        self.assertEqual(stats["valid_files"], 1)


if __name__ == "__main__":
    unittest.main()

