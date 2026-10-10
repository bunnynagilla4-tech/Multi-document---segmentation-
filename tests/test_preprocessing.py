"""Unit tests for NewsLens text preprocessing pipeline."""

import unittest
import pandas as pd
from src.preprocessing import (
    clean_text_basic,
    split_sentences,
    tokenize_words,
    extract_entities,
    preprocess_dataframe
)


class TestPreprocessing(unittest.TestCase):
    def test_clean_text_basic(self):
        """Test HTML stripping, URL removal, and whitespace handling."""
        raw = "   <p>Breaking news on <a href='https://news.org'>this link</a>! &amp; more details.   "
        cleaned = clean_text_basic(raw)
        self.assertEqual(cleaned, "Breaking news on this link ! & more details.")

    def test_split_sentences(self):
        """Test sentence splitting with honorifics preservation."""
        text = "Dr. Vance presented the chip. It achieved 99.8% accuracy! Can this change computing? Absolutely."
        sentences = split_sentences(text)
        self.assertEqual(len(sentences), 4)
        self.assertIn("Dr. Vance presented the chip.", sentences[0])
        self.assertIn("It achieved 99.8% accuracy!", sentences[1])

    def test_tokenize_words(self):
        """Test tokenization and stopword removal."""
        text = "The quick brown fox jumps over the lazy dog."
        tokens = tokenize_words(text, remove_stopwords=True)
        self.assertNotIn("the", tokens)
        self.assertNotIn("over", tokens)
        self.assertIn("quick", tokens)
        self.assertIn("brown", tokens)

    def test_extract_entities(self):
        """Test rule-based entity recognition for quantities, dates, and locations."""
        text = "On Monday morning, Dr. Sarah Vance announced a $80 billion fund in Melbourne."
        entities = extract_entities(text)
        labels = [e["label"] for e in entities]
        entity_texts = [e["text"] for e in entities]

        self.assertIn("TEMPORAL", labels)
        self.assertTrue(any("Monday" in t for t in entity_texts))
        self.assertTrue(any("$80 billion" in t for t in entity_texts))

    def test_preprocess_dataframe(self):
        """Test preprocessing on a complete DataFrame."""
        df = pd.DataFrame({
            "title": ["Test Headline"],
            "text": ["First sentence about science. Second sentence with details."]
        })
        processed = preprocess_dataframe(df)

        self.assertIn("clean_text", processed.columns)
        self.assertIn("tokens", processed.columns)
        self.assertIn("sentences", processed.columns)
        self.assertIn("entities", processed.columns)
        self.assertIn("processed_text", processed.columns)
        self.assertEqual(len(processed.loc[0, "sentences"]), 2)


if __name__ == "__main__":
    unittest.main()
