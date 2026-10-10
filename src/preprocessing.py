"""Text Preprocessing Pipeline for NewsLens.
Provides normalization, tokenization, stopword removal, sentence splitting,
keyword extraction, and rule-based named entity recognition while preserving
original texts and crucial event entities (numbers, dates, names).
"""

import re
import html
from typing import List, Dict, Any, Tuple
import pandas as pd

# Standard English stopwords fallback list to ensure zero-dependency reliability
DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more",
    "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
    "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
    "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
    "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
    "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves"
}


def clean_text_basic(text: str) -> str:
    """Normalize text: unescape HTML, remove URLs, strip tags, and collapse excess whitespace.

    Preserves case, punctuation, numbers, and dates to allow entity and sentence extraction.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    # Unescape HTML entities (e.g. &amp;, &quot;)
    text = html.unescape(text)

    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Normalize multiple whitespaces and linebreaks
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using regex boundary lookaheads.

    Avoids false splits on common abbreviations like Dr., Prof., U.S., etc.
    """
    if not text:
        return []

    # Clean text first
    cleaned = clean_text_basic(text)

    # Protect common honorifics and abbreviations
    protected = re.sub(r"\b(Dr|Mr|Mrs|Ms|Prof|Gov|Gen|U\.S|U\.K|Inc|Corp|Ltd)\.", r"\1<DOT>", cleaned)

    # Split on sentence boundaries: period, exclamation, question mark followed by space or end
    raw_sentences = re.split(r"(?<=[.!?])\s+", protected)

    sentences = []
    for s in raw_sentences:
        s = s.replace("<DOT>", ".").strip()
        if len(s) > 3:  # Discard stray punctuation fragments
            sentences.append(s)

    return sentences


def tokenize_words(
    text: str,
    remove_stopwords: bool = True,
    lowercase: bool = True,
    min_length: int = 2
) -> List[str]:
    """Tokenize text into alphanumeric words with optional stopword filtering."""
    if not text:
        return []

    cleaned = clean_text_basic(text)
    if lowercase:
        cleaned = cleaned.lower()

    # Extract alphanumeric word tokens (including hyphens inside words like 1000-qubit)
    tokens = re.findall(r"\b[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*\b", cleaned)

    if remove_stopwords:
        tokens = [t for t in tokens if t not in DEFAULT_STOPWORDS and len(t) >= min_length]
    else:
        tokens = [t for t in tokens if len(t) >= min_length]

    return tokens


def extract_entities(text: str) -> List[Dict[str, str]]:
    """Lightweight, rule-based named entity and key detail recognizer.

    Extracts:
    - QUANTITY / NUMERIC: quantities, amounts, measurements (e.g., '180 millimeters', '$80 billion')
    - TEMPORAL: dates, days, years (e.g., 'Monday morning', '2035', 'Friday')
    - PROPER_NOUN / ENTITY: capitalized multi-word phrases and key organizations
    """
    if not text:
        return []

    entities = []
    seen = set()

    # 1. Quantities and Currencies
    quant_pattern = r"(?:\$|\€|\£)?\b\d+(?:,\d+)*(?:\.\d+)?\s*(?:billion|million|percent|millimeters|mm|qubit|runs|wickets|hours)?\b"
    for match in re.finditer(quant_pattern, text, re.IGNORECASE):
        val = match.group().strip()
        if val and val not in seen and any(c.isdigit() for c in val):
            seen.add(val)
            entities.append({"text": val, "label": "CARDINAL/MEASURE"})

    # 2. Days, Months, and Years
    date_pattern = r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|January|February|March|April|May|June|July|August|September|October|November|December|\d{4}-\d{2}-\d{2}|\b20\d\d\b)(?:\s+(?:morning|afternoon|evening|night))?\b"
    for match in re.finditer(date_pattern, text, re.IGNORECASE):
        val = match.group().strip()
        if val and val not in seen:
            seen.add(val)
            entities.append({"text": val, "label": "TEMPORAL"})

    # 3. Capitalized Proper Names / Multi-Word Entities (excluding sentence starters)
    # Match sequences of Capitalized words
    propn_pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"
    for match in re.finditer(propn_pattern, text):
        val = match.group().strip()
        if val and val not in seen:
            seen.add(val)
            entities.append({"text": val, "label": "ENTITY/ORG/LOC"})

    return entities


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full preprocessing pipeline to a news DataFrame.

    Adds:
    - `clean_text`: Normalized text with HTML/URLs stripped
    - `tokens`: Filtered lowercase tokens for analysis
    - `sentences`: Extracted sentence list for extractive summarization
    - `entities`: Extracted entities
    - `processed_text`: Space-joined tokens suitable for TF-IDF / vectorization
    """
    processed = df.copy()

    processed["clean_text"] = processed["text"].apply(clean_text_basic)
    processed["sentences"] = processed["clean_text"].apply(split_sentences)
    processed["tokens"] = processed["clean_text"].apply(lambda t: tokenize_words(t, remove_stopwords=True))
    processed["processed_text"] = processed["tokens"].apply(lambda tok_list: " ".join(tok_list))
    processed["entities"] = processed["text"].apply(extract_entities)

    return processed
