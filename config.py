"""NewsLens Configuration Module
Contains system constants, default parameters, and configuration options.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"
DOCS_DIR = BASE_DIR / "docs"
SAMPLE_DATA_PATH = DATA_DIR / "sample_news.csv"

# Dataset Schema
REQUIRED_COLUMNS = ["title", "text"]
OPTIONAL_COLUMNS = ["article_id", "source", "published_date", "url", "event_id", "reference_summary"]

# Preprocessing Configuration
DEFAULT_STOPWORDS_LANGUAGE = "english"
MIN_TOKEN_LENGTH = 2
MAX_VOCAB_SIZE = 5000

# Topic Modeling Configuration
DEFAULT_NUM_TOPICS = 4
MAX_TOPIC_WORDS = 8
NMF_MAX_ITER = 500
RANDOM_STATE = 42

# Clustering & Similarity Configuration
DEFAULT_SIMILARITY_THRESHOLD = 0.25  # Minimum TF-IDF cosine similarity to be considered related
DEFAULT_DISTANCE_THRESHOLD = 0.75    # Agglomerative clustering distance threshold (1 - similarity)
CLUSTERING_LINKAGE = "average"       # Options: 'average', 'complete', 'single'

# Summarization Configuration
DEFAULT_EXTRACTIVE_SENTENCES = 3
DEFAULT_SUMMARY_MAX_LENGTH = 130
DEFAULT_SUMMARY_MIN_LENGTH = 30
TRANSFORMER_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"  # Compact, high-performing CNN/DailyMail model
LIGHTWEIGHT_FALLBACK_MODEL = "extractive"                 # Used if PyTorch/Transformers unavailable

# Evaluation Settings
ROUGE_METRICS = ["rouge1", "rouge2", "rougeL"]
