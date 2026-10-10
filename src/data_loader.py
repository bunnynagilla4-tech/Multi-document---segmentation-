"""Data loader and validation module for NewsLens.
Handles reading CSV/JSON datasets, validation, missing value cleaning,
and duplicate detection.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Union, BinaryIO, TextIO
import io
import pandas as pd
import config


def load_dataset(
    source: Union[str, Path, BinaryIO, TextIO],
    file_type: str = "csv"
) -> pd.DataFrame:
    """Load news dataset from a file path or file buffer.

    Supports CSV and JSON formats with automatic fallback encoding detection.
    """
    if file_type == "csv":
        try:
            df = pd.read_csv(source, encoding="utf-8")
        except UnicodeDecodeError:
            if hasattr(source, "seek"):
                source.seek(0)
            df = pd.read_csv(source, encoding="latin-1")
    elif file_type == "json":
        df = pd.read_json(source)
    else:
        raise ValueError(f"Unsupported file type: {file_type}. Expected 'csv' or 'json'.")

    return df


COLUMN_SYNONYMS = {
    "title": ["title", "headline", "headlines", "heading", "head", "name", "subject", "article_title"],
    "text": ["text", "content", "body", "article", "story", "description", "news", "articles", "full_text", "article_text", "news_text"],
    "source": ["source", "publisher", "publication", "outlet", "author", "agency", "site"],
    "published_date": ["published_date", "date", "time", "timestamp", "datetime", "pub_date", "publish_date"],
    "event_id": ["event_id", "event", "cluster", "label", "target", "category"]
}


def auto_map_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Automatically detect and standardize common column name variations.

    Returns:
        (mapped_dataframe, dict_of_mappings_applied)
    """
    if df is None or df.empty:
        return df, {}

    df_copy = df.copy()
    if df_copy.columns.duplicated().any():
        df_copy = df_copy.loc[:, ~df_copy.columns.duplicated()].copy()

    col_map = {}
    lower_cols = {str(col).lower().strip(): col for col in df_copy.columns}

    for standard_col, synonyms in COLUMN_SYNONYMS.items():
        if standard_col not in df_copy.columns:
            for syn in synonyms:
                if syn in lower_cols and lower_cols[syn] not in col_map:
                    original_name = lower_cols[syn]
                    col_map[original_name] = standard_col
                    break

    if col_map:
        df_copy = df_copy.rename(columns=col_map)
        if df_copy.columns.duplicated().any():
            df_copy = df_copy.loc[:, ~df_copy.columns.duplicated()].copy()

    return df_copy, col_map


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, list]:
    """Validate that the dataset contains mandatory columns and is non-empty.

    Returns:
        (is_valid, list_of_error_or_warning_messages)
    """
    errors = []

    if df is None or df.empty:
        errors.append("Dataset is completely empty.")
        return False, errors

    missing_cols = [col for col in config.REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(
            f"Missing required columns: {missing_cols}. Dataset must include {config.REQUIRED_COLUMNS}."
        )
        return False, errors

    return True, errors


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean the raw news dataset.

    - Strips whitespace from text and title.
    - Filters out missing or empty titles/texts.
    - Drops duplicate articles.
    - Generates synthetic article_id if not present.
    - Standardizes optional fields.

    Returns:
        (cleaned_dataframe, statistics_dict)
    """
    initial_count = len(df)
    stats = {
        "initial_rows": initial_count,
        "missing_filtered": 0,
        "duplicates_removed": 0,
        "final_rows": 0,
    }

    # Work on a copy
    cleaned = df.copy()

    # Deduplicate column names if any exist (e.g. multiple 'text' or 'title' columns)
    if cleaned.columns.duplicated().any():
        cleaned = cleaned.loc[:, ~cleaned.columns.duplicated()].copy()

    # Ensure required columns are 1D Series (not DataFrame) and string types
    title_series = cleaned["title"]
    if isinstance(title_series, pd.DataFrame):
        title_series = title_series.iloc[:, 0]
    cleaned["title"] = title_series.astype(str).str.strip()

    text_series = cleaned["text"]
    if isinstance(text_series, pd.DataFrame):
        text_series = text_series.iloc[:, 0]
    cleaned["text"] = text_series.astype(str).str.strip()

    # Filter out empty or 'nan' string entries
    valid_mask = (
        (cleaned["title"] != "") &
        (cleaned["title"].str.lower() != "nan") &
        (cleaned["text"] != "") &
        (cleaned["text"].str.lower() != "nan")
    )
    missing_count = initial_count - valid_mask.sum()
    stats["missing_filtered"] = int(missing_count)
    cleaned = cleaned[valid_mask].copy()

    # Drop duplicate articles based on text content
    before_dedup = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    duplicates_count = before_dedup - len(cleaned)
    stats["duplicates_removed"] = int(duplicates_count)

    # Ensure article_id exists
    if "article_id" not in cleaned.columns:
        cleaned["article_id"] = [f"art_{i+1:03d}" for i in range(len(cleaned))]

    # Ensure other optional columns are present with sensible defaults
    if "source" not in cleaned.columns:
        cleaned["source"] = "Unknown Source"
    else:
        cleaned["source"] = cleaned["source"].fillna("Unknown Source")

    if "published_date" not in cleaned.columns:
        cleaned["published_date"] = "Unknown Date"
    else:
        cleaned["published_date"] = cleaned["published_date"].fillna("Unknown Date")

    # Ensure source_dataset exists if multi-dataset ingestion was performed
    if "source_dataset" not in cleaned.columns:
        cleaned["source_dataset"] = "Primary Dataset"
    else:
        cleaned["source_dataset"] = cleaned["source_dataset"].fillna("Primary Dataset")

    stats["final_rows"] = len(cleaned)
    return cleaned, stats


def combine_multiple_datasets(
    files: List[Any],
    file_names: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
    """Ingest, validate, and combine multiple news CSV files into a unified collection.

    Cross-dataset articles are pooled together into one unified collection,
    while tracking dataset provenance.

    Args:
        files: List of file paths, file-like buffers, or UploadedFile instances.
        file_names: Optional custom names for each file.

    Returns:
        (combined_cleaned_df, file_reports_list, aggregation_stats_dict)
    """
    valid_dfs = []
    file_reports = []
    total_raw_rows = 0

    for idx, f in enumerate(files):
        name = getattr(f, "name", None) or (
            file_names[idx] if file_names and idx < len(file_names) else f"dataset_{idx+1}.csv"
        )
        report = {
            "filename": name,
            "status": "pending",
            "raw_rows": 0,
            "valid_rows": 0,
            "error": None,
            "mappings": {}
        }

        try:
            if hasattr(f, "seek"):
                f.seek(0)
            raw = load_dataset(f, file_type="csv")
            report["raw_rows"] = len(raw)
            total_raw_rows += len(raw)

            # Auto-map column synonyms (e.g. headline->title, content->text)
            mapped_df, col_mappings = auto_map_columns(raw)
            is_valid, errors = validate_dataset(mapped_df)

            if not is_valid:
                report["status"] = "invalid"
                report["error"] = "; ".join(errors)
                file_reports.append(report)
                continue

            mapped_df["source_dataset"] = name
            valid_dfs.append(mapped_df)
            report["status"] = "valid"
            report["valid_rows"] = len(mapped_df)
            report["mappings"] = col_mappings

        except Exception as exc:
            report["status"] = "error"
            report["error"] = f"Failed to parse CSV: {str(exc)}"

        file_reports.append(report)

    if not valid_dfs:
        empty_df = pd.DataFrame(columns=["article_id", "title", "text", "source", "published_date", "source_dataset"])
        stats = {
            "total_files": len(files),
            "valid_files": 0,
            "total_raw_rows": 0,
            "missing_filtered": 0,
            "duplicates_removed": 0,
            "final_rows": 0
        }
        return empty_df, file_reports, stats

    # Combine into single unified DataFrame
    combined_raw = pd.concat(valid_dfs, ignore_index=True)

    # Clean combined collection (filters empty rows and cross-dataset duplicate articles)
    cleaned_df, clean_stats = clean_dataset(combined_raw)
    clean_stats["total_files"] = len(files)
    clean_stats["valid_files"] = len(valid_dfs)
    clean_stats["total_raw_rows"] = total_raw_rows

    return cleaned_df, file_reports, clean_stats


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate summary statistics for a cleaned news dataframe."""
    if df.empty:
        return {
            "total_articles": 0,
            "avg_word_count": 0,
            "sources_count": 0,
            "datasets_count": 0,
            "has_event_labels": False
        }

    word_counts = df["text"].apply(lambda t: len(str(t).split()))

    return {
        "total_articles": len(df),
        "avg_word_count": round(float(word_counts.mean()), 1),
        "min_word_count": int(word_counts.min()),
        "max_word_count": int(word_counts.max()),
        "unique_sources": int(df["source"].nunique()) if "source" in df.columns else 1,
        "unique_datasets": int(df["source_dataset"].nunique()) if "source_dataset" in df.columns else 1,
        "has_event_labels": "event_id" in df.columns and df["event_id"].notna().any()
    }

