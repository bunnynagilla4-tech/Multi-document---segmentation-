import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import io
import pandas as pd
import config
from src.data_loader import load_dataset, validate_dataset, clean_dataset
from src.preprocessing import preprocess_dataframe
from src.topic_modeling import TopicModeler
from src.event_clustering import EventClusterer
from src.extractive_summarizer import ExtractiveSummarizer
from src.abstractive_summarizer import AbstractiveSummarizer

def test_news_sample_pipeline():
    print("=" * 60)
    print("TESTING USER news_sample.csv UPLOAD AND PIPELINE EXECUTION")
    print("=" * 60)
    
    # 1. Load dataset
    raw_df = load_dataset('news_sample.csv')
    print(f"Loaded raw dataset shape: {raw_df.shape}")
    print(f"Columns in dataset: {list(raw_df.columns)}")

    all_cols = list(raw_df.columns)

    # 2. Match column selector logic from app.py
    def find_default_col(candidates, default_val=0):
        for cand in candidates:
            for idx, c in enumerate(all_cols):
                c_lower = str(c).lower().strip()
                if c_lower == cand:
                    return idx
        for cand in candidates:
            for idx, c in enumerate(all_cols):
                c_lower = str(c).lower().strip()
                if "unnamed" in c_lower or c_lower in ["id", "index"]:
                    continue
                if cand in c_lower:
                    return idx
        return default_val

    default_title_idx = find_default_col(["title", "headline", "headlines", "heading", "name", "subject", "article_title"], 0)
    default_text_idx = find_default_col(["text", "content", "body", "article", "story", "description", "news", "articles", "full_text"], min(1, len(all_cols) - 1))

    selected_title = all_cols[default_title_idx]
    selected_text = all_cols[default_text_idx]
    print(f"Default Title Column Selected: '{selected_title}' (Index: {default_title_idx})")
    print(f"Default Text Column Selected:  '{selected_text}' (Index: {default_text_idx})")

    assert selected_title == 'title', f"Expected title, got {selected_title}"
    assert selected_text == 'content', f"Expected content, got {selected_text}"

    # 3. Simulate App Ingestion & Mapping
    title_col_data = raw_df[selected_title]
    if isinstance(title_col_data, pd.DataFrame):
        title_col_data = title_col_data.iloc[:, 0]

    text_col_data = raw_df[selected_text]
    if isinstance(text_col_data, pd.DataFrame):
        text_col_data = text_col_data.iloc[:, 0]

    new_data = {
        'title': title_col_data,
        'text': text_col_data
    }

    if 'domain' in raw_df.columns:
        domain_data = raw_df['domain']
        new_data['source'] = domain_data.iloc[:, 0] if isinstance(domain_data, pd.DataFrame) else domain_data

    mapped_df = pd.DataFrame(new_data)
    mapped_df['source_dataset'] = 'news_sample.csv'

    print(f"Mapped DataFrame shape: {mapped_df.shape}")

    # 4. Schema Validation
    is_valid, errors = validate_dataset(mapped_df)
    print(f"Validation Result: is_valid={is_valid}, errors={errors}")
    assert is_valid, f"Validation failed: {errors}"

    # 5. Data Cleaning & Deduplication
    cleaned_df, clean_stats = clean_dataset(mapped_df)
    print(f"Data Cleaning Stats: {clean_stats}")
    print(f"Cleaned DataFrame shape: {cleaned_df.shape}")
    assert len(cleaned_df) > 0, "Cleaned dataset is empty!"

    # 6. Linguistic Preprocessing
    print("Running Linguistic Preprocessing...")
    df_proc = preprocess_dataframe(cleaned_df)
    print(f"Preprocessing completed for {len(df_proc)} articles.")
    assert 'clean_text' in df_proc.columns
    assert 'tokens' in df_proc.columns

    # 7. Topic Modeling
    print("Running NMF Topic Modeling...")
    modeler = TopicModeler(n_topics=config.DEFAULT_NUM_TOPICS, method='nmf')
    modeler.fit_transform(df_proc['processed_text'].tolist())
    dominant_topics, conf_scores, topic_labels = modeler.assign_topics(df_proc['processed_text'].tolist())
    df_proc['topic_id'] = dominant_topics
    df_proc['topic_label'] = topic_labels
    df_proc['topic_confidence'] = conf_scores
    print(f"Topic modeling completed. Assigned {len(dominant_topics)} topics.")

    # 8. Event Clustering
    print("Running Event Clustering...")
    clusterer = EventClusterer(distance_threshold=config.DEFAULT_DISTANCE_THRESHOLD)
    cluster_labels = clusterer.fit_predict(
        df_proc['processed_text'].tolist(),
        titles=df_proc['title'].tolist(),
        article_ids=df_proc['article_id'].tolist() if 'article_id' in df_proc.columns else None,
        source_datasets=df_proc['source_dataset'].tolist() if 'source_dataset' in df_proc.columns else None,
        sources=df_proc['source'].tolist() if 'source' in df_proc.columns else None
    )
    df_proc['cluster_id'] = cluster_labels
    clusters = clusterer.get_cluster_summary()
    print(f"Event clustering completed. Discovered {len(clusters)} candidate clusters.")

    # 9. Multi-Document Summarization
    print("Generating Extractive Summaries for all clusters...")
    ext_sum = ExtractiveSummarizer()
    clusterer.generate_all_summaries(
        df_proc,
        extractive_engine=ext_sum,
        num_sentences=config.DEFAULT_EXTRACTIVE_SENTENCES
    )
    print("Summaries generated successfully!")
    
    # 10. CSV Export
    summary_df = clusterer.export_to_dataframe()
    mapping_df = clusterer.export_article_mapping_dataframe(cleaned_df)
    print(f"Cluster Summaries Export shape: {summary_df.shape}")
    print(f"Article Mapping Export shape: {mapping_df.shape}")

    print("\nSAMPLE DISCOVERED EVENT CLUSTERS:")
    for c in clusters[:5]:
        print(f"  [Cluster {c['event_id']}] ({c['article_count']} articles, Cohesion: {c['cohesion_level']})")
        print(f"   Representative Title: {c['representative_title']}")
        print(f"   Summary: {c['extractive_summary'][:100]}...\n")

    print("=" * 60)
    print("ALL TESTS PASSED! news_sample.csv WORKS 100% WITH ZERO ERRORS!")
    print("=" * 60)

if __name__ == '__main__':
    test_news_sample_pipeline()
