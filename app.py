"""NewsLens — Multi-Document Summarization for News Events Using NLP
Main Streamlit Application.
"""

import io
import time
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

import config
from src.data_loader import load_dataset, validate_dataset, clean_dataset, get_dataset_summary
from src.preprocessing import preprocess_dataframe
from src.topic_modeling import TopicModeler
from src.similarity import ArticleSimilarity
from src.event_clustering import EventClusterer
from src.extractive_summarizer import ExtractiveSummarizer
from src.abstractive_summarizer import AbstractiveSummarizer
from src.hybrid_summarizer import HybridSummarizer
from src.evaluation import evaluate_summary, evaluate_clustering, PerformanceBenchmark

# Streamlit Page Config
st.set_page_config(
    page_title="NewsLens — Multi-Document News Summarizer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished newsroom aesthetics
st.markdown("""
<style>
    .main-title {
        font-family: 'Georgia', serif;
        font-size: 2.3rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .event-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #e0f2fe;
        color: #0369a1;
        margin-right: 6px;
    }
    .provenance-box {
        background: #f1f5f9;
        border-left: 4px solid #3b82f6;
        padding: 12px;
        margin-bottom: 10px;
        border-radius: 0 6px 6px 0;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session():
    """Initialize session state variables."""
    if "df_clean" not in st.session_state:
        # Auto-load sample dataset by default for immediate out-of-the-box demonstration
        if config.SAMPLE_DATA_PATH.exists():
            df_raw = load_dataset(config.SAMPLE_DATA_PATH, file_type="csv")
            cleaned, _ = clean_dataset(df_raw)
            st.session_state["df_clean"] = cleaned
            _run_nlp_pipeline(cleaned)
        else:
            st.session_state["df_clean"] = pd.DataFrame()
            st.session_state["df_processed"] = pd.DataFrame()
            st.session_state["clusters"] = []


def _run_nlp_pipeline(df: pd.DataFrame):
    """Run full NLP pipeline: Preprocessing, Topic Modeling, and Event Clustering."""
    benchmark = PerformanceBenchmark()

    # Stage 1: Preprocessing
    benchmark.start_stage("preprocessing")
    df_proc = preprocess_dataframe(df)
    benchmark.end_stage("preprocessing")
    st.session_state["df_processed"] = df_proc

    # Stage 2: Topic Modeling
    benchmark.start_stage("topic_modeling")
    modeler = TopicModeler(n_topics=config.DEFAULT_NUM_TOPICS, method="nmf")
    modeler.fit_transform(df_proc["processed_text"].tolist())
    dominant_topics, conf_scores, topic_labels = modeler.assign_topics(df_proc["processed_text"].tolist())
    df_proc["topic_id"] = dominant_topics
    df_proc["topic_label"] = topic_labels
    df_proc["topic_confidence"] = conf_scores
    benchmark.end_stage("topic_modeling")
    st.session_state["topic_modeler"] = modeler

    # Stage 3: Similarity & Clustering
    benchmark.start_stage("clustering")
    clusterer = EventClusterer(distance_threshold=config.DEFAULT_DISTANCE_THRESHOLD)
    cluster_labels = clusterer.fit_predict(
        df_proc["processed_text"].tolist(),
        titles=df_proc["title"].tolist(),
        article_ids=df_proc["article_id"].tolist()
    )
    df_proc["cluster_id"] = cluster_labels
    benchmark.end_stage("clustering")

    st.session_state["clusterer"] = clusterer
    st.session_state["clusters"] = clusterer.get_cluster_summary()
    st.session_state["benchmark"] = benchmark.get_summary(len(df))


init_session()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/news.png", width=64)
st.sidebar.title("NewsLens")
st.sidebar.caption("Multi-Document News Summarizer")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home Dashboard",
        "📤 Upload Dataset",
        "🧹 Preprocessing Pipeline",
        "🧠 Topic Modeling",
        "🔍 Event Discovery",
        "📝 Summarization Studio",
        "📊 Evaluation & Metrics",
        "ℹ️ About & Methodology"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Runtime Info:**")
curr_arts = len(st.session_state.get("df_clean", []))
curr_clusters = len(st.session_state.get("clusters", []))
st.sidebar.text(f"Articles loaded: {curr_arts}")
st.sidebar.text(f"Events identified: {curr_clusters}")

# ==========================================
# PAGE 1: HOME DASHBOARD
# ==========================================
if page == "🏠 Home Dashboard":
    st.markdown("<div class='main-title'>NewsLens Intelligence Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Automated Multi-Document Event Discovery, Topic Modeling, and Salience Summarization</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='metric-card'><div class='metric-val'>{curr_arts}</div><div class='metric-label'>Articles Loaded</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><div class='metric-val'>{curr_clusters}</div><div class='metric-label'>Discovered Events</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='metric-card'><div class='metric-val'>3</div><div class='metric-label'>Summarization Modes</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='metric-card'><div class='metric-val'>ROUGE</div><div class='metric-label'>Evaluation Suite</div></div>", unsafe_allow_html=True)

    st.markdown("### 🚀 Processing Pipeline Architecture")
    st.info("""
    1. **Data Ingestion**: Ingests articles with title and text, validating schema and filtering duplicates.
    2. **Preprocessing**: Normalizes text, extracts sentences, tokens, and numerical/temporal entities.
    3. **Topic Modeling**: Discovers latent themes via Non-Negative Matrix Factorization (NMF).
    4. **Event Discovery**: Groups multi-source articles using TF-IDF cosine similarity & Agglomerative Clustering.
    5. **Summarization**: Generates Extractive (MMR), Abstractive (Neural/Fallback), and Hybrid traceable summaries.
    6. **Evaluation**: Benchmarks output using ROUGE metrics, clustering validity, and runtime latency.
    """)

    st.markdown("### 📋 Quick Demonstration Dataset")
    if curr_arts > 0:
        st.write("Currently demonstrating with preloaded multi-source news articles across 4 distinct real-world events:")
        st.dataframe(st.session_state["df_clean"][["article_id", "title", "source", "published_date"]], use_container_width=True)
    else:
        st.warning("No dataset loaded yet. Head over to 'Upload Dataset' to load news articles.")

# ==========================================
# PAGE 2: UPLOAD DATASET
# ==========================================
elif page == "📤 Upload Dataset":
    st.markdown("<div class='main-title'>Dataset Ingestion & Management</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Upload your own news articles or reload the built-in benchmark dataset</div>", unsafe_allow_html=True)

    tab_upload, tab_sample = st.tabs(["📁 Upload File (CSV / JSON)", "📦 Built-in Benchmark Data"])

    with tab_upload:
        uploaded_file = st.file_uploader("Upload CSV or JSON file", type=["csv", "json"])
        if uploaded_file is not None:
            ftype = "csv" if uploaded_file.name.endswith(".csv") else "json"
            try:
                raw_df = load_dataset(uploaded_file, file_type=ftype)
                is_valid, errors = validate_dataset(raw_df)

                if not is_valid:
                    st.error("Validation Error:\n" + "\n".join(errors))
                else:
                    st.success(f"Successfully loaded {len(raw_df)} rows!")
                    st.subheader("Raw Data Preview")
                    st.dataframe(raw_df.head(5), use_container_width=True)

                    if st.button("🧹 Clean and Process Uploaded Dataset"):
                        cleaned_df, clean_stats = clean_dataset(raw_df)
                        st.session_state["df_clean"] = cleaned_df
                        with st.spinner("Executing NLP Pipeline..."):
                            _run_nlp_pipeline(cleaned_df)
                        st.success(f"Processing Complete! Cleaned articles: {clean_stats['final_rows']} (Filtered {clean_stats['missing_filtered']} empty, {clean_stats['duplicates_removed']} duplicates)")
            except Exception as e:
                st.error(f"Error parsing file: {e}")

    with tab_sample:
        st.write("Reload the verified 12-article multi-event news dataset:")
        if st.button("🔄 Reset to Default Sample Dataset"):
            if config.SAMPLE_DATA_PATH.exists():
                raw = load_dataset(config.SAMPLE_DATA_PATH, file_type="csv")
                cleaned, _ = clean_dataset(raw)
                st.session_state["df_clean"] = cleaned
                with st.spinner("Processing benchmark data..."):
                    _run_nlp_pipeline(cleaned)
                st.success("Loaded default benchmark dataset successfully!")

    if curr_arts > 0:
        st.markdown("---")
        st.markdown("### 📊 Current Dataset Summary")
        summary = get_dataset_summary(st.session_state["df_clean"])
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Total Articles", summary["total_articles"])
        sc2.metric("Avg Words / Article", summary["avg_word_count"])
        sc3.metric("Unique Sources", summary["unique_sources"])
        sc4.metric("Has Event Ground Truth", "Yes" if summary["has_event_labels"] else "No")

# ==========================================
# PAGE 3: PREPROCESSING PIPELINE
# ==========================================
elif page == "🧹 Preprocessing Pipeline":
    st.markdown("<div class='main-title'>Text Preprocessing & Traceability</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Inspection of text normalization, sentence segmentation, tokenization, and entity extraction</div>", unsafe_allow_html=True)

    if curr_arts == 0:
        st.warning("Please load a dataset first.")
    else:
        df_proc = st.session_state["df_processed"]

        selected_art_idx = st.selectbox(
            "Select an article to inspect preprocessing stages:",
            options=range(len(df_proc)),
            format_func=lambda i: f"[{df_proc.iloc[i]['article_id']}] {df_proc.iloc[i]['title']}"
        )

        row = df_proc.iloc[selected_art_idx]

        col_orig, col_clean = st.columns(2)
        with col_orig:
            st.markdown("#### 📄 Original Article Text")
            st.info(row["text"])
            st.caption(f"Word count: {len(row['text'].split())} | Source: {row['source']}")

        with col_clean:
            st.markdown("#### ✨ Normalized Clean Text")
            st.success(row["clean_text"])
            st.caption("HTML tags, formatting anomalies, and URLs stripped; casing and punctuation preserved for syntax.")

        st.markdown("---")
        t1, t2, t3 = st.tabs(["📝 Segmented Sentences", "🔤 Clean Tokens (No Stopwords)", "🏷️ Extracted Entities & Details"])

        with t1:
            st.write(f"Total Sentences Extracted: **{len(row['sentences'])}**")
            for idx, sent in enumerate(row["sentences"]):
                st.text(f"[{idx + 1}] {sent}")

        with t2:
            st.write(f"Tokens Count: **{len(row['tokens'])}**")
            st.write(row["tokens"])

        with t3:
            st.write("Rule-based key temporal, numeric, and named entities detected:")
            if row["entities"]:
                ent_df = pd.DataFrame(row["entities"])
                st.table(ent_df)
            else:
                st.write("No distinct numeric or named entities identified.")

# ==========================================
# PAGE 4: TOPIC MODELING
# ==========================================
elif page == "🧠 Topic Modeling":
    st.markdown("<div class='main-title'>Unsupervised Topic Modeling</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Discovered topics via Non-Negative Matrix Factorization (NMF) with TF-IDF</div>", unsafe_allow_html=True)

    if curr_arts == 0 or "topic_modeler" not in st.session_state:
        st.warning("Please load and process a dataset first.")
    else:
        modeler = st.session_state["topic_modeler"]
        keywords_dict = modeler.get_topic_keywords(top_n=8)

        st.markdown("### 🔑 Discovered Topic Keywords")
        cols = st.columns(len(keywords_dict))
        for idx, (t_id, kw_list) in enumerate(keywords_dict.items()):
            with cols[idx % len(cols)]:
                st.markdown(f"**Topic {t_id + 1}**")
                for kw in kw_list:
                    st.markdown(f"- `{kw}`")

        st.markdown("---")
        st.markdown("### 📊 Topic Distribution Across Articles")
        df_proc = st.session_state["df_processed"]

        topic_counts = df_proc["topic_label"].value_counts()
        fig, ax = plt.subplots(figsize=(8, 3))
        topic_counts.plot(kind="barh", ax=ax, color="#3b82f6")
        ax.set_xlabel("Number of Articles")
        ax.set_ylabel("Topic")
        ax.set_title("Articles per Discovered Topic")
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("### 📰 Article Topic Assignments")
        st.dataframe(df_proc[["article_id", "title", "topic_label", "topic_confidence"]], use_container_width=True)

# ==========================================
# PAGE 5: EVENT DISCOVERY
# ==========================================
elif page == "🔍 Event Discovery":
    st.markdown("<div class='main-title'>Candidate Event Discovery & Clustering</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Grouping multi-source news articles into candidate events using TF-IDF and Hierarchical Clustering</div>", unsafe_allow_html=True)

    clusters = st.session_state.get("clusters", [])
    if not clusters:
        st.warning("No clusters available. Please load and process a dataset.")
    else:
        st.write(f"Identified **{len(clusters)}** distinct candidate news events:")

        for c in clusters:
            with st.expander(f"📌 {c['label']} ({c['article_count']} articles)", expanded=True):
                st.markdown(f"**Representative Headline:** {c['representative_title']}")
                c_indices = c["article_indices"]
                sub_df = st.session_state["df_clean"].iloc[c_indices]
                st.dataframe(sub_df[["article_id", "title", "source", "published_date"]], use_container_width=True)

# ==========================================
# PAGE 6: SUMMARIZATION STUDIO
# ==========================================
elif page == "📝 Summarization Studio":
    st.markdown("<div class='main-title'>Multi-Document Summarization Studio</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Generate Extractive, Abstractive, and Hybrid summaries for any candidate event</div>", unsafe_allow_html=True)

    clusters = st.session_state.get("clusters", [])
    if not clusters:
        st.warning("No candidate events available. Please load and process a dataset.")
    else:
        selected_event_idx = st.selectbox(
            "Select an Event Group to summarize:",
            options=range(len(clusters)),
            format_func=lambda i: f"Event {clusters[i]['event_id']}: {clusters[i]['representative_title']} ({clusters[i]['article_count']} articles)"
        )

        curr_event = clusters[selected_event_idx]
        event_articles = st.session_state["df_clean"].iloc[curr_event["article_indices"]].to_dict("records")

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            num_sents = st.slider("Target Extractive Sentence Count:", min_value=1, max_value=6, value=3)
        with col_opt2:
            method_choice = st.radio("Summarization Strategy:", ["Hybrid (Recommended)", "Extractive Only (MMR)", "Abstractive Only (Neural/Fallback)"], horizontal=True)

        if st.button("✨ Generate Multi-Document Summary", type="primary"):
            with st.spinner("Generating summary..."):
                ext_sum = ExtractiveSummarizer()
                abs_sum = AbstractiveSummarizer()
                hybrid_sum = HybridSummarizer(ext_sum, abs_sum)

                if "Hybrid" in method_choice:
                    res = hybrid_sum.summarize_event(event_articles, num_extracted_sentences=num_sents)
                    final_text = res["hybrid_summary"]
                    st.session_state["last_summary"] = final_text
                    st.session_state["last_method"] = "Hybrid Summarization"
                    st.session_state["last_res"] = res

                    st.markdown("### 📋 Generated Hybrid Summary")
                    st.success(final_text)

                    st.markdown("#### 🔗 Supporting Passages & Source Provenance")
                    st.caption("Each synthesized statement is grounded in the following extracted source sentences:")
                    for p in res["supporting_passages"]:
                        st.markdown(f"""
                        <div class='provenance-box'>
                            <strong>[{p['source']}]</strong> {p['passage']}<br>
                            <small><em>Source Article: {p['title']} ({p['article_id']})</em></small>
                        </div>
                        """, unsafe_allow_html=True)

                elif "Extractive" in method_choice:
                    res = ext_sum.summarize_articles(event_articles, num_sentences=num_sents)
                    final_text = res["summary_text"]
                    st.session_state["last_summary"] = final_text
                    st.session_state["last_method"] = "Extractive Summarization (MMR)"
                    st.session_state["last_res"] = res

                    st.markdown("### 📋 Generated Extractive Summary")
                    st.success(final_text)

                    st.markdown("#### 🔍 Selected Sentences & Importance Attribution")
                    for s in res["sentences"]:
                        st.markdown(f"- **[{s['source']}]** {s['text']} *(Score: {s.get('importance_score', 'N/A')})*")

                else:  # Abstractive
                    combined_text = " ".join(a["text"] for a in event_articles)
                    res = abs_sum.summarize(combined_text)
                    final_text = res["summary_text"]
                    st.session_state["last_summary"] = final_text
                    st.session_state["last_method"] = "Abstractive Summarization"
                    st.session_state["last_res"] = res

                    st.markdown("### 📋 Generated Abstractive Summary")
                    st.success(final_text)
                    st.info(f"Model: {res.get('model')} | {res.get('note')}")

                # Download Options
                st.download_button(
                    "💾 Download Summary as Text",
                    data=final_text,
                    file_name=f"summary_event_{curr_event['event_id']}.txt",
                    mime="text/plain"
                )

# ==========================================
# PAGE 7: EVALUATION & METRICS
# ==========================================
elif page == "📊 Evaluation & Metrics":
    st.markdown("<div class='main-title'>Evaluation & Benchmarking</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Quantitative assessment using ROUGE metrics, clustering validity, and runtime throughput</div>", unsafe_allow_html=True)

    tab_rouge, tab_cluster, tab_perf = st.tabs(["📏 Summary Evaluation (ROUGE)", "🎯 Clustering Evaluation (ARI/NMI)", "⏱️ Runtime Performance"])

    with tab_rouge:
        st.markdown("### ROUGE Metric Calculator")
        st.caption("Evaluates n-gram overlap between generated summary and human ground-truth reference summary.")

        clusters = st.session_state.get("clusters", [])
        if clusters:
            sel_ev_idx = st.selectbox(
                "Select Event to evaluate against Reference Summary:",
                options=range(len(clusters)),
                format_func=lambda i: f"Event {clusters[i]['event_id']}: {clusters[i]['representative_title']}",
                key="eval_event_select"
            )
            ev_data = st.session_state["df_clean"].iloc[clusters[sel_ev_idx]["article_indices"]]

            # Check if reference_summary exists in dataset
            has_ref = "reference_summary" in ev_data.columns and ev_data["reference_summary"].dropna().any()
            default_ref = str(ev_data["reference_summary"].dropna().iloc[0]) if has_ref else ""

            ref_input = st.text_area("Human Reference Summary (Ground Truth):", value=default_ref, height=100)
            gen_input = st.text_area("System Generated Summary to Evaluate:", value=st.session_state.get("last_summary", ""), height=100)

            if st.button("Calculate ROUGE Scores"):
                if not gen_input.strip():
                    st.error("Please provide or generate a system summary.")
                elif not ref_input.strip():
                    st.warning("Reference summary is missing! ROUGE requires human ground truth. Evaluation cannot be fabricated.")
                else:
                    eval_out = evaluate_summary(gen_input, ref_input)
                    metrics = eval_out.get("metrics", {})

                    m_rows = []
                    for k, v in metrics.items():
                        m_rows.append({
                            "Metric": k,
                            "Precision": v.get("precision", 0.0),
                            "Recall": v.get("recall", 0.0),
                            "F1-Score": v.get("f1", 0.0)
                        })
                    st.table(pd.DataFrame(m_rows))
                    st.info(eval_out.get("limitations_note", ""))

    with tab_cluster:
        st.markdown("### Clustering Quality Assessment")
        df_proc = st.session_state.get("df_processed", pd.DataFrame())
        if not df_proc.empty and "event_id" in df_proc.columns and df_proc["event_id"].notna().any():
            true_labels = df_proc["event_id"].tolist()
            pred_labels = df_proc["cluster_id"].tolist()
            c_eval = evaluate_clustering(true_labels, pred_labels)

            st.success("Ground truth event labels found in dataset! External cluster validation results:")
            st.json(c_eval)
        else:
            st.info("Ground-truth `event_id` labels are not present in the current dataset. External ARI/NMI metrics are only computed when reference labels exist.")

    with tab_perf:
        st.markdown("### Pipeline Throughput & Latency Profile")
        bench = st.session_state.get("benchmark", {})
        if bench:
            b1, b2, b3 = st.columns(3)
            b1.metric("Total Articles", bench.get("total_articles", 0))
            b2.metric("Total Runtime", f"{bench.get('total_runtime_seconds', 0)}s")
            b3.metric("Throughput", f"{bench.get('throughput_articles_per_second', 0)} art/sec")

            st.markdown("#### Execution Breakdown by Stage")
            st.table(pd.DataFrame(list(bench.get("stage_breakdown", {}).items()), columns=["Pipeline Stage", "Time (Seconds)"]))
            st.caption(f"Environment: {bench.get('environment', {})}")

# ==========================================
# PAGE 8: ABOUT AND METHODOLOGY
# ==========================================
elif page == "ℹ️ About & Methodology":
    st.markdown("<div class='main-title'>About NewsLens & Methodology</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Technical documentation, mathematical formulation, and academic submission details</div>", unsafe_allow_html=True)

    st.markdown("""
    ### 1. Problem Statement
    In modern digital journalism, identical events are reported by multiple agencies. Articles contain overlapping facts,
    varying perspectives, and contradictory details. NewsLens automates the discovery of related stories and synthesizes
    non-redundant multi-document summaries.

    ### 2. Methodological Architecture
    * **Text Representation**: Term Frequency-Inverse Document Frequency (TF-IDF) captures cross-article term salience.
    * **Topic Modeling**: Non-Negative Matrix Factorization (NMF) decomposes the term-document matrix into latent thematic components.
    * **Event Clustering**: Agglomerative Hierarchical Clustering with average linkage groups articles under a precomputed cosine distance metric.
    * **Extractive Summarization**: Maximal Marginal Relevance (MMR) balances centroid relevance with cross-sentence diversity penalty.
    * **Abstractive & Hybrid Synthesis**: Neural abstractive compression is applied to extracted factual cores, with full sentence-level source provenance.
    * **Evaluation**: ROUGE-1/2/L lexical overlap, Adjusted Rand Index (ARI), and runtime profiling.

    ### 3. Ethical Considerations & Limitations
    * Unsupervised clustering creates candidate event groups that should be reviewed.
    * Neural abstractive generation may produce unverified fluency alterations; NewsLens grounds each claim against source articles to combat hallucination.
    * Evaluation metrics like ROUGE measure lexical similarity, not factual correctness.
    """)
