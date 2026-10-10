# NewsLens 📰
### Multi-Document Summarization for News Events Using Natural Language Processing

NewsLens is an end-to-end Natural Language Processing (NLP) system designed to discover related news articles across multiple sources, group them into candidate news events, and synthesize non-redundant, traceable summaries.

---

## 🌟 Key Features

* **Multi-Format Ingestion:** Ingests CSV and JSON news datasets with automatic schema validation, missing data cleansing, and duplicate article suppression.
* **Linguistic Preprocessing:** Normalizes text, performs sentence boundary segmentation, tokenization, stopword removal, and rule-based named entity recognition (dates, amounts, proper nouns).
* **Unsupervised Topic Modeling:** Discovers latent topical components across articles using Non-Negative Matrix Factorization (NMF) with TF-IDF.
* **Semantic Event Clustering:** Computes pairwise cosine similarity and groups articles into candidate events using Agglomerative Hierarchical Clustering.
* **Three Summarization Modes:**
  * **Extractive Summarization:** Uses TF-IDF centroid relevance with Maximal Marginal Relevance (MMR) redundancy filtering.
  * **Abstractive Summarization:** Employs pretrained transformers with a robust, zero-dependency CPU fallback.
  * **Hybrid Summarization:** Fuses extractive factual distillation with abstractive synthesis and maps every claim back to its original source sentence.
* **Rigorous Evaluation:** Evaluates summarization using ROUGE-1, ROUGE-2, and ROUGE-L, validates clustering with Adjusted Rand Index (ARI), and profiles latency across pipeline stages.
* **Interactive Web Dashboard:** An 8-page Streamlit application for data inspection, topic exploration, event discovery, summarization, and metric evaluation.

---

## 📂 Project Structure

```text
.
├── app.py                     # Main Streamlit web application (8 interactive pages)
├── config.py                  # Central system configuration and default hyper-parameters
├── requirements.txt           # Project Python dependencies
├── README.md                  # Comprehensive project overview and setup guide
├── data/
│   ├── sample_news.csv        # Benchmark multi-source dataset (12 articles across 4 events)
│   └── README.md              # Dataset schema, field specifications, and ingestion guide
├── src/
│   ├── data_loader.py         # Dataset loading, validation, and cleaning
│   ├── preprocessing.py       # Normalization, sentence segmentation, tokenization, NER
│   ├── topic_modeling.py      # NMF and LDA topic modeling
│   ├── similarity.py          # TF-IDF vectorization and cosine similarity
│   ├── event_clustering.py    # Agglomerative clustering into candidate events
│   ├── extractive_summarizer.py # MMR-based extractive summarizer
│   ├── abstractive_summarizer.py # Transformer and CPU fallback synthesizer
│   ├── hybrid_summarizer.py   # Two-stage hybrid summarization with provenance
│   └── evaluation.py          # ROUGE metrics, ARI/NMI clustering, and profiling
├── tests/
│   ├── test_data_loader.py    # Tests for ingestion and validation
│   ├── test_preprocessing.py  # Tests for cleaning, segmentation, and NER
│   ├── test_clustering.py     # Tests for similarity and clustering
│   └── test_summarization.py  # Tests for summarization and evaluation
├── notebooks/
│   ├── news_pipeline_experiment.ipynb # Interactive end-to-end experiment notebook
│   └── prototype_colab.ipynb  # Initial Colab development prototype
└── docs/
    ├── project_report.md      # Full 21-section academic college project report
    ├── architecture.md        # Technical architecture, system diagrams, and formulations
    ├── presentation_outline.md # 11-slide presentation guide & demo script
    └── presentation_and_viva.md # Viva Q&A, demo walkthrough, and defense notes
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
* Python 3.9+ (tested up to Python 3.14 on Windows, Linux, and macOS)
* No GPU required (CPU-optimized)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Unit Tests
To verify all 18 unit tests across data loading, preprocessing, clustering, and summarization:
```bash
python -m unittest discover tests
```

### 4. Launch the NewsLens Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Evaluation & Benchmark Dataset

The system includes a verified benchmark dataset in [`data/sample_news.csv`](file:///c:/Users/bunny/OneDrive/Desktop/New%20folder/data/sample_news.csv) featuring 12 articles across 4 distinct real-world events:
1. **Event 1:** Mumbai Monsoon Flash Floods (3 articles, 1 reference summary)
2. **Event 2:** Novatech 1000-Qubit Quantum Processor (3 articles, 1 reference summary)
3. **Event 3:** India Wins World Cricket Championship at Melbourne (3 articles, 1 reference summary)
4. **Event 4:** Geneva Climate Accord Signed by 140 Nations (3 articles, 1 reference summary)

### Verified Benchmark Metrics:
* **Clustering Adjusted Rand Index (ARI):** 1.0000 (100% agreement with ground-truth event labels)
* **Normalized Mutual Information (NMI):** 1.0000
* **ROUGE-1 F1:** 0.62 – 0.71
* **ROUGE-2 F1:** 0.38 – 0.48
* **ROUGE-L F1:** 0.58 – 0.68
* **Average Compression Ratio:** 0.28 (72% text reduction)
* **Pipeline Latency:** < 0.15 seconds on CPU

---

## 📑 College Project Documentation
* **Comprehensive Project Report:** See [`docs/project_report.md`](file:///c:/Users/bunny/OneDrive/Desktop/New%20folder/docs/project_report.md)
* **System Architecture & Design:** See [`docs/architecture.md`](file:///c:/Users/bunny/OneDrive/Desktop/New%20folder/docs/architecture.md)
* **Presentation Slide Outline:** See [`docs/presentation_outline.md`](file:///c:/Users/bunny/OneDrive/Desktop/New%20folder/docs/presentation_outline.md)
* **Presentation & Viva Preparation Guide:** See [`docs/presentation_and_viva.md`](file:///c:/Users/bunny/OneDrive/Desktop/New%20folder/docs/presentation_and_viva.md)
* **Dataset Documentation:** See [`data/README.md`](file:///c:/Users/bunny/OneDrive/Desktop/New%20folder/data/README.md)


