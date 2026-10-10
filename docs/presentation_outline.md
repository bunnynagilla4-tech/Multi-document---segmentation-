# NewsLens College Presentation & Project Defense Outline

This document provides a slide-by-slide structure, talking points, and demo guidelines for presenting the **NewsLens** project.

---

## 📽️ Slide-by-Slide Outline (10-12 Slides)

### Slide 1: Title & Introduction
* **Project Title:** Multi-Document Summarization for News Events Using Natural Language Processing
* **Application Name:** NewsLens
* **Domain:** Natural Language Processing (NLP) / Information Retrieval
* **Candidate Info:** [Student Name], [Roll Number], Department of Computer Science & Engineering
* **Guide:** [Supervisor / Professor Name]

### Slide 2: Motivation & Background
* Rapid proliferation of digital media results in dozens of simultaneous reports on the same story.
* **Problems Faced by Readers:**
  * Massive redundancy across news providers.
  * Information fragmentation across varying updates.
  * Risk of unverified misinformation and cognitive fatigue.
* **Core Need:** An automated system to aggregate, cluster, and distill multi-source articles into concise, verifiable summaries.

### Slide 3: Problem Statement & Objectives
* **Formal Problem:** Given $N$ unstructured news documents from heterogeneous sources, group related documents into candidate event clusters $E_k$ and synthesize an objective, non-redundant summary $S_k$.
* **Key Objectives:**
  1. Multi-format ingestion with schema validation and deduplication.
  2. Noise-tolerant text preprocessing and entity recognition.
  3. Latent topic discovery using Non-Negative Matrix Factorization (NMF).
  4. Unsupervised event clustering using TF-IDF and Agglomerative Clustering.
  5. Extractive, Abstractive, and Hybrid Summarization with source sentence provenance.
  6. Comprehensive empirical evaluation (ROUGE-1/2/L, ARI, latency).

### Slide 4: System Architecture
* Visual diagram of the 5 architectural subsystems:
  * Ingestion ➔ Preprocessing ➔ Event Discovery ➔ Summarization ➔ Evaluation & Web UI.
* Separation of core NLP library logic (`src/`) from the Streamlit web interface (`app.py`).

### Slide 5: Text Preprocessing & Topic Modeling
* **Preprocessing:** Sentence boundary segmentation, tokenization, stopword removal, and rule-based entity detection (dates, numbers, entities).
* **Topic Modeling:** NMF on TF-IDF term-document matrices.
  * Discovers latent semantic components without supervision.
  * Generates interpretable keywords for thematic inspection.

### Slide 6: Semantic Similarity & Event Clustering
* **Representation:** Normalized TF-IDF vector space.
* **Similarity Metric:** Pairwise cosine similarity and cosine distance matrices.
* **Clustering Algorithm:** Agglomerative Hierarchical Clustering with average linkage.
* **Key Benefit:** Dynamic distance threshold ($\tau = 0.75$) groups articles naturally without having to specify cluster counts in advance.

### Slide 7: Summarization Paradigms
* **Extractive Summarization:**
  * Centroid similarity combined with Maximal Marginal Relevance (MMR).
  * Balances salience against cross-article redundancy ($\lambda = 0.7$).
* **Abstractive Summarization:**
  * Pretrained neural transformers (*e.g., DistilBART*) with automatic CPU fallback.
* **Hybrid Summarization (The NewsLens Innovation):**
  * Two-stage process: Extractive factual distillation followed by cohesive abstractive condensation.
  * Full source sentence provenance tracking to prevent hallucinations.

### Slide 8: Evaluation & Experimental Results
* **Benchmark Dataset:** 12 articles across 4 real-world news events.
* **Clustering Performance:** Adjusted Rand Index (ARI) = 1.0000, NMI = 1.0000.
* **Summarization Metrics:**
  * ROUGE-1 F1: ~0.65
  * ROUGE-2 F1: ~0.42
  * ROUGE-L F1: ~0.62
* **Efficiency:** Compression ratio of 0.28 (72% text reduction); execution latency < 0.15s on CPU.

### Slide 9: Live Application Demonstration
* Demonstration of the 8-page Streamlit web app:
  * Ingestion & Data Management
  * Preprocessing Inspection (Sentence & Token segmentation)
  * Topic Discovery & Keyword Visualization
  * Candidate Event Clusters
  * Summarization Studio with Source Provenance
  * ROUGE Metric Calculator

### Slide 10: Limitations & Future Scope
* **Limitations:** Unsupervised clustering relies on lexical-semantic overlap; references are needed for ROUGE evaluation.
* **Future Scope:** Real-time RSS web scraping feeds, multi-lingual summarization, knowledge graph verification.

### Slide 11: Conclusion
* NewsLens delivers a complete, lightweight, CPU-efficient NLP system solving multi-source news information overload.
* Combines unsupervised clustering with verifiable, provenance-grounded summarization.

---

## 🎤 Demo Script for Viva / Project Defense

1. **Step 1:** Launch the Streamlit application: `streamlit run app.py`.
2. **Step 2:** Show the **Home Dashboard** with runtime metrics and the 6-stage pipeline architecture.
3. **Step 3:** Navigate to **Event Discovery** to demonstrate that the 12 articles were clustered into 4 distinct events with 100% accuracy.
4. **Step 4:** Open the **Summarization Studio**, select Event 1 (Mumbai Flash Floods), choose *Hybrid Summarization*, and click *Generate Multi-Document Summary*.
5. **Step 5:** Point out the **Supporting Passages & Source Provenance** section, explaining how NewsLens guarantees factual auditability.
6. **Step 6:** Navigate to **Evaluation & Metrics** to display the live ROUGE score computation and throughput performance profile.
