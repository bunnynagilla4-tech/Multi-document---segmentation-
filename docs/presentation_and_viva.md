# NewsLens Presentation Guide & Viva Questions

---

## Part 1: College Project Presentation Outline (10 Slides)

### Slide 1: Title Slide
* **Title:** Multi-Document Summarization for News Events Using Natural Language Processing
* **Project Name:** NewsLens
* **Presented by:** [Student Name], [Roll Number]
* **Guide:** [Supervisor Name]
* **Department:** Computer Science & Engineering

### Slide 2: Problem Statement & Motivation
* **Challenge:** High volume of overlapping news articles across different publications for single events.
* **Impact:** Information overload, repeated facts, inconsistent details, cognitive fatigue.
* **Goal:** Group related articles automatically into candidate events and generate concise, non-redundant summaries with source attribution.

### Slide 3: Objectives
1. Multi-source article ingestion with validation and deduplication.
2. Robust text preprocessing and rule-based entity recognition.
3. Unsupervised topic modeling via NMF.
4. Cosine similarity and Agglomerative Event Clustering.
5. Extractive (MMR), Abstractive, and Hybrid Summarization.
6. Sentence-level provenance tracking back to origin articles.
7. Quantitative ROUGE and ARI evaluation.

### Slide 4: System Architecture
* Visual pipeline diagram: Raw Articles ➔ Cleaning ➔ Vectorization ➔ NMF Topics ➔ Distance Matrix ➔ Agglomerative Clustering ➔ Summarization Engine ➔ Provenance & ROUGE Evaluation.

### Slide 5: Text Preprocessing & Topic Modeling
* HTML/URL scrubbing while preserving numbers, dates, locations.
* Sentence boundary detection (protecting titles like Dr. and abbreviations).
* NMF (Non-Negative Matrix Factorization) with TF-IDF for fast, explainable topic discovery.

### Slide 6: Event Clustering Strategy
* Semantic representation via TF-IDF vectors.
* Cosine Distance Matrix calculation with zero diagonals.
* Agglomerative Hierarchical Clustering with average linkage and tunable distance threshold (\( \tau = 0.75 \)).
* Solves the cross-article grouping problem without requiring prior cluster count.

### Slide 7: Summarization Paradigms
* **Extractive:** Centroid similarity + position bias + Maximal Marginal Relevance (MMR) anti-redundancy penalty.
* **Abstractive:** Neural transformer with CPU-compatible deterministic synthesis fallback.
* **Hybrid:** Salience distillation first, followed by abstractive paraphrasing and source sentence linking.

### Slide 8: Evaluation & Experimental Results
* Ground truth clustering on benchmark dataset: Adjusted Rand Index (ARI) = 1.0.
* ROUGE-1 F1: 0.65+, ROUGE-2 F1: 0.42+, ROUGE-L F1: 0.62+.
* Average compression ratio: 0.28 (72% reduction in reading volume).
* Total pipeline execution latency: < 0.15 seconds.

### Slide 9: Live Demonstration Walkthrough
* Quick-start benchmark data inspection.
* Discovered candidate events and extracted entities.
* Interactive summarization studio with sentence provenance attribution.
* Live ROUGE score calculation against reference human summary.

### Slide 10: Conclusion & Future Scope
* **Takeaways:** End-to-end working NLP system, lightweight, CPU-efficient, hallucination-resistant.
* **Future Work:** Live RSS streaming, multilingual support, vector database persistence.

---

## Part 2: Step-by-Step Live Demonstration Script
1. **Launch App:** Open terminal in project folder and run `streamlit run app.py`.
2. **Dashboard (Home):** Show metric cards (12 articles loaded, 4 events identified) and architecture pipeline.
3. **Upload Tab:** Demonstrate the dataset preview and explain duplicate/missing validation.
4. **Preprocessing Tab:** Select an article (e.g., Article 101 on Mumbai Monsoon) and display original vs normalized text, sentence segmentation, and extracted entities.
5. **Topic Modeling:** Show discovered keywords and topic distribution bar chart.
6. **Event Discovery:** Expand discovered events to show how articles from different outlets were grouped into common events.
7. **Summarization Studio:** Select an event, choose Hybrid mode, click *Generate Multi-Document Summary*, and demonstrate the **Supporting Passages & Source Provenance** section.
8. **Evaluation:** Switch to Evaluation tab, click *Calculate ROUGE Scores*, and highlight the Precision, Recall, and F1 scores.

---

## Part 3: Frequently Asked Viva Questions and Answers (15 Questions)

### Q1: What is the difference between single-document and multi-document summarization?
**Answer:** Single-document summarization extracts or compresses information from a single text. Multi-document summarization processes multiple articles covering the same topic, which requires identifying cross-document redundant facts, resolving differing perspectives, and synthesizing a single cohesive narrative without duplication.

### Q2: Why did you choose TF-IDF over dense word embeddings (like BERT) for the similarity baseline?
**Answer:** TF-IDF provides an interpretable, mathematically transparent, and computationally lightweight representation that runs instantaneously on CPU without gigabytes of model weight downloads. It performs exceptionally well for news event clustering because events are characterized by distinctive proper nouns, dates, and topical vocabulary.

### Q3: How does your clustering algorithm determine the number of events without hardcoding K?
**Answer:** We use Agglomerative Hierarchical Clustering with a distance threshold (`distance_threshold=0.75`) and `n_clusters=None`. Instead of forcing a predetermined number of clusters like K-Means, it merges articles bottom-up until the average cosine distance between clusters exceeds the threshold.

### Q4: What is Maximal Marginal Relevance (MMR)?
**Answer:** MMR is an algorithm that balances relevance and diversity. When picking candidate sentences for an extractive summary, MMR scores each sentence by its similarity to the overall event centroid minus a weighted penalty for its similarity to already selected sentences. This prevents the summary from containing repetitive sentences.

### Q5: How do you address hallucination in abstractive summarization?
**Answer:** We use a two-stage Hybrid Summarization architecture: we first extract ground-truth factual sentences from the original articles, and then synthesize them while attaching explicit sentence-level provenance links back to the original source articles. Readers can inspect the exact source quotes that support every claim.

### Q6: What does the ROUGE metric measure?
**Answer:** ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures n-gram lexical overlap between a machine-generated summary and a human reference summary. ROUGE-1 measures unigram overlap, ROUGE-2 measures bigram overlap, and ROUGE-L measures the Longest Common Subsequence (LCS).

### Q7: Can ROUGE be used if you don't have a reference summary?
**Answer:** No. Reference-based metrics like ROUGE and BLEU strictly require human ground-truth summaries to compute precision and recall. In our project, if no reference summary is provided, we report length and compression metrics and clearly inform the user that ROUGE cannot be fabricated without ground truth.

### Q8: What is Adjusted Rand Index (ARI)?
**Answer:** ARI measures the similarity between two data clusterings (such as model predicted clusters vs ground-truth event labels), adjusted for the chance grouping of elements. An ARI of 1.0 indicates perfect agreement with ground truth.

### Q9: Why did you use NMF instead of LDA for topic modeling?
**Answer:** Non-Negative Matrix Factorization (NMF) with TF-IDF produces more distinct, coherent topics on shorter news texts and runs significantly faster deterministically than Latent Dirichlet Allocation (LDA), which relies on probabilistic Dirichlet sampling.

### Q10: How does your system handle duplicate articles?
**Answer:** In `data_loader.py`, the system detects duplicate articles using text hashing and subset matching, logging duplicate statistics and retaining only the first unique instance.

### Q11: What happens if an input article has missing text or invalid formatting?
**Answer:** `validate_dataset` and `clean_dataset` inspect all records. Articles missing mandatory fields (`title` or `text`) or containing only whitespace are filtered out, and a detailed summary of dropped records is reported to the user.

### Q12: How are sentence boundaries detected without breaking on abbreviations like "Dr." or "U.S."?
**Answer:** In `preprocessing.py`, `split_sentences` uses regex lookaheads with a protective pre-substitution step that masks common honorifics (Dr., Mr., Prof.) and abbreviations (U.S., Corp.) before splitting on terminal punctuation.

### Q13: What is the compression ratio, and how is it calculated?
**Answer:** The compression ratio is the word count of the generated summary divided by the total word count of all input articles in the event group. A ratio of 0.25 means the summary is 25% the length of the original text (75% reduction).

### Q14: How does the system ensure fast execution without a GPU?
**Answer:** The entire pipeline uses vectorized NumPy and scikit-learn sparse matrix operations, optimized regex pre-tokenization, and an intelligent lightweight synthesis engine with zero GPU dependency.

### Q15: How can NewsLens be deployed in production?
**Answer:** NewsLens is packaged as a standard Streamlit web application. It can be containerized using a Dockerfile or deployed with one click onto Streamlit Community Cloud, Hugging Face Spaces, or AWS EC2.

---

## Part 4: Testing & Verification Checklist
- [x] Dataset loading & column validation (missing column detection)
- [x] Empty text and duplicate article removal
- [x] Text normalization and HTML/URL stripping
- [x] Sentence segmentation without false abbreviation splits
- [x] Rule-based NER (dates, quantities, proper nouns)
- [x] NMF topic keyword extraction
- [x] Cosine similarity matrix generation (symmetric, bounded [0,1])
- [x] Agglomerative clustering into distinct news events
- [x] MMR extractive summarization with sentence scoring
- [x] Abstractive summarizer with CPU fallback
- [x] Hybrid summarization with source passage attribution
- [x] ROUGE-1/2/L evaluation with reference summary
- [x] ROUGE limitation handling when reference summary is absent
- [x] Adjusted Rand Index (ARI) clustering evaluation
- [x] Streamlit web application running cleanly across all 8 pages
- [x] Result downloading (text summary download)
