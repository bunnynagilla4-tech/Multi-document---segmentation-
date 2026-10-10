# NewsLens Datasets Documentation

This directory contains benchmark and reference datasets for the **NewsLens** multi-document news summarization and event discovery system.

---

## 📁 Files in This Directory

* `sample_news.csv`: Curated multi-source benchmark dataset containing 12 real-world news articles categorized into 4 distinct news events, complete with reference ground-truth summaries and event IDs.

---

## 📋 Schema Specifications

NewsLens accepts tabular datasets in both **CSV** and **JSON** formats. The schema includes mandatory and optional fields:

| Field Name | Type | Status | Description |
| :--- | :--- | :--- | :--- |
| `title` | `string` | **Required** | The headline or title of the news article. |
| `text` | `string` | **Required** | The full textual content of the news article. |
| `article_id` | `string` | Optional | Unique identifier (e.g., `art_001`). Auto-generated if omitted. |
| `source` | `string` | Optional | News publication or agency (e.g., *Reuters*, *BBC*, *The Hindu*). Defaults to `"Unknown Source"`. |
| `published_date` | `string` | Optional | Publication timestamp or ISO date (e.g., `2026-08-14`). Defaults to `"Unknown Date"`. |
| `url` | `string` | Optional | Canonical web link to original publication for auditability. |
| `event_id` | `integer` | Optional | Ground-truth event cluster label for external clustering validation (ARI/NMI). |
| `reference_summary` | `string` | Optional | Human gold-standard reference summary used for automated ROUGE evaluation. |

---

## 🌟 Benchmark Events in `sample_news.csv`

The bundled benchmark dataset comprises 12 articles across 4 diverse news domains:

1. **Event 1: Mumbai Monsoon Flash Floods (Disaster & Civic)**
   * *Articles:* 3 (art_001, art_002, art_003)
   * *Sources:* The Daily Chronicler, National News Wire, Metropolis Dispatch
   * *Ground-Truth Summary Included:* Yes
2. **Event 2: Novatech 1000-Qubit Quantum Processor (Technology & Computing)**
   * *Articles:* 3 (art_004, art_005, art_006)
   * *Sources:* TechSphere Daily, Silicon Review, Computing Frontiers
   * *Ground-Truth Summary Included:* Yes
3. **Event 3: India Wins World Cricket Championship at Melbourne (Sports)**
   * *Articles:* 3 (art_007, art_008, art_009)
   * *Sources:* Sports Gazette, Cricket Today, World Sports Network
   * *Ground-Truth Summary Included:* Yes
4. **Event 4: Geneva Climate Accord Signed by 140 Nations (Global Politics & Environment)**
   * *Articles:* 3 (art_010, art_011, art_012)
   * *Sources:* Global Environmental Report, World Times, Diplomatic Herald
   * *Ground-Truth Summary Included:* Yes

---

## 🛠️ Adding Custom Datasets

To load your own dataset:
1. Ensure the file has at least `title` and `text` columns.
2. UTF-8 or Latin-1 encoding is supported automatically.
3. Open the NewsLens web interface, navigate to **📤 Upload Dataset**, and drop your CSV or JSON file.
4. The system will automatically validate the schema, filter empty rows, eliminate duplicate articles, and run the NLP pipeline.

---

## ⚖️ Ethics, Licensing & Attribution

* **Origin & Synthetic Representation:** The benchmark articles in `sample_news.csv` are designed for academic and developmental benchmarking, synthesizing journalistic styles without infringing on copyrighted paywalled newsfeeds.
* **Fair Use:** Used under academic Fair Use principles for Natural Language Processing research and development.
* **Limitations:** Automated clustering produces candidate event groups that reflect linguistic similarity; users should verify claims before editorial publication.
