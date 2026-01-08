## Search Ranking System using BM25 and Learning-to-Rank

### Overview

This project implements an end-to-end **search ranking system** that retrieves and ranks documents for user queries.
It begins with a traditional Information Retrieval baseline (BM25) and improves ranking quality using a machine learning based **Learning-to-Rank (LTR)** model.

The project focuses on **evaluation, failure analysis, and evidence-based model selection**, similar to real-world search systems.

---

## Problem Statement

The goal is to build a ranking pipeline that:

* Retrieves relevant documents for a given query
* Orders them effectively
* Improves over keyword-based retrieval using machine learning

---

## Dataset

* **Dataset:** MS MARCO (BEIR framework, dev split)
* **Queries:** ~7,000
* **Documents:** Filtered subset based on relevance judgments
* **Labels:** Binary relevance (relevant / non-relevant)

The dataset contains real search queries and document passages.

---

## Project Pipeline

The system is built in the following stages:

1. BM25 baseline retrieval
2. Feature engineering for query–document pairs
3. Learning-to-Rank (LTR) model training
4. Evaluation using ranking metrics
5. Failure analysis and hybrid ranking experiment

---

## Phase 1 — BM25 Baseline

BM25 was implemented as the initial lexical baseline to rank documents using keyword matching.

**Results (approximate):**

* NDCG@10 ≈ 0.68
* MAP ≈ 0.64

This baseline served as a reference point for improvement.

---

## Phase 2 — Feature Engineering

A Learning-to-Rank dataset was constructed using query–document pairs.

**Features used:**

* TF-IDF cosine similarity
* BM25 score
* Token overlap count
* Overlap ratio

Query-level splits were used to avoid data leakage between training and testing.

---

## Phase 3 — Learning-to-Rank (Final Model)

A **Logistic Regression** model was trained to predict the relevance of a document given a query.

The predicted probability was used directly as the ranking score.

**Results:**

* NDCG@10 ≈ 0.83
* MAP ≈ 0.77

The LTR model significantly outperformed the BM25 baseline and was selected as the **final ranking system**.

---

## Failure Analysis

Query-wise failure analysis was performed by comparing BM25 and LTR rankings.

Key observations:

* BM25 performs better on exact-match and factoid queries
* LTR performs better on broader or semantic queries
* Some errors are caused by ambiguous queries or noisy relevance labels

This analysis helped explain model behavior beyond aggregate metrics.

---

## Hybrid Ranking Experiment

A **BM25-gated hybrid ranking** strategy was explored.

In this approach:

* Queries with strong lexical signals rely on BM25
* Other queries rely on the LTR model

However, the hybrid approach did **not outperform the LTR model**.
Since BM25 was already included as a feature in the LTR model, the hybrid did not introduce a sufficiently new signal.

Based on empirical evaluation, the **LTR model was retained as the final system**.

---

## Final Conclusion

This project demonstrates the complete lifecycle of building a search ranking system:

* Starting from a classical IR baseline
* Improving with machine learning
* Making decisions based on evaluation and failure analysis

The final model choice was driven by measured performance rather than assumptions.

---

## Technologies Used

* Python
* scikit-learn
* BM25 (rank-bm25)
* BEIR / MS MARCO
* pandas, numpy
* ranx (evaluation metrics)

---

## Key Takeaways

* Learning-to-Rank can significantly outperform lexical baselines
* Hybrid ranking is not always beneficial
* Failure analysis is essential for understanding ranking behavior
* Negative experimental results are valuable when properly analyzed

---

## How to Run

1. Run the BM25 baseline notebook
2. Generate the feature dataset
3. Train the Learning-to-Rank model
4. Evaluate results and perform failure analysis

---

## Final Note

This project emphasizes clarity, correctness, and reasoning over over-optimization.
It reflects practical machine learning and search system design decisions.

---