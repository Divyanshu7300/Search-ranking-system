# Search Ranking System using BM25 and Learning-to-Rank

## Overview

This project builds a reproducible search ranking pipeline on the BEIR MS MARCO `dev` split.
It starts with a BM25 baseline, adds compact lexical and semantic query-document features, trains a logistic-regression reranker, and compares both standalone and hybrid ranking outputs on a held-out query split.

The code is organized under `src/` so the same experiment can be run from the CLI or from the notebooks without relying on hidden notebook state.

The current version also adds:

* resumable dataset downloads and safer dataset extraction checks
* query-level `5`-fold cross-validation for logistic-regression hyperparameter tuning
* validation-based feature selection
* explicit missing-value and non-finite-value handling through imputation
* saved model artifacts plus a simple custom-query testing CLI

## What This Version Fixes

Compared with the earlier notebook-only flow, this version makes the experiment more trustworthy:

* query-level train/test splitting happens before model training
* TF-IDF is fit without leaking held-out query text into training
* metrics are computed against the original BEIR `qrels`
* test-time reranking uses BM25-retrieved candidates only
* notebooks can run from a fresh kernel
* CLI runs now show progress bars for long stages

## What Improved From The Starting Version

Compared with the original base version, this project is now much more robust and reusable:

* the workflow moved from notebook-heavy execution to a shared CLI plus reusable `src/` pipeline
* dataset handling is safer, with resumable downloads, zip validation, and incomplete extraction recovery
* query-level train/test splitting is explicit, which reduces leakage risk
* BM25, feature engineering, reranking, evaluation, and hybrid scoring are separated into maintainable modules
* logistic-regression training now includes query-level `5`-fold cross-validation for hyperparameter tuning
* validation-based feature selection was added instead of relying only on a fixed feature list
* missing values and non-finite values are now handled explicitly before model fitting
* feature diagnostics now capture missing counts, correlations, selected features, and coefficient importance
* trained model artifacts are saved and can be reused for custom-query testing
* notebooks were reorganized so they inspect saved artifacts instead of depending on hidden execution state
* a custom-query testing flow was added through both CLI and notebook interfaces

## Project Structure

* `src/ranking_system/`
  Shared pipeline code for data loading, retrieval, features, modeling, and evaluation
* `src/run_pipeline.py`
  CLI entry point
* `notebooks/`
  Thin notebooks for inspecting baseline, features, LTR results, and hybrid results
* `data/raw/`
  Downloaded dataset files
* `data/processed/`
  Generated feature tables and ranked outputs
* `data/experiments/`
  Metrics and split metadata

## Pipeline

1. Load or download BEIR MS MARCO
2. Sample a fixed set of judged queries
3. Split sampled queries into train and test sets
4. Build a BM25 index over the chosen corpus slice
5. Evaluate BM25 on held-out queries
6. Build LTR training rows from positives plus BM25 negatives
7. Build test rows from BM25-retrieved candidates
8. Tune logistic-regression hyperparameters with query-level `5`-fold cross-validation
9. Select a compact feature subset using validation `ndcg@10`
10. Train the final logistic-regression reranker with imputation and scaling
11. Tune a weighted BM25-LTR blend on the training split
12. Evaluate BM25, LTR, and hybrid runs on held-out queries

## Features Used

The rerankers use a compact feature set that mixes lexical, semantic, and structural signals:

* TF-IDF cosine similarity
* sentence-transformer cosine similarity
* log-compressed BM25 score
* reciprocal BM25 rank
* common-term overlap and query coverage
* IDF-weighted overlap
* query length, document length, and length ratio

The raw BM25 score is also preserved in the output files for analysis and hybrid scoring.

Current selected feature subset from the latest saved run:

* `tfidf`
* `semantic_similarity`
* `bm25_feature`
* `common_terms`
* `overlap_ratio`
* `idf_overlap`
* `doc_length_feature`
* `query_length_feature`
* `length_ratio`

## Models

* `BM25`
  Lexical retrieval baseline
* `Logistic Regression`
  Lightweight pointwise reranker with median imputation, standardization, `5`-fold CV tuning, and validation-based feature selection
* `Hybrid`
  Alpha-tuned weighted blend of BM25 and Logistic Regression LTR scores

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## How To Run

Quick smoke test:

```bash
python src/run_pipeline.py --profile tiny
```

Recommended main local experiment:

```bash
venv/bin/python src/run_pipeline.py --profile medium --sample-size 500 --max-docs 500000 --test-size 0.1 --positives-per-query 2 --negatives-per-query 12
```

If you manually download the dataset from another source such as Kaggle, you can bypass the default BEIR download:

```bash
venv/bin/python src/run_pipeline.py --profile medium --dataset-zip /absolute/path/to/msmarco.zip
```

If you already have the extracted dataset folder, point directly to it:

```bash
venv/bin/python src/run_pipeline.py --profile medium --dataset-dir /absolute/path/to/msmarco
```

Faster development run:

```bash
venv/bin/python src/run_pipeline.py --profile medium --sample-size 500 --max-docs 100000 --test-size 0.1 --positives-per-query 2 --negatives-per-query 12
```

Quick end-to-end verification:

```bash
venv/bin/python src/run_pipeline.py --profile tiny
```

Note:
larger corpus slices make feature generation much slower, so `100k` is a good tuning setup and `500k` is a stronger final evaluation setup on a laptop.

## Custom Query Testing

After a successful pipeline run, you can test the saved model on your own query:

```bash
venv/bin/python src/test_model.py --query "best budget gaming laptop" --mode hybrid --top-k 5
```

Modes:

* `bm25`
  Show BM25-only ranking
* `ltr`
  Show logistic-regression reranking
* `hybrid`
  Show the tuned BM25-LTR blend

The script prints document ids, scores, titles, and text previews for the top results.

## Profiles

* `tiny`
  Fast verification run for checking that the pipeline works end-to-end
* `medium`
  Recommended main experiment profile for local runs
* `full`
  Very heavy configuration intended for larger experiments

## Reference Results

Latest saved reference run:

Command:

```bash
venv/bin/python src/run_pipeline.py --profile medium --sample-size 500 --max-docs 500000 --test-size 0.1 --positives-per-query 2 --negatives-per-query 12
```

Results:

* split: `450 train / 50 test`
* feature rows: `10872`
* logistic tuning: `5-fold CV`, best params = `C=0.5`, `solver=liblinear`, `class_weight=balanced`
* feature selection: `9` selected features, best CV `ndcg@10 = 0.9330`
* BM25: `ndcg@10 = 0.4414`, `map = 0.4123`
* Logistic Regression LTR: `ndcg@10 = 0.5656`, `map = 0.5207`
* Hybrid: `ndcg@10 = 0.5656`, `map = 0.5207`
* best hybrid alpha: `0.0`

Interpretation:

* the tuned logistic-regression reranker still clearly beats the BM25 baseline on the saved `500k` run
* BM25 remains a strong lexical baseline and candidate generator
* the saved hybrid selected `alpha = 0.0`, so it collapsed to the standalone reranker on this run
* the most defensible final model in the current codebase is still the logistic-regression LTR reranker, now with tuning, diagnostics, and feature selection

## Generated Artifacts

After each successful run, the main outputs are:

* `data/processed/features_ltr.csv`
  Query-document feature table
* `data/processed/ltr_ranked_results.csv`
  Candidate rows with reranker scores
* `data/processed/hybrid_ranked_results.csv`
  Candidate rows with hybrid scores
* `data/experiments/bm25_metrics.json`
  BM25 metrics
* `data/experiments/feature_diagnostics.json`
  Missing-value summary, correlation matrix, feature importance, tuning, and feature-selection details
* `data/experiments/ltr_metrics.json`
  LTR metrics
* `data/experiments/hybrid_metrics.json`
  Hybrid metrics
* `data/experiments/logistic_model.joblib`
  Saved trained reranker for reuse in query testing
* `data/experiments/query_split.json`
  Saved train/test split
* `data/experiments/run_config.json`
  Saved pipeline configuration used to produce the current artifacts
* `data/experiments/summary.json`
  Compact summary

## Notebooks

The notebooks now act as lightweight inspection layers:

* `notebooks/01_bm25_baseline.ipynb`
  Run the medium configuration and view BM25 baseline metrics
* `notebooks/02_feature_diagnostics.ipynb`
  Inspect generated features, missing values, correlations, and diagnostics
* `notebooks/03_ltr_tuning_and_selection.ipynb`
  Inspect reranker metrics, selected hyperparameters, and selected features
* `notebooks/04_hybrid_ranking.ipynb`
  Run the medium configuration and inspect weighted BM25-LTR hybrid results
* `notebooks/05_user_query_testing.ipynb`
  Test the saved model on a custom query and inspect readable ranked outputs

## Current Takeaways

* BM25 remains a strong baseline and candidate generator
* semantic similarity and IDF-heavy lexical signals carry a lot of weight in the saved reranker
* the current codebase now has basic ML hygiene for this scale: `5`-fold tuning, feature selection, diagnostics, and imputation
* the current best saved result is still from the logistic-regression LTR model
* the hybrid is available as a blended alternative, but in the latest saved run it collapses to the same effective ranking as LTR


## Known Limitations

* the semantic feature stage still depends on the `sentence-transformers` model and may need `huggingface.co` access on first use if the model is not cached
* the learned reranker is still `LogisticRegression`; stronger ranking models such as `LightGBM` or `LambdaMART` are not implemented yet
* the current validation setup is much better than before, but it is still offline experimentation rather than full production validation
* there is no serving/API layer, online feedback loop, or A/B testing framework yet
* the hybrid model is available, but in the latest saved run it selected `alpha = 0.0`, so it behaved the same as the standalone LTR reranker
