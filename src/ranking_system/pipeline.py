from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from .config import PipelineConfig
from .data import ensure_directories, load_dataset, sample_queries, split_query_ids
from .evaluation import evaluate_run, reranked_run
from .features import build_feature_frame
from .modeling import (
    save_trained_model,
    score_rows,
    train_logistic_regression,
)
from .retrieval import BM25Index


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def log_stage(message: str) -> None:
    print(f"[pipeline] {message}", flush=True)


def normalized(values: pd.Series) -> pd.Series:
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value <= min_value:
        return pd.Series(0.0, index=values.index)
    return (values - min_value) / (max_value - min_value)


def blend_hybrid_scores(
    base_df: pd.DataFrame,
    alpha: float,
    bm25_column: str = "bm25",
    ltr_column: str = "score",
) -> pd.DataFrame:
    hybrid_df = base_df.copy()
    hybrid_df["hybrid_score"] = 0.0

    for _, group in hybrid_df.groupby("query_id", sort=False):
        bm25_scores = normalized(group[bm25_column])
        ltr_scores = normalized(group[ltr_column])
        hybrid_scores = alpha * bm25_scores + (1.0 - alpha) * ltr_scores
        hybrid_df.loc[group.index, "hybrid_score"] = hybrid_scores.to_numpy()

    return hybrid_df


def run_pipeline(config: PipelineConfig) -> dict[str, dict[str, float]]:
    start_time = time.perf_counter()
    ensure_directories(config)
    save_json(config.experiments_dir / "run_config.json", config.to_dict())
    log_stage(
        "starting run "
        f"(dataset={config.dataset}, split={config.split}, sample_size={config.sample_size}, "
        f"max_docs={config.max_docs}, bm25_top_k={config.bm25_top_k})"
    )

    stage_start = time.perf_counter()
    log_stage("loading dataset")
    corpus, queries, qrels = load_dataset(config)
    log_stage(
        f"dataset ready in {time.perf_counter() - stage_start:.1f}s "
        f"({len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrels queries)"
    )

    stage_start = time.perf_counter()
    log_stage("sampling queries and creating train/test split")
    sampled_queries = sample_queries(queries, qrels, config)
    train_qids, test_qids = split_query_ids(sampled_queries, config)
    save_json(
        config.experiments_dir / "query_split.json",
        {"train_qids": train_qids, "test_qids": test_qids},
    )
    log_stage(
        f"split ready in {time.perf_counter() - stage_start:.1f}s "
        f"({len(train_qids)} train, {len(test_qids)} test)"
    )

    stage_start = time.perf_counter()
    log_stage("building BM25 index")
    bm25_index = BM25Index.from_corpus(corpus)
    log_stage(f"BM25 index built in {time.perf_counter() - stage_start:.1f}s")

    stage_start = time.perf_counter()
    log_stage("evaluating BM25 baseline")
    bm25_run = {
        qid: {doc_id: score for doc_id, score in bm25_index.top_k(sampled_queries[qid], config.rerank_eval_k)}
        for qid in test_qids
    }
    bm25_metrics = evaluate_run(bm25_run, qrels, test_qids)
    save_json(config.experiments_dir / "bm25_metrics.json", bm25_metrics)
    log_stage(f"BM25 metrics saved in {time.perf_counter() - stage_start:.1f}s")

    stage_start = time.perf_counter()
    log_stage("building feature frame")
    features_df = build_feature_frame(
        corpus=corpus,
        queries=sampled_queries,
        qrels=qrels,
        train_qids=train_qids,
        test_qids=test_qids,
        bm25_index=bm25_index,
        config=config,
    )
    features_df.to_csv(config.processed_dir / "features_ltr.csv", index=False)
    log_stage(
        f"feature frame saved in {time.perf_counter() - stage_start:.1f}s "
        f"({len(features_df)} rows)"
    )

    stage_start = time.perf_counter()
    log_stage("training logistic regression reranker")
    trained_model = train_logistic_regression(features_df, config)
    save_trained_model(trained_model, config.experiments_dir / "logistic_model.joblib")
    scored_df = score_rows(trained_model, features_df)
    scored_df.to_csv(config.processed_dir / "ltr_ranked_results.csv", index=False)
    save_json(config.experiments_dir / "feature_diagnostics.json", trained_model.diagnostics)
    log_stage(f"reranker trained and scored in {time.perf_counter() - stage_start:.1f}s")

    stage_start = time.perf_counter()
    log_stage("evaluating LTR reranker")
    ltr_run = reranked_run(scored_df, "score", test_qids, config.rerank_eval_k)
    ltr_metrics = evaluate_run(ltr_run, qrels, test_qids)
    save_json(config.experiments_dir / "ltr_metrics.json", ltr_metrics)
    log_stage(f"LTR metrics saved in {time.perf_counter() - stage_start:.1f}s")

    stage_start = time.perf_counter()
    log_stage("building and evaluating hybrid run")
    hybrid_source_df = scored_df.copy()
    alpha_grid = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    alpha_results: dict[str, dict[str, float]] = {}
    best_alpha = alpha_grid[0]
    best_ndcg = float("-inf")

    for alpha in alpha_grid:
        candidate_hybrid_df = blend_hybrid_scores(hybrid_source_df, alpha)
        candidate_run = reranked_run(candidate_hybrid_df, "hybrid_score", train_qids, config.rerank_eval_k)
        candidate_metrics = evaluate_run(candidate_run, qrels, train_qids)
        alpha_results[str(alpha)] = candidate_metrics
        if candidate_metrics["ndcg@10"] > best_ndcg:
            best_ndcg = candidate_metrics["ndcg@10"]
            best_alpha = alpha

    hybrid_df = blend_hybrid_scores(hybrid_source_df, best_alpha)
    hybrid_df.to_csv(config.processed_dir / "hybrid_ranked_results.csv", index=False)

    hybrid_run = reranked_run(hybrid_df, "hybrid_score", test_qids, config.rerank_eval_k)
    hybrid_metrics = evaluate_run(hybrid_run, qrels, test_qids)
    save_json(config.experiments_dir / "hybrid_metrics.json", hybrid_metrics)
    save_json(
        config.experiments_dir / "hybrid_alpha_tuning.json",
        {"best_alpha": best_alpha, "train_metrics_by_alpha": alpha_results},
    )

    summary = {
        "bm25": bm25_metrics,
        "ltr": ltr_metrics,
        "hybrid": hybrid_metrics,
        "hybrid_alpha": best_alpha,
        "hybrid_components": ["bm25", "ltr"],
        "logistic_tuning": trained_model.diagnostics["tuning"],
        "feature_selection": trained_model.diagnostics["selection"],
        "model_artifact": str(config.experiments_dir / "logistic_model.joblib"),
        "train_queries": len(train_qids),
        "test_queries": len(test_qids),
        "feature_rows": int(len(features_df)),
    }
    save_json(config.experiments_dir / "summary.json", summary)
    log_stage(f"run completed in {time.perf_counter() - start_time:.1f}s")
    return {
        "bm25": bm25_metrics,
        "ltr": ltr_metrics,
        "hybrid": hybrid_metrics,
    }
