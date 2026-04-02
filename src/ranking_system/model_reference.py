from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .evaluation import reranked_run
from .features import FEATURE_COLUMNS
from .modeling import (
    DEFAULT_LOGISTIC_PARAMS,
    LOGISTIC_PARAM_GRID,
    TrainedLogisticModel,
    logistic_pipeline,
    sanitize_feature_frame,
    score_rows as score_trained_rows,
    train_logistic_regression,
)
from .pipeline import normalized
from .retrieval import BM25Index


"""
Reference file for ranking-model definitions used in this project.

Use this file to inspect:
- the current logistic-regression defaults
- the current candidate hyperparameter grid
- BM25 and hybrid scoring helpers used around the learned reranker

The actual training implementation lives in `modeling.py`.
This file stays intentionally lightweight so it does not drift from the live pipeline.
"""


def bm25_baseline_run(
    bm25_index: BM25Index,
    queries: dict[str, str],
    query_ids: list[str],
    top_k: int,
) -> dict[str, dict[str, float]]:
    return {
        qid: {doc_id: score for doc_id, score in bm25_index.top_k(queries[qid], top_k)}
        for qid in query_ids
    }


def logistic_ltr_model() -> Pipeline:
    """Return the current default logistic-regression reranker pipeline."""
    return logistic_pipeline(DEFAULT_LOGISTIC_PARAMS)


def logistic_hyperparameter_grid() -> list[dict[str, object]]:
    """Return the current query-level CV search grid."""
    return [dict(params) for params in LOGISTIC_PARAM_GRID]


def train_logistic_ltr(features_df: pd.DataFrame, config) -> TrainedLogisticModel:
    """Train the live reranker implementation used by the pipeline."""
    return train_logistic_regression(features_df, config)


def score_logistic_ltr(model: TrainedLogisticModel, features_df: pd.DataFrame) -> pd.DataFrame:
    """Score rows with the trained reranker, including sanitization and selected features."""
    return score_trained_rows(model, features_df)


def current_selected_features(model: TrainedLogisticModel) -> list[str]:
    """Return the feature subset used by the final trained model."""
    return list(model.feature_columns)


def current_feature_space() -> list[str]:
    """Return the full engineered feature list before selection."""
    return list(FEATURE_COLUMNS)


def sanitize_reference_frame(features_df: pd.DataFrame) -> pd.DataFrame:
    """Expose the same non-finite cleanup used by the live training code."""
    return sanitize_feature_frame(features_df)


def blend_bm25_ltr(
    scored_df: pd.DataFrame,
    alpha: float,
    bm25_column: str = "bm25",
    ltr_column: str = "score",
) -> pd.DataFrame:
    hybrid_df = scored_df.copy()
    hybrid_df["hybrid_score"] = 0.0

    for _, group in hybrid_df.groupby("query_id", sort=False):
        bm25_scores = normalized(group[bm25_column])
        ltr_scores = normalized(group[ltr_column])
        hybrid_scores = alpha * bm25_scores + (1.0 - alpha) * ltr_scores
        hybrid_df.loc[group.index, "hybrid_score"] = hybrid_scores.to_numpy()

    return hybrid_df


def build_hybrid_run(
    scored_df: pd.DataFrame,
    query_ids: list[str],
    rerank_eval_k: int,
    alpha: float,
) -> dict[str, dict[str, float]]:
    hybrid_df = blend_bm25_ltr(scored_df, alpha=alpha)
    return reranked_run(hybrid_df, "hybrid_score", query_ids, rerank_eval_k)
