from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import PipelineConfig
from .evaluation import evaluate_run, reranked_run
from .features import FEATURE_COLUMNS


@dataclass
class TrainedLogisticModel:
    pipeline: Pipeline
    feature_columns: list[str]
    diagnostics: dict[str, object]


def save_trained_model(model: TrainedLogisticModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_trained_model(path: Path) -> TrainedLogisticModel:
    return joblib.load(path)


DEFAULT_LOGISTIC_PARAMS = {
    "C": 0.5,
    "solver": "liblinear",
    "class_weight": "balanced",
    "max_iter": 2000,
}

LOGISTIC_PARAM_GRID = [
    {"C": 0.1, "solver": "liblinear", "class_weight": "balanced", "max_iter": 2000},
    {"C": 0.5, "solver": "liblinear", "class_weight": "balanced", "max_iter": 2000},
    {"C": 1.0, "solver": "liblinear", "class_weight": "balanced", "max_iter": 2000},
    {"C": 2.0, "solver": "liblinear", "class_weight": "balanced", "max_iter": 2000},
    {"C": 1.0, "solver": "lbfgs", "class_weight": "balanced", "max_iter": 2000},
    {"C": 2.0, "solver": "lbfgs", "class_weight": "balanced", "max_iter": 2000},
]


def logistic_pipeline(params: dict[str, object]) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=int(params["max_iter"]),
                    random_state=42,
                    solver=str(params["solver"]),
                    class_weight=params["class_weight"],
                    C=float(params["C"]),
                ),
            ),
        ]
    )


def train_logistic_regression(
    features_df: pd.DataFrame,
    config: PipelineConfig,
) -> TrainedLogisticModel:
    raw_train_df = features_df[features_df["split"] == "train"].copy()
    diagnostics = build_feature_diagnostics(raw_train_df, FEATURE_COLUMNS)
    train_df = sanitize_feature_frame(raw_train_df)
    tuning_details = {
        "enabled": bool(config.enable_model_tuning),
        "selected_params": dict(DEFAULT_LOGISTIC_PARAMS),
        "best_cv_ndcg@10": None,
        "fold_count": 0,
        "candidate_results": [],
    }

    selected_params = dict(DEFAULT_LOGISTIC_PARAMS)
    if config.enable_model_tuning:
        selected_params, tuning_details = tune_logistic_regression(train_df, config)

    selected_features = list(FEATURE_COLUMNS)
    selection_details = {
        "enabled": bool(config.enable_feature_selection),
        "selected_features": list(selected_features),
        "best_cv_ndcg@10": tuning_details["best_cv_ndcg@10"],
        "history": [],
    }
    if config.enable_feature_selection:
        selected_features, selection_details = select_features(
            train_df=train_df,
            config=config,
            params=selected_params,
        )

    model = logistic_pipeline(selected_params)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*encountered in matmul")
        model.fit(train_df[selected_features], train_df["label"])
    tuning_details["selected_params"] = dict(selected_params)
    diagnostics["selected_features"] = list(selected_features)
    diagnostics["importance"] = coefficient_importance(model, selected_features)
    diagnostics["selection"] = selection_details
    diagnostics["tuning"] = tuning_details
    return TrainedLogisticModel(
        pipeline=model,
        feature_columns=list(selected_features),
        diagnostics=diagnostics,
    )


def tune_logistic_regression(
    train_df: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[dict[str, object], dict[str, object]]:
    query_ids = sorted(train_df["query_id"].unique())
    fold_count = min(max(2, int(config.tuning_folds)), len(query_ids))
    if len(query_ids) < 2:
        return dict(DEFAULT_LOGISTIC_PARAMS), {
            "enabled": True,
            "selected_params": dict(DEFAULT_LOGISTIC_PARAMS),
            "best_cv_ndcg@10": None,
            "fold_count": 0,
            "candidate_results": [],
            "note": "Not enough train queries for tuning; used defaults.",
        }

    folds = build_query_folds(query_ids, fold_count)
    candidate_results: list[dict[str, object]] = []
    best_params = dict(DEFAULT_LOGISTIC_PARAMS)
    best_score = float("-inf")

    for params in LOGISTIC_PARAM_GRID:
        fold_scores: list[float] = []
        for validation_qids in folds:
            training_qids = [qid for qid in query_ids if qid not in validation_qids]
            if not training_qids or not validation_qids:
                continue

            fit_df = train_df[train_df["query_id"].isin(training_qids)].copy()
            validation_df = train_df[train_df["query_id"].isin(validation_qids)].copy()
            if fit_df["label"].nunique() < 2 or validation_df.empty:
                continue

            model = logistic_pipeline(params)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*encountered in matmul")
                model.fit(fit_df[FEATURE_COLUMNS], fit_df["label"])
                validation_margins = model.decision_function(validation_df[FEATURE_COLUMNS])

            scored_validation = validation_df.copy()
            clipped_margins = np.clip(validation_margins, -500.0, 500.0)
            scored_validation["score"] = 1.0 / (1.0 + np.exp(-clipped_margins))
            validation_qrels = frame_to_qrels(validation_df)
            validation_run = reranked_run(scored_validation, "score", list(validation_qids), top_k=10)
            metrics = evaluate_run(validation_run, validation_qrels, list(validation_qids))
            fold_scores.append(float(metrics["ndcg@10"]))

        mean_score = float(np.mean(fold_scores)) if fold_scores else float("nan")
        candidate_results.append(
            {
                "params": dict(params),
                "mean_ndcg@10": None if math.isnan(mean_score) else mean_score,
                "fold_scores": fold_scores,
            }
        )
        if fold_scores and mean_score > best_score:
            best_score = mean_score
            best_params = dict(params)

    tuning_details = {
        "enabled": True,
        "selected_params": dict(best_params),
        "best_cv_ndcg@10": None if best_score == float("-inf") else best_score,
        "fold_count": len(folds),
        "candidate_results": candidate_results,
    }
    return best_params, tuning_details


def select_features(
    train_df: pd.DataFrame,
    config: PipelineConfig,
    params: dict[str, object],
) -> tuple[list[str], dict[str, object]]:
    selected_features = list(FEATURE_COLUMNS)
    min_features = max(1, min(int(config.feature_selection_min_features), len(selected_features)))
    best_score = cross_validated_ndcg(train_df, selected_features, params, config.tuning_folds)
    history: list[dict[str, object]] = [
        {
            "step": "baseline",
            "features": list(selected_features),
            "cv_ndcg@10": best_score,
        }
    ]

    improved = True
    while improved and len(selected_features) > min_features:
        improved = False
        best_candidate_features: list[str] | None = None
        best_candidate_score = best_score

        for feature_name in list(selected_features):
            candidate_features = [feature for feature in selected_features if feature != feature_name]
            candidate_score = cross_validated_ndcg(train_df, candidate_features, params, config.tuning_folds)
            history.append(
                {
                    "step": f"drop_{feature_name}",
                    "features": list(candidate_features),
                    "cv_ndcg@10": candidate_score,
                }
            )
            if comparable_score(candidate_score) > comparable_score(best_candidate_score) + 1e-6:
                best_candidate_score = candidate_score
                best_candidate_features = candidate_features

        if best_candidate_features is not None:
            selected_features = best_candidate_features
            best_score = best_candidate_score
            improved = True

    return selected_features, {
        "enabled": True,
        "selected_features": list(selected_features),
        "best_cv_ndcg@10": best_score,
        "history": history,
        "min_features": min_features,
    }


def build_query_folds(query_ids: list[str], fold_count: int) -> list[list[str]]:
    fold_count = min(fold_count, len(query_ids))
    return [query_ids[index::fold_count] for index in range(fold_count) if query_ids[index::fold_count]]


def frame_to_qrels(features_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    relevant_rows = features_df[features_df["label"] > 0]
    for row in relevant_rows.itertuples():
        qrels.setdefault(str(row.query_id), {})[str(row.doc_id)] = int(row.label)
    return qrels


def cross_validated_ndcg(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    params: dict[str, object],
    fold_count: int,
) -> float | None:
    query_ids = sorted(train_df["query_id"].unique())
    fold_count = min(max(2, int(fold_count)), len(query_ids))
    if len(query_ids) < 2:
        return None

    folds = build_query_folds(query_ids, fold_count)
    fold_scores: list[float] = []
    for validation_qids in folds:
        training_qids = [qid for qid in query_ids if qid not in validation_qids]
        if not training_qids or not validation_qids:
            continue

        fit_df = train_df[train_df["query_id"].isin(training_qids)].copy()
        validation_df = train_df[train_df["query_id"].isin(validation_qids)].copy()
        if fit_df["label"].nunique() < 2 or validation_df.empty:
            continue

        model = logistic_pipeline(params)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*encountered in matmul")
            model.fit(fit_df[feature_columns], fit_df["label"])
            validation_margins = model.decision_function(validation_df[feature_columns])

        scored_validation = validation_df.copy()
        clipped_margins = np.clip(validation_margins, -500.0, 500.0)
        scored_validation["score"] = 1.0 / (1.0 + np.exp(-clipped_margins))
        validation_qrels = frame_to_qrels(validation_df)
        validation_run = reranked_run(scored_validation, "score", list(validation_qids), top_k=10)
        metrics = evaluate_run(validation_run, validation_qrels, list(validation_qids))
        fold_scores.append(float(metrics["ndcg@10"]))

    if not fold_scores:
        return None
    return float(np.mean(fold_scores))


def sanitize_feature_frame(features_df: pd.DataFrame) -> pd.DataFrame:
    sanitized = features_df.copy()
    sanitized.loc[:, FEATURE_COLUMNS] = sanitized[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return sanitized


def build_feature_diagnostics(
    train_df: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, object]:
    feature_frame = sanitize_feature_frame(train_df)[feature_columns]
    correlation_matrix = feature_frame.corr(method="spearman").fillna(0.0)
    summary = []
    for feature_name in feature_columns:
        original_series = train_df[feature_name]
        series = feature_frame[feature_name]
        numeric_original = pd.to_numeric(original_series, errors="coerce")
        numeric_values = numeric_original.to_numpy(dtype=float, copy=True)
        summary.append(
            {
                "feature": feature_name,
                "missing_count": int(series.isna().sum()),
                "missing_ratio": float(series.isna().mean()),
                "inf_count": int(np.isinf(numeric_values).sum()),
                "variance": float(series.var(skipna=True)) if series.notna().any() else 0.0,
                "mean": float(series.mean(skipna=True)) if series.notna().any() else 0.0,
                "std": float(series.std(skipna=True)) if series.notna().any() else 0.0,
            }
        )

    return {
        "feature_summary": summary,
        "correlation_matrix": correlation_matrix.round(6).to_dict(),
    }


def coefficient_importance(model: Pipeline, feature_columns: list[str]) -> list[dict[str, float | str]]:
    classifier = model.named_steps["classifier"]
    coefficients = getattr(classifier, "coef_", None)
    if coefficients is None or len(coefficients) == 0:
        return []

    values = coefficients[0]
    importance = [
        {
            "feature": feature_name,
            "coefficient": float(value),
            "abs_coefficient": float(abs(value)),
        }
        for feature_name, value in zip(feature_columns, values)
    ]
    importance.sort(key=lambda row: row["abs_coefficient"], reverse=True)
    return importance


def comparable_score(value: float | None) -> float:
    return float("-inf") if value is None else float(value)


def score_rows(model: TrainedLogisticModel, features_df: pd.DataFrame) -> pd.DataFrame:
    scored = sanitize_feature_frame(features_df)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*encountered in matmul")
        margins = model.pipeline.decision_function(scored[model.feature_columns])
    clipped_margins = np.clip(margins, -500.0, 500.0)
    scored["score"] = 1.0 / (1.0 + np.exp(-clipped_margins))
    return scored
