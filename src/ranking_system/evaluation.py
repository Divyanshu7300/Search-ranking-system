from __future__ import annotations

import os
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="urllib3 v2 only supports OpenSSL 1.1.1+.*",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
)

cache_root = Path("data/.cache")
(cache_root / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))

from ranx import Qrels, Run, evaluate

from .data import Qrels as QrelsDict


def evaluate_run(run_dict: dict[str, dict[str, float]], qrels: QrelsDict, query_ids: list[str]) -> dict[str, float]:
    filtered_qrels = {qid: qrels[qid] for qid in query_ids if qid in qrels}
    metrics = evaluate(Qrels(filtered_qrels), Run(run_dict), ["ndcg@10", "map"])
    return {metric: float(value) for metric, value in metrics.items()}


def reranked_run(scored_df: pd.DataFrame, score_column: str, query_ids: list[str], top_k: int) -> dict[str, dict[str, float]]:
    run: dict[str, dict[str, float]] = {}
    filtered = scored_df[scored_df["query_id"].isin(query_ids)]

    for qid, group in filtered.groupby("query_id"):
        ranked = group.sort_values(score_column, ascending=False).head(top_k)
        run[str(qid)] = {str(row.doc_id): float(getattr(row, score_column)) for row in ranked.itertuples()}

    return run
