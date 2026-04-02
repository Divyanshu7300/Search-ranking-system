from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message="urllib3 v2 only supports OpenSSL 1.1.1+.*",
)

from ranking_system.config import PipelineConfig
from ranking_system.pipeline import run_pipeline


PROFILE_DEFAULTS = {
    "tiny": {
        "sample_size": 200,
        "max_docs": 50000,
        "bm25_top_k": 50,
        "rerank_eval_k": 10,
        "positives_per_query": 1,
        "negatives_per_query": 6,
        "hybrid_threshold": 8.0,
        "test_size": 0.15,
    },
    "medium": {
        "sample_size": 500,
        "max_docs": 500000,
        "bm25_top_k": 100,
        "rerank_eval_k": 10,
        "positives_per_query": 2,
        "negatives_per_query": 12,
        "hybrid_threshold": 8.0,
        "test_size": 0.1,
    },
    "full": {
        "sample_size": 1000,
        "max_docs": None,
        "bm25_top_k": 200,
        "rerank_eval_k": 10,
        "positives_per_query": 3,
        "negatives_per_query": 20,
        "hybrid_threshold": 8.0,
        "test_size": 0.1,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the search ranking pipeline.")
    parser.add_argument("--profile", choices=sorted(PROFILE_DEFAULTS), default="tiny")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--max-docs", type=int)
    parser.add_argument("--bm25-top-k", type=int)
    parser.add_argument("--rerank-eval-k", type=int)
    parser.add_argument("--positives-per-query", type=int)
    parser.add_argument("--negatives-per-query", type=int)
    parser.add_argument("--hybrid-threshold", type=float)
    parser.add_argument("--test-size", type=float)
    parser.add_argument("--tuning-folds", type=int)
    parser.add_argument("--disable-model-tuning", action="store_true")
    parser.add_argument("--disable-feature-selection", action="store_true")
    parser.add_argument("--feature-selection-min-features", type=int)
    parser.add_argument("--dataset-url")
    parser.add_argument("--dataset-zip")
    parser.add_argument("--dataset-dir")
    return parser.parse_args()


def with_default(value: int | float | None, fallback: int | float) -> int | float:
    return fallback if value is None else value


def main() -> None:
    args = parse_args()
    profile_defaults = PROFILE_DEFAULTS[args.profile]
    config = PipelineConfig(
        dataset_url=args.dataset_url,
        dataset_zip_path=Path(args.dataset_zip).expanduser().resolve() if args.dataset_zip else None,
        dataset_local_dir=Path(args.dataset_dir).expanduser().resolve() if args.dataset_dir else None,
        sample_size=int(with_default(args.sample_size, profile_defaults["sample_size"])),
        max_docs=with_default(args.max_docs, profile_defaults["max_docs"]),
        bm25_top_k=int(with_default(args.bm25_top_k, profile_defaults["bm25_top_k"])),
        rerank_eval_k=int(with_default(args.rerank_eval_k, profile_defaults["rerank_eval_k"])),
        positives_per_query=int(
            with_default(args.positives_per_query, profile_defaults["positives_per_query"])
        ),
        negatives_per_query=int(
            with_default(args.negatives_per_query, profile_defaults["negatives_per_query"])
        ),
        hybrid_threshold=float(
            with_default(args.hybrid_threshold, profile_defaults["hybrid_threshold"])
        ),
        test_size=float(with_default(args.test_size, profile_defaults["test_size"])),
        tuning_folds=int(with_default(args.tuning_folds, 5)),
        enable_model_tuning=not args.disable_model_tuning,
        enable_feature_selection=not args.disable_feature_selection,
        feature_selection_min_features=int(with_default(args.feature_selection_min_features, 5)),
    )
    metrics = run_pipeline(config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
