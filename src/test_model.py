from __future__ import annotations

import argparse
import json
from pathlib import Path

from ranking_system.config import PipelineConfig
from ranking_system.data import load_dataset, sample_queries
from ranking_system.features import (
    build_candidate_doc_lists,
    build_inference_frame,
    fit_feature_assets,
)
from ranking_system.modeling import load_trained_model, score_rows
from ranking_system.pipeline import blend_hybrid_scores
from ranking_system.retrieval import BM25Index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the trained ranking model with a custom query.")
    parser.add_argument("--query", required=True, help="Free-text query to test.")
    parser.add_argument("--mode", choices=["bm25", "ltr", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--show-text-chars", type=int, default=220)
    parser.add_argument("--experiments-dir", default="data/experiments")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def prepare_runtime(experiments_dir: Path):
    required_files = [
        experiments_dir / "run_config.json",
        experiments_dir / "query_split.json",
        experiments_dir / "summary.json",
        experiments_dir / "logistic_model.joblib",
    ]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        joined = ", ".join(missing_files)
        raise FileNotFoundError(
            f"Missing required artifacts: {joined}. Run the training pipeline once to generate them."
        )

    project_root = experiments_dir.resolve().parent.parent
    config = PipelineConfig.from_dict(load_json(experiments_dir / "run_config.json")).resolve_paths(project_root)
    split_payload = load_json(experiments_dir / "query_split.json")
    summary_payload = load_json(experiments_dir / "summary.json")

    corpus, queries, qrels = load_dataset(config)
    sampled_queries = sample_queries(queries, qrels, config)
    train_qids = list(split_payload["train_qids"])
    test_qids = list(split_payload["test_qids"])

    bm25_index = BM25Index.from_corpus(corpus)
    _, _, train_candidate_doc_ids = build_candidate_doc_lists(
        queries=sampled_queries,
        qrels=qrels,
        train_qids=train_qids,
        test_qids=test_qids,
        bm25_index=bm25_index,
        config=config,
    )
    feature_assets = fit_feature_assets(
        corpus=corpus,
        queries=sampled_queries,
        train_qids=train_qids,
        train_candidate_doc_ids=train_candidate_doc_ids,
        bm25_index=bm25_index,
    )
    trained_model = load_trained_model(experiments_dir / "logistic_model.joblib")
    hybrid_alpha = float(summary_payload["hybrid_alpha"])
    return config, corpus, bm25_index, feature_assets, trained_model, hybrid_alpha


def render_results(rows, corpus, show_text_chars: int) -> None:
    if not rows:
        print("No results found.")
        return

    for rank, row in enumerate(rows, start=1):
        doc = corpus[row["doc_id"]]
        title = doc.get("title", "").strip()
        text = doc.get("text", "").replace("\n", " ").strip()
        preview = text[:show_text_chars]
        print(f"{rank}. doc_id={row['doc_id']} score={row['score']:.4f}")
        if title:
            print(f"   title: {title}")
        print(f"   text: {preview}")
        print()


def main() -> None:
    args = parse_args()
    experiments_dir = Path(args.experiments_dir)
    try:
        _, corpus, bm25_index, feature_assets, trained_model, hybrid_alpha = prepare_runtime(experiments_dir)
    except FileNotFoundError as exc:
        print(exc)
        return

    candidates = bm25_index.top_k(args.query, args.candidate_k)
    candidate_doc_ids = [doc_id for doc_id, _ in candidates]

    if args.mode == "bm25":
        results = [
            {"doc_id": doc_id, "score": score}
            for doc_id, score in candidates[: args.top_k]
        ]
        render_results(results, corpus, args.show_text_chars)
        return

    inference_df = build_inference_frame(
        query_id="user_query",
        query=args.query,
        candidate_doc_ids=candidate_doc_ids,
        corpus=corpus,
        bm25_index=bm25_index,
        feature_assets=feature_assets,
    )
    scored_df = score_rows(trained_model, inference_df)

    if args.mode == "hybrid":
        scored_df = blend_hybrid_scores(scored_df, hybrid_alpha)
        ranked = scored_df.sort_values("hybrid_score", ascending=False).head(args.top_k)
        results = [
            {"doc_id": row.doc_id, "score": float(row.hybrid_score)}
            for row in ranked.itertuples()
        ]
    else:
        ranked = scored_df.sort_values("score", ascending=False).head(args.top_k)
        results = [
            {"doc_id": row.doc_id, "score": float(row.score)}
            for row in ranked.itertuples()
        ]

    render_results(results, corpus, args.show_text_chars)


if __name__ == "__main__":
    main()
