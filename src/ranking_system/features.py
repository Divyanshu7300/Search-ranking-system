from __future__ import annotations

from dataclasses import dataclass
import math
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from .config import PipelineConfig
from .data import Corpus, Qrels, Queries
from .retrieval import BM25Index, tokenize


FEATURE_COLUMNS = [
    "tfidf",
    "semantic_similarity",
    "bm25_feature",
    "bm25_rank_feature",
    "common_terms",
    "overlap_ratio",
    "query_coverage",
    "idf_overlap",
    "doc_length_feature",
    "query_length_feature",
    "length_ratio",
]
SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

warnings.filterwarnings(
    "ignore",
    message="urllib3 v2 only supports OpenSSL 1.1.1+.*",
)


@dataclass
class FeatureAssets:
    vectorizer: TfidfVectorizer
    idf_lookup: dict[str, float]
    semantic_model: object | None


def compress_score(value: float) -> float:
    return math.copysign(math.log1p(abs(value)), value)


def overlap_features(
    query_tokens: list[str],
    document_tokens: list[str],
    idf_lookup: dict[str, float],
) -> dict[str, float]:
    query_terms = set(query_tokens)
    document_terms = set(document_tokens)
    common_terms = query_terms & document_terms
    idf_overlap = sum(idf_lookup.get(term, 0.0) for term in common_terms)
    return {
        "common_terms": float(len(common_terms)),
        "overlap_ratio": float(len(common_terms) / max(len(query_terms), 1)),
        "query_coverage": float(len(common_terms) / max(len(query_tokens), 1)),
        "idf_overlap": float(idf_overlap),
    }


def load_semantic_model():
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(SEMANTIC_MODEL_NAME)
    except Exception as exc:
        warnings.warn(f"SentenceTransformer failed: {exc}", RuntimeWarning)
        return None


def build_candidate_doc_lists(
    queries: Queries,
    qrels: Qrels,
    train_qids: list[str],
    test_qids: list[str],
    bm25_index: BM25Index,
    config: PipelineConfig,
) -> tuple[dict[str, list[str]], list[str], list[str]]:
    all_qids = train_qids + test_qids
    candidate_doc_ids_by_qid: dict[str, list[str]] = {}
    candidate_doc_ids: list[str] = []
    seen_doc_ids: set[str] = set()

    for qid in all_qids:
        ranked_candidates = bm25_index.top_k(queries[qid], config.bm25_top_k)
        doc_ids = [doc_id for doc_id, _ in ranked_candidates]
        if qid in train_qids:
            candidate_scores = {doc_id: score for doc_id, score in ranked_candidates}
            positive_doc_ids = sorted(
                qrels[qid],
                key=lambda doc_id: (-int(qrels[qid][doc_id]), candidate_scores.get(doc_id, float("-inf")), doc_id),
            )[: config.positives_per_query]
            for doc_id in positive_doc_ids:
                if doc_id not in doc_ids:
                    doc_ids.append(doc_id)
        candidate_doc_ids_by_qid[qid] = doc_ids
        for doc_id in doc_ids:
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                candidate_doc_ids.append(doc_id)

    train_candidate_doc_ids: list[str] = []
    seen_train_doc_ids: set[str] = set()
    for qid in train_qids:
        for doc_id in candidate_doc_ids_by_qid[qid]:
            if doc_id not in seen_train_doc_ids:
                seen_train_doc_ids.add(doc_id)
                train_candidate_doc_ids.append(doc_id)

    return candidate_doc_ids_by_qid, candidate_doc_ids, train_candidate_doc_ids


def fit_feature_assets(
    corpus: Corpus,
    queries: Queries,
    train_qids: list[str],
    train_candidate_doc_ids: list[str],
    bm25_index: BM25Index,
) -> FeatureAssets:
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    train_queries = [queries[qid] for qid in train_qids]
    train_docs_text = [corpus[doc_id]["text"] for doc_id in train_candidate_doc_ids]
    vectorizer.fit(train_queries + train_docs_text)
    idf_lookup = {
        term: float(value)
        for term, value in getattr(bm25_index.bm25, "idf", {}).items()
        if np.isfinite(value)
    }
    semantic_model = load_semantic_model()
    if semantic_model is None:
        warnings.warn("semantic_similarity will be 0.0 for this run.", RuntimeWarning)
    return FeatureAssets(
        vectorizer=vectorizer,
        idf_lookup=idf_lookup,
        semantic_model=semantic_model,
    )


def build_inference_frame(
    query_id: str,
    query: str,
    candidate_doc_ids: list[str],
    corpus: Corpus,
    bm25_index: BM25Index,
    feature_assets: FeatureAssets,
    split: str = "inference",
) -> pd.DataFrame:
    if not candidate_doc_ids:
        return pd.DataFrame(columns=["query_id", "doc_id", "split", *FEATURE_COLUMNS, "bm25", "label"])

    candidate_scores = bm25_index.score_candidates(query, candidate_doc_ids)
    candidate_ranks = {
        doc_id: rank
        for rank, doc_id in enumerate(
            sorted(candidate_doc_ids, key=lambda doc_id: candidate_scores.get(doc_id, float("-inf")), reverse=True),
            start=1,
        )
    }
    query_tokens = tokenize(query)
    query_vector = feature_assets.vectorizer.transform([query])
    candidate_docs_text = [corpus[doc_id]["text"] for doc_id in candidate_doc_ids]
    doc_vectors = feature_assets.vectorizer.transform(candidate_docs_text)
    candidate_doc_tokens = {doc_id: tokenize(corpus[doc_id]["text"]) for doc_id in candidate_doc_ids}

    if feature_assets.semantic_model is None:
        query_semantic_vector = None
        semantic_doc_vectors = None
    else:
        query_semantic_vector = feature_assets.semantic_model.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        semantic_doc_vectors = feature_assets.semantic_model.encode(
            candidate_docs_text,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    rows: list[dict[str, object]] = []
    for index, doc_id in enumerate(candidate_doc_ids):
        document_tokens = candidate_doc_tokens[doc_id]
        tfidf_score = float(cosine_similarity(query_vector, doc_vectors[index])[0][0])
        if query_semantic_vector is None or semantic_doc_vectors is None:
            semantic_score = 0.0
        else:
            semantic_score = float(np.dot(query_semantic_vector, semantic_doc_vectors[index]))
        overlap = overlap_features(query_tokens, document_tokens, feature_assets.idf_lookup)
        raw_bm25 = float(candidate_scores.get(doc_id, 0.0))
        bm25_rank = float(candidate_ranks.get(doc_id, len(candidate_doc_ids) + 1))
        query_length = float(len(query_tokens))
        doc_length = float(len(document_tokens))
        rows.append(
            {
                "query_id": query_id,
                "doc_id": doc_id,
                "split": split,
                "tfidf": tfidf_score,
                "semantic_similarity": semantic_score,
                "bm25": raw_bm25,
                "bm25_feature": compress_score(raw_bm25),
                "bm25_rank_feature": 1.0 / bm25_rank,
                "common_terms": overlap["common_terms"],
                "overlap_ratio": overlap["overlap_ratio"],
                "query_coverage": overlap["query_coverage"],
                "idf_overlap": overlap["idf_overlap"],
                "doc_length_feature": compress_score(doc_length),
                "query_length_feature": compress_score(query_length),
                "length_ratio": float(query_length / max(doc_length, 1.0)),
                "label": 0,
            }
        )
    return pd.DataFrame(rows)


def build_feature_frame(
    corpus: Corpus,
    queries: Queries,
    qrels: Qrels,
    train_qids: list[str],
    test_qids: list[str],
    bm25_index: BM25Index,
    config: PipelineConfig,
) -> pd.DataFrame:
    if not train_qids:
        raise ValueError("Training split is empty. Increase sample size or lower test_size.")

    all_qids = train_qids + test_qids
    candidate_doc_ids_by_qid, candidate_doc_ids, train_candidate_doc_ids = build_candidate_doc_lists(
        queries=queries,
        qrels=qrels,
        train_qids=train_qids,
        test_qids=test_qids,
        bm25_index=bm25_index,
        config=config,
    )
    feature_assets = fit_feature_assets(
        corpus=corpus,
        queries=queries,
        train_qids=train_qids,
        train_candidate_doc_ids=train_candidate_doc_ids,
        bm25_index=bm25_index,
    )

    candidate_docs_text = [corpus[doc_id]["text"] for doc_id in candidate_doc_ids]
    doc_vectors = feature_assets.vectorizer.transform(candidate_docs_text)
    candidate_doc_id_to_index = {doc_id: index for index, doc_id in enumerate(candidate_doc_ids)}
    candidate_doc_tokens = {doc_id: tokenize(corpus[doc_id]["text"]) for doc_id in candidate_doc_ids}
    semantic_model = feature_assets.semantic_model
    if semantic_model is None:
        semantic_doc_vectors = None
    else:
        semantic_doc_vectors = semantic_model.encode(
            candidate_docs_text,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    rows: list[dict[str, object]] = []
    if semantic_model is None or semantic_doc_vectors is None:
        query_semantic_by_qid = {qid: None for qid in all_qids}
    else:
        query_semantic_vectors = semantic_model.encode(
            [queries[qid] for qid in all_qids],
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        query_semantic_by_qid = {
            qid: query_semantic_vectors[index]
            for index, qid in enumerate(all_qids)
        }

    for qid in tqdm(all_qids, desc="Building features", unit="query"):
        query = queries[qid]
        query_tokens = tokenize(query)
        ranked_candidates = bm25_index.top_k(query, config.bm25_top_k)
        candidate_scores = {doc_id: score for doc_id, score in ranked_candidates}
        candidate_ranks = {
            doc_id: rank
            for rank, (doc_id, _) in enumerate(ranked_candidates, start=1)
        }
        candidate_doc_ids = candidate_doc_ids_by_qid[qid]

        if qid in train_qids:
            positive_doc_ids = sorted(
                qrels[qid],
                key=lambda doc_id: (-int(qrels[qid][doc_id]), candidate_scores.get(doc_id, float("-inf")), doc_id),
            )[: config.positives_per_query]

            chosen_negatives: list[str] = []
            for doc_id in candidate_doc_ids:
                if doc_id not in qrels[qid]:
                    chosen_negatives.append(doc_id)
                if len(chosen_negatives) >= config.negatives_per_query:
                    break

            selected_doc_ids = positive_doc_ids + chosen_negatives
        else:
            selected_doc_ids = candidate_doc_ids

        query_vector = feature_assets.vectorizer.transform([query])
        query_semantic_vector = query_semantic_by_qid[qid]

        for doc_id in selected_doc_ids:
            doc_index = candidate_doc_id_to_index[doc_id]
            document_tokens = candidate_doc_tokens[doc_id]
            tfidf_score = float(cosine_similarity(query_vector, doc_vectors[doc_index])[0][0])
            if query_semantic_vector is None or semantic_doc_vectors is None:
                semantic_score = 0.0
            else:
                semantic_score = float(np.dot(query_semantic_vector, semantic_doc_vectors[doc_index]))
            overlap = overlap_features(query_tokens, document_tokens, feature_assets.idf_lookup)
            raw_bm25 = float(candidate_scores.get(doc_id, 0.0))
            bm25_rank = float(candidate_ranks.get(doc_id, len(candidate_doc_ids) + 1))
            query_length = float(len(query_tokens))
            doc_length = float(len(document_tokens))
            length_ratio = float(query_length / max(doc_length, 1.0))

            rows.append(
                {
                    "query_id": qid,
                    "doc_id": doc_id,
                    "split": "train" if qid in train_qids else "test",
                    "tfidf": tfidf_score,
                    "semantic_similarity": semantic_score,
                    "bm25": raw_bm25,
                    "bm25_feature": compress_score(raw_bm25),
                    "bm25_rank_feature": 1.0 / bm25_rank,
                    "common_terms": overlap["common_terms"],
                    "overlap_ratio": overlap["overlap_ratio"],
                    "query_coverage": overlap["query_coverage"],
                    "idf_overlap": overlap["idf_overlap"],
                    "doc_length_feature": compress_score(doc_length),
                    "query_length_feature": compress_score(query_length),
                    "length_ratio": length_ratio,
                    "label": int(doc_id in qrels[qid]),
                }
            )

    return pd.DataFrame(rows)
