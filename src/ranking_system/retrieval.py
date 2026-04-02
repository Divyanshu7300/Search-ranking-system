from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from .data import Corpus


def tokenize(text: str) -> list[str]:
    return text.lower().split()


@dataclass
class BM25Index:
    bm25: BM25Okapi
    doc_ids: list[str]
    docs_text: list[str]
    doc_id_to_index: dict[str, int]

    @classmethod
    def from_corpus(cls, corpus: Corpus) -> "BM25Index":
        doc_ids = list(corpus.keys())
        docs_text = [corpus[doc_id]["text"] for doc_id in doc_ids]
        tokenized_docs = [
            tokenize(text)
            for text in tqdm(docs_text, desc="BM25 tokenizing docs", unit="doc")
        ]
        return cls(
            bm25=BM25Okapi(tokenized_docs),
            doc_ids=doc_ids,
            docs_text=docs_text,
            doc_id_to_index={doc_id: idx for idx, doc_id in enumerate(doc_ids)},
        )

    def score_query(self, query: str):
        return self.bm25.get_scores(tokenize(query))

    def top_k(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = self.score_query(query)
        if k <= 0 or len(scores) == 0:
            return []

        k = min(k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        ranked_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [(self.doc_ids[idx], float(scores[idx])) for idx in ranked_indices]

    def score_candidates(self, query: str, doc_ids: Iterable[str]) -> dict[str, float]:
        scores = self.score_query(query)
        return {
            doc_id: float(scores[self.doc_id_to_index[doc_id]])
            for doc_id in doc_ids
            if doc_id in self.doc_id_to_index
        }
