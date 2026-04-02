from __future__ import annotations

import random
import time
import warnings
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Tuple

import requests
from tqdm.auto import tqdm

from .config import PipelineConfig

Corpus = Dict[str, Dict[str, str]]
Queries = Dict[str, str]
Qrels = Dict[str, Dict[str, int]]

warnings.filterwarnings(
    "ignore",
    message="urllib3 v2 only supports OpenSSL 1.1.1+.*",
)


def ensure_directories(config: PipelineConfig) -> None:
    for path in (config.raw_dir, config.processed_dir, config.experiments_dir):
        path.mkdir(parents=True, exist_ok=True)


def ensure_dataset(config: PipelineConfig) -> Path:
    ensure_directories(config)
    if config.dataset_local_dir is not None:
        validate_dataset_dir(config.dataset_local_dir)
        return config.dataset_local_dir
    if is_valid_dataset_dir(config.dataset_dir):
        return config.dataset_dir

    archive_path = config.raw_dir / f"{config.dataset}.zip"
    if config.dataset_zip_path is not None:
        archive_path = config.dataset_zip_path

    archive_url = config.dataset_url or (
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/"
        f"{config.dataset}.zip"
    )

    if config.dataset_zip_path is not None:
        if not is_valid_zip(archive_path):
            raise ValueError(f"Dataset zip {archive_path} is missing or invalid.")
        extract_dataset_archive(archive_path, config.raw_dir)
        return config.dataset_dir

    if not is_valid_zip(archive_path):
        download_with_resume(archive_url, archive_path)

    if not is_valid_zip(archive_path):
        raise RuntimeError(
            f"Downloaded archive at {archive_path} is still invalid. "
            "The source may be sending incomplete data; try again later or use --dataset-zip/--dataset-dir."
        )

    extract_dataset_archive(archive_path, config.raw_dir)
    return config.dataset_dir


def is_valid_zip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(path, "r") as zip_handle:
            return zip_handle.testzip() is None
    except zipfile.BadZipFile:
        return False


def is_valid_dataset_dir(path: Path) -> bool:
    required_paths = (
        path / "corpus.jsonl",
        path / "queries.jsonl",
        path / "qrels",
    )
    return all(required_path.exists() for required_path in required_paths)


def validate_dataset_dir(path: Path) -> None:
    if not is_valid_dataset_dir(path):
        raise ValueError(
            f"Dataset directory {path} is incomplete. Expected corpus.jsonl, queries.jsonl, and qrels/."
        )


def extract_dataset_archive(archive_path: Path, destination_dir: Path) -> None:
    with zipfile.ZipFile(archive_path, "r") as zip_handle:
        zip_handle.extractall(path=destination_dir)


def download_with_resume(url: str, destination: Path, max_attempts: int = 5) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        existing_size = destination.stat().st_size if destination.exists() else 0
        headers = {"Range": f"bytes={existing_size}-"} if existing_size > 0 else {}

        try:
            with requests.get(url, stream=True, headers=headers, timeout=(10, 60)) as response:
                if response.status_code == 416 and is_valid_zip(destination):
                    return

                if response.status_code == 200 and existing_size > 0:
                    destination.unlink(missing_ok=True)
                    existing_size = 0

                response.raise_for_status()

                total_size = _expected_total_size(response, existing_size)
                mode = "ab" if response.status_code == 206 and existing_size > 0 else "wb"
                initial = existing_size if mode == "ab" else 0

                with tqdm(
                    total=total_size,
                    initial=initial,
                    unit="iB",
                    unit_scale=True,
                    desc=str(destination),
                ) as progress_bar:
                    with destination.open(mode) as file_handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            file_handle.write(chunk)
                            progress_bar.update(len(chunk))
            return
        except KeyboardInterrupt as exc:
            raise RuntimeError(
                f"Download interrupted. Partial file kept at {destination}; rerun the same command to resume."
            ) from exc
        except requests.RequestException as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            wait_seconds = min(2**attempt, 20)
            print(
                f"[dataset] download attempt {attempt}/{max_attempts} failed: {exc}. "
                f"Retrying in {wait_seconds}s..."
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Unable to download dataset from {url}. "
        "The host may be slow or blocking the connection; use --dataset-zip/--dataset-dir if needed."
    ) from last_error


def _expected_total_size(response: requests.Response, existing_size: int) -> int | None:
    content_range = response.headers.get("Content-Range")
    if content_range and "/" in content_range:
        total_part = content_range.rsplit("/", 1)[-1]
        if total_part.isdigit():
            return int(total_part)

    content_length = response.headers.get("Content-Length")
    if content_length and content_length.isdigit():
        return int(content_length) + existing_size

    return None


def load_dataset(config: PipelineConfig) -> Tuple[Corpus, Queries, Qrels]:
    from beir.datasets.data_loader import GenericDataLoader

    data_path = ensure_dataset(config)
    corpus, queries, qrels = GenericDataLoader(str(data_path)).load(split=config.split)
    corpus, qrels = maybe_limit_corpus(corpus, qrels, config)
    return corpus, queries, qrels


def maybe_limit_corpus(corpus: Corpus, qrels: Qrels, config: PipelineConfig) -> Tuple[Corpus, Qrels]:
    if config.max_docs is None or config.max_docs <= 0 or len(corpus) <= config.max_docs:
        return corpus, qrels

    judged_doc_ids = sorted({doc_id for query_qrels in qrels.values() for doc_id in query_qrels if doc_id in corpus})
    selected_doc_ids = list(judged_doc_ids[: config.max_docs])

    if len(selected_doc_ids) < config.max_docs:
        for doc_id in sorted(corpus):
            if doc_id in judged_doc_ids:
                continue
            selected_doc_ids.append(doc_id)
            if len(selected_doc_ids) >= config.max_docs:
                break

    selected_doc_ids_set = set(selected_doc_ids)
    limited_corpus = {doc_id: corpus[doc_id] for doc_id in selected_doc_ids}
    limited_qrels = {
        qid: {doc_id: label for doc_id, label in query_qrels.items() if doc_id in selected_doc_ids_set}
        for qid, query_qrels in qrels.items()
    }
    limited_qrels = {qid: query_qrels for qid, query_qrels in limited_qrels.items() if query_qrels}
    return limited_corpus, limited_qrels


def sample_queries(queries: Queries, qrels: Qrels, config: PipelineConfig) -> Queries:
    eligible_qids = sorted(qid for qid in queries if qid in qrels and qrels[qid])
    if config.sample_size >= len(eligible_qids):
        return {qid: queries[qid] for qid in eligible_qids}

    rng = random.Random(config.random_state)
    sampled_qids = rng.sample(eligible_qids, config.sample_size)
    return {qid: queries[qid] for qid in sorted(sampled_qids)}


def split_query_ids(sampled_queries: Queries, config: PipelineConfig) -> Tuple[list[str], list[str]]:
    qids = sorted(sampled_queries)
    if len(qids) < 2:
        raise ValueError("At least 2 sampled queries are required to create train/test splits.")

    rng = random.Random(config.random_state)
    rng.shuffle(qids)

    test_count = max(1, int(len(qids) * config.test_size))
    test_count = min(test_count, len(qids) - 1)
    test_qids = sorted(qids[:test_count])
    train_qids = sorted(qids[test_count:])
    return train_qids, test_qids


def subset_qrels(qrels: Qrels, query_ids: Iterable[str]) -> Qrels:
    return {qid: qrels[qid] for qid in query_ids if qid in qrels}
