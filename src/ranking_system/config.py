from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    dataset: str = "msmarco"
    split: str = "dev"
    dataset_url: str | None = None
    dataset_zip_path: Path | None = None
    dataset_local_dir: Path | None = None
    sample_size: int = 75
    random_state: int = 42
    max_docs: int | None = 20000
    bm25_top_k: int = 30
    rerank_eval_k: int = 10
    positives_per_query: int = 1
    negatives_per_query: int = 4
    test_size: float = 0.2
    hybrid_threshold: float = 8.0
    tuning_folds: int = 5
    enable_model_tuning: bool = True
    enable_feature_selection: bool = True
    feature_selection_min_features: int = 5
    data_dir: Path = Path("data")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def experiments_dir(self) -> Path:
        return self.data_dir / "experiments"

    @property
    def dataset_dir(self) -> Path:
        return self.raw_dir / self.dataset

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key in ("dataset_zip_path", "dataset_local_dir", "data_dir"):
            value = payload[key]
            payload[key] = str(value) if value is not None else None
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PipelineConfig":
        normalized = dict(payload)
        for key in ("dataset_zip_path", "dataset_local_dir", "data_dir"):
            value = normalized.get(key)
            normalized[key] = Path(value) if value is not None else None
        if normalized.get("data_dir") is None:
            normalized["data_dir"] = Path("data")
        return cls(**normalized)

    def resolve_paths(self, base_dir: Path) -> "PipelineConfig":
        def resolve_path(value: Path | None) -> Path | None:
            if value is None or value.is_absolute():
                return value
            return (base_dir / value).resolve()

        return PipelineConfig(
            dataset=self.dataset,
            split=self.split,
            dataset_url=self.dataset_url,
            dataset_zip_path=resolve_path(self.dataset_zip_path),
            dataset_local_dir=resolve_path(self.dataset_local_dir),
            sample_size=self.sample_size,
            random_state=self.random_state,
            max_docs=self.max_docs,
            bm25_top_k=self.bm25_top_k,
            rerank_eval_k=self.rerank_eval_k,
            positives_per_query=self.positives_per_query,
            negatives_per_query=self.negatives_per_query,
            test_size=self.test_size,
            hybrid_threshold=self.hybrid_threshold,
            tuning_folds=self.tuning_folds,
            enable_model_tuning=self.enable_model_tuning,
            enable_feature_selection=self.enable_feature_selection,
            feature_selection_min_features=self.feature_selection_min_features,
            data_dir=resolve_path(self.data_dir) or (base_dir / "data").resolve(),
        )
