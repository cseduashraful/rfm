from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RelBenchDuckDBConfig:
    dataset_name: str
    project_root: Path = Path(__file__).resolve().parents[2]
    cache_root: Path | None = None
    duckdb_path: Path | None = None
    download: bool = False
    upto_test_timestamp: bool = True

    def __post_init__(self) -> None:
        if self.cache_root is None:
            self.cache_root = self.project_root / "artifacts" / "relbench"
        if self.duckdb_path is None:
            self.duckdb_path = (
                self.project_root / "artifacts" / "duckdb" / f"{self.dataset_name}.duckdb"
            )

    @property
    def dataset_cache_dir(self) -> Path:
        return Path(self.cache_root) / self.dataset_name

    @property
    def parquet_dir(self) -> Path:
        return self.dataset_cache_dir / "db"
