from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RelBenchDuckDBConfig:
    dataset_name: str
    project_root: Path = Path(__file__).resolve().parents[1]
    cache_root: Path | None = None
    duckdb_path: Path | None = None
    download: bool = True
    upto_test_timestamp: bool = True
    relbench_cache_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.duckdb_path is None:
            self.duckdb_path = (
                self.project_root / "artifacts" / "duckdb" / f"{self.dataset_name}.duckdb"
            )

    @property
    def dataset_cache_dir(self) -> Path:
        if self.relbench_cache_dir is None:
            raise ValueError("RelBench cache directory is not resolved yet.")
        return Path(self.relbench_cache_dir)

    @property
    def parquet_dir(self) -> Path:
        return self.dataset_cache_dir / "db"
