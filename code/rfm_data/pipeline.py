from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
from relbench.base import Database
from relbench.datasets import dataset_registry, download_dataset

from .config import RelBenchDuckDBConfig


@dataclass(slots=True)
class TableInfo:
    name: str
    parquet_path: Path
    rows: int
    columns: list[str]
    primary_key: str | None
    time_column: str | None
    foreign_keys: dict[str, str]


class RelBenchDuckDBPipeline:
    """Materialize a RelBench dataset and expose it through DuckDB."""

    def __init__(self, config: RelBenchDuckDBConfig):
        self.config = config

    def _make_dataset(self):
        if self.config.dataset_name not in dataset_registry:
            supported = ", ".join(sorted(dataset_registry))
            raise ValueError(
                f"Unknown dataset '{self.config.dataset_name}'. Supported datasets: {supported}"
            )

        dataset_cls, args, kwargs = dataset_registry[self.config.dataset_name]
        kwargs = dict(kwargs)
        kwargs["cache_dir"] = str(self.config.dataset_cache_dir)

        if self.config.download:
            download_dataset(self.config.dataset_name)

        return dataset_cls(*args, **kwargs)

    def materialize(self) -> Database:
        self.config.dataset_cache_dir.mkdir(parents=True, exist_ok=True)
        dataset = self._make_dataset()
        return dataset.get_db(upto_test_timestamp=self.config.upto_test_timestamp)

    def list_tables(self) -> list[TableInfo]:
        db = self.materialize()
        tables: list[TableInfo] = []
        for name, table in db.table_dict.items():
            tables.append(
                TableInfo(
                    name=name,
                    parquet_path=self.config.parquet_dir / f"{name}.parquet",
                    rows=len(table.df),
                    columns=list(table.df.columns),
                    primary_key=table.pkey_col,
                    time_column=table.time_col,
                    foreign_keys=dict(table.fkey_col_to_pkey_table),
                )
            )
        return tables

    def connect(self, read_only: bool = False) -> duckdb.DuckDBPyConnection:
        duckdb_path = Path(self.config.duckdb_path)
        duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(str(duckdb_path), read_only=read_only)
        self._register_views(conn)
        return conn

    def _register_views(self, conn: duckdb.DuckDBPyConnection) -> None:
        tables = self.list_tables()
        conn.execute(
            """
            create schema if not exists relbench;
            create schema if not exists relbench_meta;
            """
        )

        for info in tables:
            parquet_path = info.parquet_path.as_posix()
            conn.execute(
                f"""
                create or replace view {info.name} as
                select * from read_parquet('{parquet_path}');
                """
            )
            conn.execute(
                f"""
                create or replace view relbench.{info.name} as
                select * from read_parquet('{parquet_path}');
                """
            )

        conn.execute("drop table if exists relbench_meta.table_info")
        conn.execute(
            """
            create table relbench_meta.table_info (
                table_name varchar,
                parquet_path varchar,
                num_rows bigint,
                num_columns integer,
                primary_key varchar,
                time_column varchar,
                foreign_keys json
            )
            """
        )
        rows = [
            (
                info.name,
                str(info.parquet_path),
                info.rows,
                len(info.columns),
                info.primary_key,
                info.time_column,
                asdict(info)["foreign_keys"],
            )
            for info in tables
        ]
        conn.executemany(
            """
            insert into relbench_meta.table_info
            values (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def query(self, sql: str) -> Any:
        with self.connect(read_only=False) as conn:
            return conn.execute(sql).fetchdf()
