from __future__ import annotations

from dataclasses import dataclass
import json
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

    def _ensure_dataset_cache_dir(self) -> None:
        self.config.dataset_cache_dir.mkdir(parents=True, exist_ok=True)
        if not self.config.parquet_dir.exists():
            self.materialize()

    def _iter_parquet_paths(self) -> list[Path]:
        self._ensure_dataset_cache_dir()
        parquet_dir = self.config.parquet_dir
        if not parquet_dir.exists():
            return []
        return sorted(parquet_dir.glob("*.parquet"))

    def _load_persisted_table_info(
        self,
        conn: duckdb.DuckDBPyConnection,
    ) -> dict[str, dict[str, Any]]:
        try:
            exists = conn.execute(
                "select count(*) from information_schema.tables "
                "where table_schema = 'relbench_meta' and table_name = 'table_info'"
            ).fetchone()
            if not exists or int(exists[0]) == 0:
                return {}
            rows = conn.execute(
                "select table_name, parquet_path, num_rows, num_columns, primary_key, time_column, foreign_keys "
                "from relbench_meta.table_info"
            ).fetchall()
        except Exception:
            return {}

        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            foreign_keys = row[6]
            if isinstance(foreign_keys, str):
                try:
                    foreign_keys = json.loads(foreign_keys)
                except Exception:
                    foreign_keys = {}
            out[str(row[0])] = {
                "parquet_path": row[1],
                "num_rows": row[2],
                "num_columns": row[3],
                "primary_key": row[4],
                "time_column": row[5],
                "foreign_keys": foreign_keys if isinstance(foreign_keys, dict) else {},
            }
        return out

    def _table_info_needs_bootstrap(self, conn: duckdb.DuckDBPyConnection) -> bool:
        expected_tables = len(self._iter_parquet_paths())
        try:
            exists = conn.execute(
                "select count(*) from information_schema.tables "
                "where table_schema = 'relbench_meta' and table_name = 'table_info'"
            ).fetchone()
            if not exists or int(exists[0]) == 0:
                return True
            row_count = conn.execute("select count(*) from relbench_meta.table_info").fetchone()
            if not row_count:
                return True
            return int(row_count[0]) != expected_tables
        except Exception:
            return True

    def _build_table_info_rows(self) -> list[tuple[Any, ...]]:
        db = self.materialize()
        rows: list[tuple[Any, ...]] = []
        for name, table in db.table_dict.items():
            rows.append(
                (
                    name,
                    str(self.config.parquet_dir / f"{name}.parquet"),
                    len(table.df),
                    len(table.df.columns),
                    table.pkey_col,
                    table.time_col,
                    json.dumps(dict(table.fkey_col_to_pkey_table or {}), sort_keys=True),
                )
            )
        return rows

    def _ensure_persisted_table_info(self, conn: duckdb.DuckDBPyConnection) -> None:
        if not self._table_info_needs_bootstrap(conn):
            return
        rows = self._build_table_info_rows()
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
        if rows:
            conn.executemany(
                """
                insert into relbench_meta.table_info
                values (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def list_tables(self) -> list[TableInfo]:
        self._ensure_dataset_cache_dir()
        tables: list[TableInfo] = []
        with self.connect(read_only=False) as conn:
            metadata_by_table = self._load_persisted_table_info(conn)

        with duckdb.connect(database=":memory:") as scratch:
            for parquet_path in self._iter_parquet_paths():
                name = parquet_path.stem
                desc = scratch.execute(
                    f"describe select * from read_parquet('{parquet_path.as_posix()}')"
                ).fetchall()
                meta = metadata_by_table.get(name, {})
                row_count = meta.get("num_rows")
                if row_count is None:
                    row_count = scratch.execute(
                        f"select count(*) from read_parquet('{parquet_path.as_posix()}')"
                    ).fetchone()[0]
                tables.append(
                    TableInfo(
                        name=name,
                        parquet_path=parquet_path,
                        rows=int(row_count),
                        columns=[str(row[0]) for row in desc],
                        primary_key=meta.get("primary_key"),
                        time_column=meta.get("time_column"),
                        foreign_keys=dict(meta.get("foreign_keys", {})),
                    )
                )
        return tables

    def connect(self, read_only: bool = False) -> duckdb.DuckDBPyConnection:
        self._ensure_dataset_cache_dir()
        duckdb_path = Path(self.config.duckdb_path)
        duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(str(duckdb_path), read_only=read_only)
        self._register_views(conn)
        return conn

    def _register_views(self, conn: duckdb.DuckDBPyConnection) -> None:
        conn.execute(
            """
            create schema if not exists relbench;
            create schema if not exists relbench_meta;
            """
        )

        for parquet_path in self._iter_parquet_paths():
            table_name = parquet_path.stem
            parquet_path_str = parquet_path.as_posix()
            conn.execute(
                f"""
                create or replace view "{table_name}" as
                select * from read_parquet('{parquet_path_str}');
                """
            )
            conn.execute(
                f"""
                create or replace view relbench."{table_name}" as
                select * from read_parquet('{parquet_path_str}');
                """
            )
        self._ensure_persisted_table_info(conn)

    def query(self, sql: str) -> Any:
        with self.connect(read_only=False) as conn:
            return conn.execute(sql).fetchdf()
