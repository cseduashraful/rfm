from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

# Ensure sibling project modules under RFM/code are importable when running from RFM/v2.
import sys

_RFM_ROOT = Path(__file__).resolve().parents[2]
_RFM_CODE_DIR = _RFM_ROOT / "code"
if _RFM_CODE_DIR.exists():
    sys.path.insert(0, str(_RFM_CODE_DIR))

from data_config import RelBenchDuckDBConfig
from data_pipeline import RelBenchDuckDBPipeline
from phase1_utils import (
    is_temporal_dtype_name,
    normalize_identifier_tokens,
    quote_ident,
    quote_qualified,
    singularize_identifier,
    sql_str_literal,
    sql_timestamp_literal,
)


@dataclass(slots=True)
class TableMetadata:
    table_name: str
    parquet_path: Path
    columns: list[str]
    dtypes: dict[str, str]
    provided_primary_key: str | None = None
    provided_foreign_keys: dict[str, str] = field(default_factory=dict)
    time_column: str | None = None
    detected_time_columns: list[str] = field(default_factory=list)
    metadata_source: str = "heuristic"
    metadata_warnings: list[str] = field(default_factory=list)
    raw_row_count_hint: int | None = None


class Phase1DataAccess:
    def __init__(
        self,
        *,
        dataset_name: str,
        random_seed: int = 42,
        download: bool = True,
    ) -> None:
        self.dataset_name = dataset_name
        self.random_seed = int(random_seed)
        self.pipeline = RelBenchDuckDBPipeline(
            RelBenchDuckDBConfig(
                dataset_name=dataset_name,
                download=download,
                upto_test_timestamp=False,
            )
        )
        # `_make_dataset` resolves the dataset cache dir without forcing a pandas materialization.
        self.pipeline._make_dataset()
        if not self.pipeline.config.parquet_dir.exists():
            # Fresh caches may require one materialization step before parquet-backed lazy access is possible.
            self.pipeline.materialize()
        self.parquet_dir = self.pipeline.config.parquet_dir
        self.conn = duckdb.connect(str(self.pipeline.config.duckdb_path), read_only=False)
        self._cutoff: pd.Timestamp | None = None
        self._row_count_cache: dict[tuple[str, bool], int] = {}
        self._register_views()
        self._metadata_by_table = self._build_metadata()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def __enter__(self) -> "Phase1DataAccess":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def table_names(self) -> list[str]:
        return sorted(self._metadata_by_table)

    def table_meta(self, table_name: str) -> TableMetadata:
        return self._metadata_by_table[table_name]

    def set_cutoff(self, cutoff: pd.Timestamp) -> None:
        self._cutoff = pd.Timestamp(cutoff)

    def cutoff(self) -> pd.Timestamp:
        if self._cutoff is None:
            raise RuntimeError("Global cutoff is not set yet.")
        return self._cutoff

    def fetchdf(self, sql: str) -> pd.DataFrame:
        return self.conn.execute(sql).fetchdf()

    def fetch_scalar(self, sql: str) -> Any:
        row = self.conn.execute(sql).fetchone()
        return None if row is None else row[0]

    def count_rows(self, table_name: str, *, filtered: bool) -> int:
        key = (table_name, filtered)
        if key in self._row_count_cache:
            return self._row_count_cache[key]
        meta = self.table_meta(table_name)
        if not filtered and meta.raw_row_count_hint is not None:
            self._row_count_cache[key] = int(meta.raw_row_count_hint)
            return int(meta.raw_row_count_hint)
        count = int(self.fetch_scalar(f"SELECT COUNT(*) FROM ({self.relation_sql(table_name, filtered=filtered)}) t") or 0)
        self._row_count_cache[key] = count
        return count

    def count_invalid_time_rows(self, table_name: str) -> int:
        meta = self.table_meta(table_name)
        if meta.time_column is None:
            return 0
        tcol = quote_ident(meta.time_column)
        sql = (
            f"SELECT COUNT(*) FROM {quote_ident(table_name)} "
            f"WHERE {tcol} IS NOT NULL AND TRY_CAST({tcol} AS TIMESTAMP) IS NULL"
        )
        return int(self.fetch_scalar(sql) or 0)

    def relation_sql(
        self,
        table_name: str,
        *,
        filtered: bool,
        columns: list[str] | None = None,
    ) -> str:
        meta = self.table_meta(table_name)
        select_cols = "*"
        if columns:
            select_cols = ", ".join(quote_ident(col) for col in columns)
        base = f"SELECT {select_cols} FROM {quote_ident(table_name)}"
        if not filtered or meta.time_column is None:
            return base
        cutoff = self.cutoff()
        tcol = quote_ident(meta.time_column)
        return (
            f"{base} WHERE TRY_CAST({tcol} AS TIMESTAMP) IS NOT NULL "
            f"AND TRY_CAST({tcol} AS TIMESTAMP) < {sql_timestamp_literal(cutoff)}"
        )

    def sampled_relation_sql(
        self,
        table_name: str,
        *,
        filtered: bool,
        sample_size: int,
        columns: list[str] | None = None,
    ) -> str:
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        meta = self.table_meta(table_name)
        rel_sql = self.relation_sql(table_name, filtered=filtered, columns=columns)
        order_cols = self._sample_order_columns(meta, columns)
        order_expr = ", ".join(f"hash(rel.{quote_ident(col)})" for col in order_cols)
        return f"SELECT * FROM ({rel_sql}) rel ORDER BY {order_expr} LIMIT {int(sample_size)}"

    def sample_df(
        self,
        table_name: str,
        *,
        filtered: bool,
        sample_size: int,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        sql = self.sampled_relation_sql(
            table_name,
            filtered=filtered,
            sample_size=sample_size,
            columns=columns,
        )
        return self.fetchdf(sql)

    def has_persisted_metadata(self) -> bool:
        rows = self._load_persisted_metadata_rows()
        return bool(rows)

    def dataset_manifest(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "duckdb_path": str(self.pipeline.config.duckdb_path),
            "parquet_dir": str(self.parquet_dir),
            "tables": {
                tname: {
                    "parquet_path": str(meta.parquet_path),
                    "columns": meta.columns,
                    "dtypes": meta.dtypes,
                    "provided_primary_key": meta.provided_primary_key,
                    "provided_foreign_keys": meta.provided_foreign_keys,
                    "time_column": meta.time_column,
                    "detected_time_columns": meta.detected_time_columns,
                    "metadata_source": meta.metadata_source,
                    "metadata_warnings": meta.metadata_warnings,
                    "raw_row_count_hint": meta.raw_row_count_hint,
                }
                for tname, meta in self._metadata_by_table.items()
            },
        }

    def _register_views(self) -> None:
        self.conn.execute("CREATE SCHEMA IF NOT EXISTS relbench")
        self.conn.execute("CREATE SCHEMA IF NOT EXISTS relbench_meta")
        if not self.parquet_dir.exists():
            raise FileNotFoundError(
                f"Expected parquet directory for dataset={self.dataset_name} at {self.parquet_dir}"
            )
        for parquet_path in sorted(self.parquet_dir.glob("*.parquet")):
            table_name = parquet_path.stem
            path_literal = sql_str_literal(parquet_path.as_posix())
            table_ident = quote_ident(table_name)
            relbench_ident = quote_qualified("relbench", table_name)
            self.conn.execute(
                f"CREATE OR REPLACE VIEW {table_ident} AS SELECT * FROM read_parquet({path_literal})"
            )
            self.conn.execute(
                f"CREATE OR REPLACE VIEW {relbench_ident} AS SELECT * FROM read_parquet({path_literal})"
            )

    def _build_metadata(self) -> dict[str, TableMetadata]:
        persisted = self._load_persisted_metadata_rows()
        metadata: dict[str, TableMetadata] = {}
        for parquet_path in sorted(self.parquet_dir.glob("*.parquet")):
            table_name = parquet_path.stem
            describe = self.fetchdf(
                f"DESCRIBE SELECT * FROM read_parquet({sql_str_literal(parquet_path.as_posix())})"
            )
            columns = [
                str(v)
                for v in describe["column_name"].tolist()
                if str(v) and str(v) != "__index_level_0__"
            ]
            dtypes = {
                str(row["column_name"]): str(row["column_type"])
                for row in describe.to_dict(orient="records")
                if row.get("column_name") and row.get("column_name") != "__index_level_0__"
            }
            persisted_row = persisted.get(table_name, {})
            meta = self._build_table_metadata(
                table_name=table_name,
                parquet_path=parquet_path,
                columns=columns,
                dtypes=dtypes,
                persisted_row=persisted_row,
            )
            metadata[table_name] = meta
        return metadata

    def _build_table_metadata(
        self,
        *,
        table_name: str,
        parquet_path: Path,
        columns: list[str],
        dtypes: dict[str, str],
        persisted_row: dict[str, Any],
    ) -> TableMetadata:
        warnings: list[str] = []
        lower_to_actual = {col.lower(): col for col in columns}
        detected_time_columns = [
            col
            for col in columns
            if is_temporal_dtype_name(dtypes.get(col, "")) or self._looks_like_time_name(col)
        ]
        provided_pk = self._normalized_persisted_column(lower_to_actual, persisted_row.get("primary_key"))
        persisted_time = self._normalized_persisted_column(lower_to_actual, persisted_row.get("time_column"))
        provided_fks = self._normalized_foreign_keys(lower_to_actual, persisted_row.get("foreign_keys", {}))

        time_column = persisted_time
        metadata_source = "relbench_meta_table_info" if persisted_row else "heuristic"
        if time_column is None and len(detected_time_columns) == 1:
            time_column = detected_time_columns[0]
        elif time_column is None and len(detected_time_columns) > 1:
            time_column = sorted(detected_time_columns)[0]
            warnings.append(
                f"multiple_time_candidates={','.join(sorted(detected_time_columns))};chosen={time_column}"
            )
        if provided_pk is None:
            provided_pk = self._guess_primary_key(table_name, columns)
            if provided_pk is not None:
                warnings.append(f"primary_key_heuristic={provided_pk}")

        raw_row_count_hint = None
        if "num_rows" in persisted_row and persisted_row["num_rows"] is not None:
            try:
                raw_row_count_hint = int(persisted_row["num_rows"])
            except Exception:
                raw_row_count_hint = None

        return TableMetadata(
            table_name=table_name,
            parquet_path=parquet_path,
            columns=columns,
            dtypes=dtypes,
            provided_primary_key=provided_pk,
            provided_foreign_keys=provided_fks,
            time_column=time_column,
            detected_time_columns=sorted(detected_time_columns),
            metadata_source=metadata_source,
            metadata_warnings=warnings,
            raw_row_count_hint=raw_row_count_hint,
        )

    def _load_persisted_metadata_rows(self) -> dict[str, dict[str, Any]]:
        try:
            exists = self.fetch_scalar(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema = 'relbench_meta' AND table_name = 'table_info'"
            )
            if not exists:
                return {}
            frame = self.fetchdf(
                "SELECT table_name, parquet_path, num_rows, num_columns, primary_key, time_column, foreign_keys "
                "FROM relbench_meta.table_info"
            )
        except Exception:
            return {}

        rows: dict[str, dict[str, Any]] = {}
        for row in frame.to_dict(orient="records"):
            foreign_keys = row.get("foreign_keys")
            if isinstance(foreign_keys, str):
                try:
                    foreign_keys = json.loads(foreign_keys)
                except Exception:
                    foreign_keys = {}
            rows[str(row.get("table_name", ""))] = {
                "parquet_path": row.get("parquet_path"),
                "num_rows": row.get("num_rows"),
                "num_columns": row.get("num_columns"),
                "primary_key": row.get("primary_key"),
                "time_column": row.get("time_column"),
                "foreign_keys": foreign_keys if isinstance(foreign_keys, dict) else {},
            }
        return rows

    def _guess_primary_key(self, table_name: str, columns: list[str]) -> str | None:
        lower_to_actual = {col.lower(): col for col in columns}
        singular = singularize_identifier(table_name)
        candidates = [
            "id",
            f"{table_name.lower()}_id",
            f"{singular}_id",
            singular,
        ]
        for candidate in candidates:
            actual = lower_to_actual.get(candidate)
            if actual is not None:
                return actual
        return None

    def _normalized_persisted_column(
        self,
        lower_to_actual: dict[str, str],
        candidate: Any,
    ) -> str | None:
        if candidate is None:
            return None
        candidate_s = str(candidate).strip()
        if not candidate_s:
            return None
        if candidate_s in lower_to_actual.values():
            return candidate_s
        return lower_to_actual.get(candidate_s.lower())

    def _normalized_foreign_keys(
        self,
        lower_to_actual: dict[str, str],
        payload: Any,
    ) -> dict[str, str]:
        if not isinstance(payload, dict):
            return {}
        out: dict[str, str] = {}
        for key, value in payload.items():
            actual_key = self._normalized_persisted_column(lower_to_actual, key)
            if actual_key is None:
                continue
            out[actual_key] = str(value)
        return out

    def _looks_like_time_name(self, column_name: str) -> bool:
        toks = normalize_identifier_tokens(column_name)
        return bool(toks & {"time", "timestamp", "date", "datetime"})

    def _sample_order_columns(
        self,
        meta: TableMetadata,
        selected_columns: list[str] | None,
    ) -> list[str]:
        selected = set(selected_columns or meta.columns)
        candidates: list[str] = []
        for col in [meta.provided_primary_key, meta.time_column]:
            if col and col in selected and col not in candidates:
                candidates.append(col)
        for col in meta.columns:
            if col in selected and col not in candidates:
                candidates.append(col)
            if len(candidates) >= 3:
                break
        if not candidates:
            candidates = [meta.columns[0]]
        return candidates
