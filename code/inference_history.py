from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import torch

from load_inference_config import normalize_inference_config
from task_history_queries import get_task_history_query, get_task_history_query_bulk
from train_data import (
    TaskResource,
    build_task_resource,
    maybe_materialize_dataset,
)


def _select_recent_min_overlap_indices(
    timestamp_values: list[int],
    k: int,
    min_gap_ns: int,
) -> list[int]:
    if k <= 0 or not timestamp_values:
        return []

    selected: list[int] = []
    for idx in range(len(timestamp_values) - 1, -1, -1):
        ts_value = timestamp_values[idx]
        if all(abs(ts_value - timestamp_values[prev_idx]) >= min_gap_ns for prev_idx in selected):
            selected.append(idx)
            if len(selected) == k:
                break

    if len(selected) < k:
        seen = set(selected)
        for idx in range(len(timestamp_values) - 1, -1, -1):
            if idx in seen:
                continue
            selected.append(idx)
            if len(selected) == k:
                break

    selected.sort(key=lambda idx: timestamp_values[idx])
    return selected


@dataclass(slots=True)
class TemporalHistoryStore:
    entity_to_history: dict[Any, list[tuple[int, int]]] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def _timestamp_to_int(value: Any) -> int:
        if isinstance(value, (int, float)):
            return int(value)
        return int(pd.Timestamp(value).value)

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        entity_col: str,
        time_col: str,
    ) -> "TemporalHistoryStore":
        store = cls()
        for row in frame.itertuples(index=False):
            record = row._asdict()
            entity_value = record[entity_col]
            timestamp_value = cls._timestamp_to_int(record[time_col])
            row_id = len(store.records)
            store.records.append(record)
            store.entity_to_history.setdefault(entity_value, []).append((timestamp_value, row_id))

        for history in store.entity_to_history.values():
            history.sort(key=lambda item: item[0])
        return store

    def get_history_before(
        self,
        entity_value: Any,
        cutoff_time: Any,
        k: int,
        *,
        include_equal: bool = False,
        sampling_strategy: str = "most_recent_k",
        min_gap_ns: int = 0,
    ) -> list[dict[str, Any]]:
        history = self.entity_to_history.get(entity_value)
        if not history or k <= 0:
            return []

        cutoff = self._timestamp_to_int(cutoff_time)
        timestamps = [timestamp for timestamp, _ in history]
        side = "right" if include_equal else "left"
        pos = int(torch.searchsorted(torch.tensor(timestamps, dtype=torch.long), cutoff, right=(side == "right")).item())
        if pos == 0:
            return []

        candidate_row_ids = [row_id for _, row_id in history[:pos]]
        if sampling_strategy == "most_recent_k":
            sampled_row_ids = candidate_row_ids[-k:]
        elif sampling_strategy == "recent_min_overlap":
            candidate_timestamps = [timestamp for timestamp, _ in history[:pos]]
            selected_indices = _select_recent_min_overlap_indices(
                candidate_timestamps,
                k,
                min_gap_ns,
            )
            sampled_row_ids = [candidate_row_ids[idx] for idx in selected_indices]
        elif sampling_strategy == "random_prior":
            candidate_row_ids_tensor = torch.tensor(candidate_row_ids, dtype=torch.long)
            sample_size = min(k, int(candidate_row_ids_tensor.numel()))
            permutation = torch.randperm(candidate_row_ids_tensor.numel())[:sample_size]
            sampled_row_ids = candidate_row_ids_tensor[permutation].tolist()
        else:
            raise ValueError(
                f"Unsupported history sampling strategy: {sampling_strategy}"
            )
        sampled_row_ids.sort(key=lambda row_id: self._timestamp_to_int(self.records[row_id]["timestamp"]))
        return [self.records[row_id] for row_id in sampled_row_ids]


@dataclass(slots=True)
class DatasetHistoryDiskCache:
    db_path: Path
    entity_col: str
    time_col: str


_DATASET_HISTORY_CACHE_VERSION = "v1"
_DATASET_HISTORY_CHUNK_SIZE = 16
_DATASET_HISTORY_MAX_ROWS_PER_CHUNK = 250_000


def inference_history_cache_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / "inference_history_cache"


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _dataset_history_cache_path(resource: TaskResource) -> Path:
    filename = (
        f"{_DATASET_HISTORY_CACHE_VERSION}__{resource.dataset}__{resource.task}"
        "__dataset_history.duckdb"
    )
    return inference_history_cache_dir() / filename


def _create_empty_history_table(conn: duckdb.DuckDBPyConnection, resource: TaskResource) -> None:
    empty_frame = resource.task_object.get_table("train", mask_input_cols=False).df.head(0)
    conn.register("empty_history_frame", empty_frame)
    conn.execute("CREATE TABLE history AS SELECT * FROM empty_history_frame")
    conn.unregister("empty_history_frame")


def _iter_timestamp_batches(timestamps: pd.Series, batch_size: int):
    for chunk_start in range(0, len(timestamps), batch_size):
        chunk = timestamps.iloc[chunk_start : chunk_start + batch_size]
        if not chunk.empty:
            yield chunk


def _materialize_history_chunks(
    resource: TaskResource,
    timestamps: pd.Series,
    *,
    max_rows_per_chunk: int,
) -> list[pd.DataFrame]:
    if timestamps.empty:
        return []

    history_table = resource.task_object.make_table(resource.db, timestamps)
    chunk_frame = history_table.df
    if chunk_frame.empty:
        return []

    if len(chunk_frame) <= max_rows_per_chunk or len(timestamps) == 1:
        return [chunk_frame]

    midpoint = len(timestamps) // 2
    left = timestamps.iloc[:midpoint]
    right = timestamps.iloc[midpoint:]
    out: list[pd.DataFrame] = []
    out.extend(
        _materialize_history_chunks(
            resource,
            left,
            max_rows_per_chunk=max_rows_per_chunk,
        )
    )
    out.extend(
        _materialize_history_chunks(
            resource,
            right,
            max_rows_per_chunk=max_rows_per_chunk,
        )
    )
    return out


def _build_dataset_history_disk_cache(
    resource: TaskResource,
    timestamps: pd.Series,
    *,
    force_rebuild: bool = False,
    verbose: bool = False,
) -> DatasetHistoryDiskCache:
    cache_path = _dataset_history_cache_path(resource)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if force_rebuild and cache_path.exists():
        cache_path.unlink()

    if cache_path.exists():
        return DatasetHistoryDiskCache(
            db_path=cache_path,
            entity_col=resource.entity_col,
            time_col=resource.time_col,
        )

    temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()

    if verbose:
        print(f"building dataset history cache at {cache_path}")

    conn = duckdb.connect(str(temp_path))
    conn.execute(f"PRAGMA threads={max(1, resource.history_parallel_workers)}")

    history_initialized = False
    total_rows = 0
    entity_col_sql = _quote_ident(resource.entity_col)
    time_col_sql = _quote_ident(resource.time_col)

    try:
        base_batches = list(_iter_timestamp_batches(timestamps, _DATASET_HISTORY_CHUNK_SIZE))
        num_base_batches = len(base_batches)
        for batch_index, chunk in enumerate(base_batches, start=1):
            materialized_chunks = _materialize_history_chunks(
                resource,
                chunk,
                max_rows_per_chunk=_DATASET_HISTORY_MAX_ROWS_PER_CHUNK,
            )
            for chunk_frame in materialized_chunks:
                total_rows += len(chunk_frame)
                conn.register("history_chunk", chunk_frame)
                if not history_initialized:
                    conn.execute("CREATE TABLE history AS SELECT * FROM history_chunk")
                    history_initialized = True
                else:
                    conn.execute("INSERT INTO history SELECT * FROM history_chunk")
                conn.unregister("history_chunk")

            if verbose and (batch_index == 1 or batch_index % 25 == 0 or batch_index == num_base_batches):
                print(
                    "  history cache batch "
                    f"{batch_index}/{max(1, num_base_batches)} "
                    f"(rows so far={total_rows})"
                )

        if not history_initialized:
            _create_empty_history_table(conn, resource)
        else:
            conn.execute(
                f"""
                CREATE TABLE history_dedup AS
                SELECT * EXCLUDE (__rn)
                FROM (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY {entity_col_sql}, {time_col_sql}
                               ORDER BY {time_col_sql}
                           ) AS __rn
                    FROM history
                )
                WHERE __rn = 1
                ORDER BY {entity_col_sql}, {time_col_sql}
                """
            )
            conn.execute("DROP TABLE history")
            conn.execute("ALTER TABLE history_dedup RENAME TO history")

        conn.execute(
            f"CREATE INDEX history_entity_time_idx ON history ({entity_col_sql}, {time_col_sql})"
        )
        conn.execute("CHECKPOINT")
    finally:
        conn.close()

    temp_path.replace(cache_path)
    if verbose:
        print(f"finished dataset history cache at {cache_path}")

    return DatasetHistoryDiskCache(
        db_path=cache_path,
        entity_col=resource.entity_col,
        time_col=resource.time_col,
    )


def build_raw_dataset_timestamps(db: Any) -> pd.Series:
    timestamp_series: list[pd.Series] = []
    for table in db.table_dict.values():
        time_col = table.time_col
        if time_col is None or time_col not in table.df.columns:
            continue
        series = pd.to_datetime(table.df[time_col], errors="coerce").dropna()
        if not series.empty:
            timestamp_series.append(series)

    if not timestamp_series:
        return pd.Series(dtype="datetime64[ns]")

    timestamps = pd.concat(timestamp_series, ignore_index=True)
    timestamps = pd.Series(timestamps.sort_values().unique())
    return timestamps.reset_index(drop=True)


def _materialize_history(
    records: list[dict[str, Any]],
    entity_col: str,
    time_col: str,
    output_col: str,
) -> list[dict[str, Any]]:
    return [dict(record) for record in records]


def _sample_history_frame(
    history_frame: pd.DataFrame,
    resource: TaskResource,
) -> pd.DataFrame:
    if history_frame.empty:
        return history_frame

    if resource.history_sampling_strategy not in {
        "random_prior",
        "most_recent_k",
        "recent_min_overlap",
    }:
        raise ValueError(
            f"Unsupported history sampling strategy: {resource.history_sampling_strategy}"
        )

    history_frame = history_frame.sort_values(
        by=[resource.time_col],
        kind="stable",
    ).reset_index(drop=True)
    history_frame = history_frame.drop_duplicates(
        subset=[resource.entity_col, resource.time_col],
        keep="first",
    ).reset_index(drop=True)
    if resource.history_sampling_strategy == "most_recent_k":
        return history_frame.tail(resource.history_length)
    if resource.history_sampling_strategy == "recent_min_overlap":
        timestamp_values = [
            int(pd.Timestamp(value).value) for value in history_frame[resource.time_col].tolist()
        ]
        selected_indices = _select_recent_min_overlap_indices(
            timestamp_values,
            resource.history_length,
            resource.label_horizon_ns,
        )
        return history_frame.iloc[selected_indices]

    sample_size = min(resource.history_length, len(history_frame))
    permutation = torch.randperm(len(history_frame))[:sample_size].tolist()
    return history_frame.iloc[permutation].sort_values(
        by=[resource.time_col],
        kind="stable",
    )


def build_inference_resource(
    config: dict[str, Any],
    no_dataset_download: bool = False,
    no_task_download: bool = False,
) -> TaskResource:
    normalized = (
        config if "datasets" in config else normalize_inference_config(config)
    )
    normalized = dict(normalized)
    if normalized.get("use_full_raw_db", True):
        normalized["include_future_rows"] = True
    dataset_name = normalized["datasets"][0]["name"]
    task_name = normalized["datasets"][0]["tasks"][0]
    db = maybe_materialize_dataset(dataset_name, normalized, no_dataset_download)
    return build_task_resource(
        dataset_name=dataset_name,
        task_name=task_name,
        config=normalized,
        no_task_download=no_task_download,
        db=db,
    )


def build_history_store(resource: TaskResource) -> tuple[Any, bool]:
    return build_history_store_with_options(resource)


def build_history_store_with_options(
    resource: TaskResource,
    *,
    force_rebuild: bool = False,
    verbose: bool = False,
) -> tuple[Any, bool]:
    if resource.history_source == "task_table":
        frame = resource.frame.copy()
        frame = frame.rename(columns={resource.time_col: "timestamp"})
        return TemporalHistoryStore.from_frame(frame, resource.entity_col, "timestamp"), False

    if resource.history_source != "dataset":
        raise ValueError(f"Unsupported history source: {resource.history_source}")

    optimized_query = get_task_history_query(resource.dataset, resource.task)
    optimized_bulk_query = get_task_history_query_bulk(resource.dataset, resource.task)
    if optimized_query is not None:
        if verbose:
            print(f"using optimized history query for {resource.dataset}/{resource.task}")
        return {
            "optimized_query": optimized_query,
            "optimized_bulk_query": optimized_bulk_query,
        }, True

    timestamps = build_raw_dataset_timestamps(resource.db)
    cache = _build_dataset_history_disk_cache(
        resource,
        timestamps,
        force_rebuild=force_rebuild,
        verbose=verbose,
    )
    return cache, True


def build_inference_history(
    store: Any,
    resource: TaskResource,
    entity_value: Any,
    cutoff_time: Any,
) -> list[dict[str, Any]]:
    if resource.history_source == "task_table":
        records = store.get_history_before(
            entity_value,
            cutoff_time,
            resource.history_length,
            include_equal=False,
            sampling_strategy=resource.history_sampling_strategy,
            min_gap_ns=resource.label_horizon_ns,
        )
        return _materialize_history(records, resource.entity_col, "timestamp", resource.output_col)

    if resource.history_source != "dataset":
        raise ValueError(f"Unsupported history source: {resource.history_source}")

    query_time = pd.Timestamp(cutoff_time)
    effective_cutoff = query_time - pd.to_timedelta(
        resource.label_horizon_ns,
        unit="ns",
    )
    optimized_query = None
    if isinstance(store, dict):
        optimized_query = store.get("optimized_query")
    if optimized_query is not None:
        history_frame = optimized_query(resource, entity_value, effective_cutoff)
        if history_frame.empty:
            return []
        history_frame = history_frame[
            pd.to_datetime(history_frame[resource.time_col]) <= effective_cutoff
        ]
        if history_frame.empty:
            return []
    else:
        with duckdb.connect(str(store.db_path), read_only=True) as conn:
            history_frame = conn.execute(
                f"""
                SELECT *
                FROM history
                WHERE {_quote_ident(store.entity_col)} = ?
                  AND {_quote_ident(store.time_col)} <= ?
                ORDER BY {_quote_ident(store.time_col)}
                """,
                [entity_value, effective_cutoff],
            ).fetch_df()
        if history_frame.empty:
            return []

    sampled_records = _sample_history_frame(history_frame, resource)
    return _materialize_history(
        sampled_records.to_dict(orient="records"),
        resource.entity_col,
        resource.time_col,
        resource.output_col,
    )


def build_inference_histories_bulk(
    store: Any,
    resource: TaskResource,
    requests: list[tuple[int, Any, Any]],
) -> dict[int, list[dict[str, Any]]]:
    if not requests:
        return {}

    out: dict[int, list[dict[str, Any]]] = {}
    if resource.history_source == "task_table":
        for request_id, entity_value, cutoff_time in requests:
            out[request_id] = build_inference_history(
                store,
                resource,
                entity_value,
                cutoff_time,
            )
        return out

    if resource.history_source != "dataset":
        raise ValueError(f"Unsupported history source: {resource.history_source}")

    optimized_bulk_query = None
    if isinstance(store, dict):
        optimized_bulk_query = store.get("optimized_bulk_query")

    if optimized_bulk_query is None:
        for request_id, entity_value, cutoff_time in requests:
            out[request_id] = build_inference_history(
                store,
                resource,
                entity_value,
                cutoff_time,
            )
        return out

    request_rows: list[dict[str, Any]] = []
    for request_id, entity_value, cutoff_time in requests:
        query_time = pd.Timestamp(cutoff_time)
        effective_cutoff = query_time - pd.to_timedelta(
            resource.label_horizon_ns,
            unit="ns",
        )
        request_rows.append(
            {
                "request_id": int(request_id),
                "entity_value": entity_value,
                "effective_cutoff": effective_cutoff,
                "query_time": query_time,
            }
        )

    request_frame = pd.DataFrame(request_rows)
    history_frame = optimized_bulk_query(resource, request_frame)

    if history_frame.empty:
        for request_id, _, _ in requests:
            out[int(request_id)] = []
        return out

    history_by_request = {
        int(request_id): frame.reset_index(drop=True)
        for request_id, frame in history_frame.groupby("request_id", sort=False)
    }

    for request_row in request_rows:
        request_id = int(request_row["request_id"])
        query_time = request_row["query_time"]
        frame = history_by_request.get(request_id)
        if frame is None or frame.empty:
            out[request_id] = []
            continue
        if "request_id" in frame.columns:
            frame = frame.drop(columns=["request_id"])
        frame = frame[
            pd.to_datetime(frame[resource.time_col]) <= request_row["effective_cutoff"]
        ]
        if frame.empty:
            out[request_id] = []
            continue
        sampled_records = _sample_history_frame(frame, resource)
        materialized = _materialize_history(
            sampled_records.to_dict(orient="records"),
            resource.entity_col,
            resource.time_col,
            resource.output_col,
        )
        validate_history_non_overlap(materialized, resource, query_time)
        out[request_id] = materialized

    return out


def validate_history_non_overlap(
    history: list[dict[str, Any]],
    resource: TaskResource,
    query_time: Any,
) -> None:
    query_timestamp = pd.Timestamp(query_time)
    seen_keys: set[tuple[Any, pd.Timestamp]] = set()

    for item in history:
        item_timestamp = pd.Timestamp(item[resource.time_col])
        item_key = (item[resource.entity_col], item_timestamp)

        if item_key in seen_keys:
            raise ValueError(
                f"Duplicate history item detected for entity={item[resource.entity_col]} at timestamp={item_timestamp}."
            )
        seen_keys.add(item_key)

        if item_timestamp >= query_timestamp:
            raise ValueError(
                f"History item timestamp {item_timestamp} is not strictly before query time {query_timestamp}."
            )

        if resource.history_source == "dataset":
            if item_timestamp + resource.task_object.timedelta > query_timestamp:
                raise ValueError(
                    "Dataset history item violates non-overlap constraint: "
                    f"{item_timestamp} + {resource.task_object.timedelta} > {query_timestamp}."
                )
