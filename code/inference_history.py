from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import torch

from load_inference_config import normalize_inference_config
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
    if resource.history_source == "task_table":
        frame = resource.frame.copy()
        frame = frame.rename(columns={resource.time_col: "timestamp"})
        return TemporalHistoryStore.from_frame(frame, resource.entity_col, "timestamp"), False

    if resource.history_source != "dataset":
        raise ValueError(f"Unsupported history source: {resource.history_source}")

    return build_raw_dataset_timestamps(resource.db), True


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
    candidate_timestamps = store[store < query_time]
    if candidate_timestamps.empty:
        return []

    history_table = resource.task_object.make_table(resource.db, candidate_timestamps)
    history_frame = history_table.df
    history_frame = history_frame[history_frame[resource.entity_col] == entity_value]
    history_frame = history_frame[
        pd.to_datetime(history_frame[resource.time_col]) <= effective_cutoff
    ]
    if history_frame.empty:
        return []

    if resource.history_sampling_strategy not in {
        "random_prior",
        "most_recent_k",
        "recent_min_overlap",
    }:
        raise ValueError(
            f"Unsupported history sampling strategy: {resource.history_sampling_strategy}"
        )

    history_frame = history_frame.sort_values(by=[resource.time_col], kind="stable").reset_index(drop=True)
    history_frame = history_frame.drop_duplicates(
        subset=[resource.entity_col, resource.time_col],
        keep="first",
    ).reset_index(drop=True)
    if resource.history_sampling_strategy == "most_recent_k":
        sampled_records = history_frame.tail(resource.history_length)
    elif resource.history_sampling_strategy == "recent_min_overlap":
        timestamp_values = [
            int(pd.Timestamp(value).value) for value in history_frame[resource.time_col].tolist()
        ]
        selected_indices = _select_recent_min_overlap_indices(
            timestamp_values,
            resource.history_length,
            resource.label_horizon_ns,
        )
        sampled_records = history_frame.iloc[selected_indices]
    else:
        sample_size = min(resource.history_length, len(history_frame))
        permutation = torch.randperm(len(history_frame))[:sample_size].tolist()
        sampled_records = history_frame.iloc[permutation].sort_values(
            by=[resource.time_col],
            kind="stable",
        )
    return _materialize_history(
        sampled_records.to_dict(orient="records"),
        resource.entity_col,
        resource.time_col,
        resource.output_col,
    )


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
