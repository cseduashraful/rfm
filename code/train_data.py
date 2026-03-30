from __future__ import annotations

from dataclasses import dataclass, field
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from relbench.base import Database
from relbench.tasks import get_task, get_task_names

from data_config import RelBenchDuckDBConfig
from data_pipeline import RelBenchDuckDBPipeline


_HISTORY_POOL: mp.pool.Pool | None = None
_HISTORY_POOL_WORKERS: int | None = None
_HISTORY_POOL_RESOURCES: dict[tuple[str, str], "TaskResource"] | None = None


@dataclass(slots=True)
class MixedTaskExampleRef:
    dataset: str
    task: str
    split: str
    row_index: int


@dataclass(slots=True)
class TaskResource:
    dataset: str
    task: str
    split: str
    frame: pd.DataFrame
    time_col: str
    output_col: str
    entity_col: str
    entity_table: str
    history_length: int
    history_sampling_strategy: str
    history_source: str
    history_parallel_mode: str
    history_parallel_workers: int
    cache_dataset_history_labels: bool
    cache_train_history_candidates: bool
    task_object: Any
    db: Database
    task_frame_time_values: torch.Tensor
    task_schedule_timestamps: torch.Tensor
    label_horizon_ns: int
    entity_to_row_indices: dict[Any, torch.Tensor]
    entity_to_time_values: dict[Any, torch.Tensor]
    history_candidate_offsets: torch.Tensor | None = None
    history_candidate_indices: torch.Tensor | None = None
    dataset_history_label_cache: dict[int, dict[Any, dict[str, Any]]] = field(
        default_factory=dict
    )


class MixedTaskTrainDataset(Dataset):
    def __init__(
        self,
        examples: list[MixedTaskExampleRef],
        task_resources: dict[tuple[str, str], TaskResource],
    ):
        self.examples = examples
        self.task_resources = task_resources

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        resource = self.task_resources[(example.dataset, example.task)]
        row = resource.frame.iloc[example.row_index].to_dict()
        return {
            "dataset": example.dataset,
            "task": example.task,
            "split": example.split,
            "row_index": example.row_index,
            "timestamp": row[resource.time_col],
            "output": row[resource.output_col],
            "train_item": row,
        }


def collate_mixed_batch(batch: list[dict[str, Any]]) -> dict[str, list[Any]]:
    return {
        "dataset": [item["dataset"] for item in batch],
        "task": [item["task"] for item in batch],
        "split": [item["split"] for item in batch],
        "row_index": [item["row_index"] for item in batch],
        "timestamp": [item["timestamp"] for item in batch],
        "output": [item["output"] for item in batch],
        "train_item": [item["train_item"] for item in batch],
    }


def infer_output_column(task: Any) -> str:
    if getattr(task, "target_col", None) is not None:
        return task.target_col
    if getattr(task, "dst_entity_col", None) is not None:
        return task.dst_entity_col
    raise ValueError(f"Could not infer output column for task {task}.")


def infer_entity_column(task: Any) -> str:
    if getattr(task, "entity_col", None) is not None:
        return task.entity_col
    raise ValueError(
        f"Task {task} is not currently supported for history sampling because "
        "it does not expose entity_col."
    )


def maybe_materialize_dataset(
    dataset_name: str,
    config: dict[str, Any],
    no_dataset_download: bool,
) -> Database:
    pipeline = RelBenchDuckDBPipeline(
        RelBenchDuckDBConfig(
            dataset_name=dataset_name,
            download=(not no_dataset_download)
            and config.get("dataset_download", True),
            upto_test_timestamp=not config.get("include_future_rows", False),
        )
    )
    return pipeline.materialize()


def history_cache_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / "history_cache"


def preprocessed_train_dir(config_name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / "preprocessed_train" / config_name


def build_task_resource(
    dataset_name: str,
    task_name: str,
    config: dict[str, Any],
    no_task_download: bool,
    db: Database,
) -> TaskResource:
    task = get_task(
        dataset_name,
        task_name,
        download=(not no_task_download) and config.get("task_download", True),
    )
    table = task.get_table("train", mask_input_cols=False)
    frame = table.df.reset_index(drop=True).copy()
    time_col = getattr(task, "time_col", table.time_col)
    output_col = infer_output_column(task)
    entity_col = infer_entity_column(task)

    frame = frame.sort_values(by=[time_col], kind="stable").reset_index(drop=True)
    time_values = torch.as_tensor(
        frame[time_col].astype("int64").to_numpy(),
        dtype=torch.long,
    )

    entity_to_row_indices: dict[Any, torch.Tensor] = {}
    entity_to_time_values: dict[Any, torch.Tensor] = {}
    grouped = frame.groupby(entity_col, sort=False).indices
    for entity_value, row_indices in grouped.items():
        indices = torch.as_tensor(row_indices, dtype=torch.long)
        entity_to_row_indices[entity_value] = indices
        entity_to_time_values[entity_value] = time_values[indices]

    entity_table = getattr(task, "entity_table", None)
    if entity_table is None:
        raise ValueError(f"Task {task} is not currently supported because entity_table is missing.")

    return TaskResource(
        dataset=dataset_name,
        task=task_name,
        split="train",
        frame=frame,
        time_col=time_col,
        output_col=output_col,
        entity_col=entity_col,
        entity_table=entity_table,
        history_length=config["history_length"],
        history_sampling_strategy=config.get("history_sampling_strategy", "random_prior"),
        history_source=config["history_source"],
        history_parallel_mode=config["history_parallel_mode"],
        history_parallel_workers=config["history_parallel_workers"],
        cache_dataset_history_labels=config["cache_dataset_history_labels"],
        cache_train_history_candidates=config["cache_train_history_candidates"],
        task_object=task,
        db=db,
        task_frame_time_values=time_values,
        task_schedule_timestamps=torch.unique_consecutive(time_values),
        label_horizon_ns=int(task.timedelta.value),
        entity_to_row_indices=entity_to_row_indices,
        entity_to_time_values=entity_to_time_values,
    )


def history_candidate_cache_path(resource: TaskResource) -> Path:
    cache_root = history_cache_dir()
    cache_version = "v1"
    filename = (
        f"{cache_version}__{resource.dataset}__{resource.task}__{resource.history_source}"
        f"__h{resource.history_length}__{resource.history_sampling_strategy}.pt"
    )
    return cache_root / filename


def build_history_candidate_cache(resource: TaskResource) -> tuple[torch.Tensor, torch.Tensor]:
    num_rows = len(resource.frame)
    candidate_lists: list[list[int]] = [[] for _ in range(num_rows)]

    for entity_value, row_indices in resource.entity_to_row_indices.items():
        time_values = resource.entity_to_time_values[entity_value]
        for position in range(len(row_indices)):
            current_row_index = int(row_indices[position].item())
            current_time = int(time_values[position].item())

            valid_end = position
            valid_start = 0
            while valid_start < valid_end and int(time_values[valid_start].item()) + resource.label_horizon_ns > current_time:
                valid_start += 1

            candidate_lists[current_row_index] = row_indices[valid_start:valid_end].tolist()

    offsets = torch.zeros(num_rows + 1, dtype=torch.long)
    flat_candidates: list[int] = []
    for row_index in range(num_rows):
        row_candidates = candidate_lists[row_index]
        flat_candidates.extend(row_candidates)
        offsets[row_index + 1] = len(flat_candidates)

    if flat_candidates:
        indices = torch.tensor(flat_candidates, dtype=torch.long)
    else:
        indices = torch.empty(0, dtype=torch.long)

    return offsets, indices


def load_or_create_history_candidate_cache(resource: TaskResource) -> None:
    if resource.history_source != "dataset":
        return

    cache_path = history_candidate_cache_path(resource)
    if resource.cache_train_history_candidates and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        resource.history_candidate_offsets = cached["offsets"].to(torch.long)
        resource.history_candidate_indices = cached["indices"].to(torch.long)
        return

    offsets, indices = build_history_candidate_cache(resource)
    resource.history_candidate_offsets = offsets
    resource.history_candidate_indices = indices

    if resource.cache_train_history_candidates:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"offsets": offsets, "indices": indices}, cache_path)


def build_examples_and_resources(
    config: dict[str, Any],
    no_dataset_download: bool,
    no_task_download: bool,
) -> tuple[list[MixedTaskExampleRef], dict[tuple[str, str], TaskResource]]:
    examples: list[MixedTaskExampleRef] = []
    task_resources: dict[tuple[str, str], TaskResource] = {}

    for dataset_cfg in config["datasets"]:
        dataset_name = dataset_cfg["name"]
        db = maybe_materialize_dataset(dataset_name, config, no_dataset_download)

        available_tasks = set(get_task_names(dataset_name))
        for task_name in dataset_cfg["tasks"]:
            if task_name not in available_tasks:
                raise ValueError(
                    f"Unknown task '{task_name}' for dataset '{dataset_name}'. "
                    f"Available tasks: {sorted(available_tasks)}"
                )

            resource = build_task_resource(
                dataset_name=dataset_name,
                task_name=task_name,
                config=config,
                no_task_download=no_task_download,
                db=db,
            )
            load_or_create_history_candidate_cache(resource)
            task_resources[(dataset_name, task_name)] = resource
            for row_index in range(len(resource.frame)):
                examples.append(
                    MixedTaskExampleRef(
                        dataset=dataset_name,
                        task=task_name,
                        split="train",
                        row_index=row_index,
                    )
                )

    return examples, task_resources


def sample_history_for_example(
    resource: TaskResource,
    row_index: int,
) -> list[dict[str, Any]]:
    if resource.history_length <= 0:
        return []
    if resource.history_sampling_strategy != "random_prior":
        raise ValueError(
            f"Unsupported history sampling strategy: {resource.history_sampling_strategy}"
        )

    row = resource.frame.iloc[row_index]
    entity_value = row[resource.entity_col]
    current_time_value = torch.tensor(
        int(pd.Timestamp(row[resource.time_col]).value),
        dtype=torch.long,
    )
    candidate_indices = resource.entity_to_row_indices[entity_value]
    candidate_times = resource.entity_to_time_values[entity_value]
    cutoff = int(torch.searchsorted(candidate_times, current_time_value, right=False).item())
    prior_indices = candidate_indices[:cutoff]

    if prior_indices.numel() == 0:
        return []

    sample_size = min(resource.history_length, int(prior_indices.numel()))
    permutation = torch.randperm(prior_indices.numel())[:sample_size]
    chosen = prior_indices[permutation]
    chosen_times = torch.as_tensor(
        resource.frame.iloc[chosen.tolist()][resource.time_col].astype("int64").to_numpy(),
        dtype=torch.long,
    )
    order = torch.argsort(chosen_times)
    ordered_indices = chosen[order].tolist()

    history: list[dict[str, Any]] = []
    for history_row_index in ordered_indices:
        history_row = resource.frame.iloc[history_row_index].to_dict()
        history.append(
            {
                "x": history_row[resource.entity_col],
                "timestamp": history_row[resource.time_col],
                "output": history_row[resource.output_col],
                "train_item": history_row,
            }
        )

    return history


def build_task_table_history_for_batch(
    batch: dict[str, list[Any]],
    task_resources: dict[tuple[str, str], TaskResource],
) -> list[list[dict[str, Any]]]:
    histories: list[list[dict[str, Any]]] = []
    for dataset_name, task_name, row_index in zip(
        batch["dataset"],
        batch["task"],
        batch["row_index"],
    ):
        resource = task_resources[(dataset_name, task_name)]
        histories.append(sample_history_for_example(resource, row_index))
    return histories


def build_dataset_history_for_batch(
    batch: dict[str, list[Any]],
    task_resources: dict[tuple[str, str], TaskResource],
) -> list[list[dict[str, Any]]]:
    if all(
        task_resources[(dataset_name, task_name)].history_candidate_offsets is not None
        and task_resources[(dataset_name, task_name)].history_candidate_indices is not None
        for dataset_name, task_name in zip(batch["dataset"], batch["task"])
    ):
        return build_cached_dataset_history_for_batch(batch, task_resources)

    example_requests: list[tuple[TaskResource, Any, int]] = []
    grouped_timestamps: dict[tuple[str, str], set[int]] = {}

    for dataset_name, task_name, row_index in zip(
        batch["dataset"],
        batch["task"],
        batch["row_index"],
    ):
        resource = task_resources[(dataset_name, task_name)]
        row = resource.frame.iloc[row_index]
        entity_value = row[resource.entity_col]
        current_time_value = int(pd.Timestamp(row[resource.time_col]).value)
        effective_cutoff = current_time_value - resource.label_horizon_ns
        cutoff = int(
            torch.searchsorted(
                resource.task_schedule_timestamps,
                torch.tensor(effective_cutoff, dtype=torch.long),
                right=True,
            ).item()
        )
        prior_schedule = resource.task_schedule_timestamps[:cutoff]
        example_requests.append((resource, entity_value, current_time_value))
        grouped_timestamps.setdefault((dataset_name, task_name), set()).update(
            int(x) for x in prior_schedule.tolist()
        )

    label_lookup_per_task: dict[tuple[str, str], dict[tuple[Any, int], dict[str, Any]]] = {}
    for key, timestamp_values in grouped_timestamps.items():
        resource = task_resources[key]
        if not timestamp_values:
            label_lookup_per_task[key] = {}
            continue

        label_lookup = resolve_dataset_history_labels(
            resource=resource,
            timestamp_values=timestamp_values,
        )
        label_lookup_per_task[key] = label_lookup

    histories: list[list[dict[str, Any]]] = []
    for resource, entity_value, current_time_value in example_requests:
        label_lookup = label_lookup_per_task[(resource.dataset, resource.task)]
        candidate_records: list[dict[str, Any]] = []
        for (candidate_entity, candidate_time), record in label_lookup.items():
            if candidate_entity != entity_value:
                continue
            if candidate_time + resource.label_horizon_ns > current_time_value:
                continue
            candidate_records.append(record)

        if not candidate_records:
            histories.append([])
            continue

        if resource.history_sampling_strategy != "random_prior":
            raise ValueError(
                f"Unsupported history sampling strategy: {resource.history_sampling_strategy}"
            )

        candidate_records.sort(key=lambda record: record[resource.time_col])
        sample_size = min(resource.history_length, len(candidate_records))
        permutation = torch.randperm(len(candidate_records))[:sample_size].tolist()
        sampled_records = [candidate_records[index] for index in permutation]
        sampled_records.sort(key=lambda record: record[resource.time_col])

        history: list[dict[str, Any]] = []
        for record in sampled_records:
            history.append(
                {
                    "x": record[resource.entity_col],
                    "timestamp": record[resource.time_col],
                    "output": record[resource.output_col],
                    "train_item": record,
                }
            )
        histories.append(history)
    return histories


def build_cached_dataset_history_for_batch(
    batch: dict[str, list[Any]],
    task_resources: dict[tuple[str, str], TaskResource],
) -> list[list[dict[str, Any]]]:
    first_resource = task_resources[(batch["dataset"][0], batch["task"][0])]
    if first_resource.history_parallel_mode == "grouped_vectorized":
        return build_cached_dataset_history_grouped(batch, task_resources)
    if first_resource.history_parallel_mode == "multiprocess":
        return build_cached_dataset_history_multiprocess(batch, task_resources)

    histories: list[list[dict[str, Any]]] = []

    for dataset_name, task_name, row_index in zip(
        batch["dataset"],
        batch["task"],
        batch["row_index"],
    ):
        resource = task_resources[(dataset_name, task_name)]
        offsets = resource.history_candidate_offsets
        indices = resource.history_candidate_indices
        assert offsets is not None
        assert indices is not None

        start = int(offsets[row_index].item())
        end = int(offsets[row_index + 1].item())
        candidate_row_indices = indices[start:end]

        if candidate_row_indices.numel() == 0:
            histories.append([])
            continue

        sample_size = min(resource.history_length, int(candidate_row_indices.numel()))
        permutation = torch.randperm(candidate_row_indices.numel())[:sample_size]
        sampled = candidate_row_indices[permutation]
        sampled_times = resource.task_frame_time_values[sampled]
        order = torch.argsort(sampled_times)
        ordered_indices = sampled[order].tolist()

        history: list[dict[str, Any]] = []
        for history_row_index in ordered_indices:
            record = resource.frame.iloc[history_row_index].to_dict()
            history.append(
                {
                    "x": record[resource.entity_col],
                    "timestamp": record[resource.time_col],
                    "output": record[resource.output_col],
                    "train_item": record,
                }
            )
        histories.append(history)

    return histories


def build_cached_dataset_history_grouped(
    batch: dict[str, list[Any]],
    task_resources: dict[tuple[str, str], TaskResource],
) -> list[list[dict[str, Any]]]:
    grouped_positions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for position, (dataset_name, task_name, row_index) in enumerate(
        zip(batch["dataset"], batch["task"], batch["row_index"])
    ):
        grouped_positions.setdefault((dataset_name, task_name), []).append((position, row_index))

    histories: list[list[dict[str, Any]] | None] = [None] * len(batch["dataset"])
    for key, position_pairs in grouped_positions.items():
        resource = task_resources[key]
        offsets = resource.history_candidate_offsets
        indices = resource.history_candidate_indices
        assert offsets is not None
        assert indices is not None

        row_indices = torch.tensor([row_index for _, row_index in position_pairs], dtype=torch.long)
        starts = offsets[row_indices]
        ends = offsets[row_indices + 1]

        for local_idx, (position, row_index) in enumerate(position_pairs):
            start = int(starts[local_idx].item())
            end = int(ends[local_idx].item())
            candidate_row_indices = indices[start:end]
            histories[position] = materialize_history_from_candidate_indices(
                resource,
                candidate_row_indices,
            )

    return [history if history is not None else [] for history in histories]


def _history_worker_process_group(
    task_key: tuple[str, str],
    row_indices: list[int],
) -> list[list[dict[str, Any]]]:
    assert _HISTORY_POOL_RESOURCES is not None
    resource = _HISTORY_POOL_RESOURCES[task_key]
    offsets = resource.history_candidate_offsets
    indices = resource.history_candidate_indices
    assert offsets is not None
    assert indices is not None

    histories: list[list[dict[str, Any]]] = []
    for row_index in row_indices:
        start = int(offsets[row_index].item())
        end = int(offsets[row_index + 1].item())
        candidate_row_indices = indices[start:end]
        histories.append(materialize_history_from_candidate_indices(resource, candidate_row_indices))
    return histories


def get_history_pool(
    task_resources: dict[tuple[str, str], TaskResource],
    workers: int,
) -> mp.pool.Pool:
    global _HISTORY_POOL, _HISTORY_POOL_WORKERS, _HISTORY_POOL_RESOURCES
    if (
        _HISTORY_POOL is not None
        and _HISTORY_POOL_WORKERS == workers
        and _HISTORY_POOL_RESOURCES is task_resources
    ):
        return _HISTORY_POOL

    if _HISTORY_POOL is not None:
        _HISTORY_POOL.close()
        _HISTORY_POOL.join()

    _HISTORY_POOL_RESOURCES = task_resources
    _HISTORY_POOL_WORKERS = workers
    ctx = mp.get_context("fork")
    _HISTORY_POOL = ctx.Pool(processes=workers)
    return _HISTORY_POOL


def build_cached_dataset_history_multiprocess(
    batch: dict[str, list[Any]],
    task_resources: dict[tuple[str, str], TaskResource],
) -> list[list[dict[str, Any]]]:
    first_resource = task_resources[(batch["dataset"][0], batch["task"][0])]
    workers = max(1, int(first_resource.history_parallel_workers))
    pool = get_history_pool(task_resources, workers)

    grouped_positions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for position, (dataset_name, task_name, row_index) in enumerate(
        zip(batch["dataset"], batch["task"], batch["row_index"])
    ):
        grouped_positions.setdefault((dataset_name, task_name), []).append((position, row_index))

    async_jobs: list[tuple[list[int], Any]] = []
    for key, position_pairs in grouped_positions.items():
        positions = [position for position, _ in position_pairs]
        row_indices = [row_index for _, row_index in position_pairs]
        job = pool.apply_async(_history_worker_process_group, (key, row_indices))
        async_jobs.append((positions, job))

    histories: list[list[dict[str, Any]] | None] = [None] * len(batch["dataset"])
    for positions, job in async_jobs:
        group_histories = job.get()
        for position, history in zip(positions, group_histories):
            histories[position] = history

    return [history if history is not None else [] for history in histories]


def materialize_history_from_candidate_indices(
    resource: TaskResource,
    candidate_row_indices: torch.Tensor,
) -> list[dict[str, Any]]:
    if candidate_row_indices.numel() == 0:
        return []

    sample_size = min(resource.history_length, int(candidate_row_indices.numel()))
    permutation = torch.randperm(candidate_row_indices.numel())[:sample_size]
    sampled = candidate_row_indices[permutation]
    sampled_times = resource.task_frame_time_values[sampled]
    order = torch.argsort(sampled_times)
    ordered_indices = sampled[order].tolist()

    history: list[dict[str, Any]] = []
    for history_row_index in ordered_indices:
        record = resource.frame.iloc[history_row_index].to_dict()
        history.append(
            {
                "x": record[resource.entity_col],
                "timestamp": record[resource.time_col],
                "output": record[resource.output_col],
                "train_item": record,
            }
        )
    return history


def construct_batch_history(
    batch: dict[str, list[Any]],
    task_resources: dict[tuple[str, str], TaskResource],
) -> list[list[dict[str, Any]]]:
    if not batch["dataset"]:
        return []

    first_resource = task_resources[(batch["dataset"][0], batch["task"][0])]
    history_source = first_resource.history_source
    if history_source == "task_table":
        return build_task_table_history_for_batch(batch, task_resources)
    if history_source == "dataset":
        return build_dataset_history_for_batch(batch, task_resources)
    raise ValueError(f"Unsupported history source: {history_source}")


def resolve_dataset_history_labels(
    resource: TaskResource,
    timestamp_values: set[int],
) -> dict[tuple[Any, int], dict[str, Any]]:
    label_lookup: dict[tuple[Any, int], dict[str, Any]] = {}
    missing_timestamps: list[int] = []

    for timestamp_value in sorted(timestamp_values):
        cached_rows = resource.dataset_history_label_cache.get(timestamp_value)
        if cached_rows is None:
            missing_timestamps.append(timestamp_value)
            continue
        for entity_value, record in cached_rows.items():
            label_lookup[(entity_value, timestamp_value)] = record

    if missing_timestamps:
        timestamps = pd.to_datetime(missing_timestamps)
        label_table = resource.task_object.make_table(resource.db, pd.Series(timestamps))
        label_df = label_table.df

        fresh_cache: dict[int, dict[Any, dict[str, Any]]] = {
            timestamp_value: {} for timestamp_value in missing_timestamps
        }
        for record in label_df.to_dict(orient="records"):
            timestamp_value = int(pd.Timestamp(record[resource.time_col]).value)
            entity_value = record[resource.entity_col]
            fresh_cache[timestamp_value][entity_value] = record
            label_lookup[(entity_value, timestamp_value)] = record

        if resource.cache_dataset_history_labels:
            for timestamp_value, rows in fresh_cache.items():
                resource.dataset_history_label_cache[timestamp_value] = rows

    return label_lookup


def build_train_loader_with_resources(
    config: dict[str, Any],
    no_dataset_download: bool = False,
    no_task_download: bool = False,
    num_workers: int = 0,
) -> tuple[DataLoader, dict[tuple[str, str], TaskResource], list[MixedTaskExampleRef]]:
    batch_size = config.get("batch_size")
    if batch_size is None:
        raise ValueError("TRAIN_CONFIG must define 'batch_size'.")

    examples, task_resources = build_examples_and_resources(
        config,
        no_dataset_download,
        no_task_download,
    )
    loader = DataLoader(
        MixedTaskTrainDataset(examples, task_resources),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_mixed_batch,
    )
    return loader, task_resources, examples


def preprocess_and_save_train_data(
    config: dict[str, Any],
    no_dataset_download: bool = False,
    no_task_download: bool = False,
) -> Path:
    config_name = config.get("name", "unnamed")
    output_dir = preprocessed_train_dir(config_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples, task_resources = build_examples_and_resources(
        config,
        no_dataset_download,
        no_task_download,
    )

    manifest: dict[str, Any] = {
        "config_name": config_name,
        "batch_size": config["batch_size"],
        "history_length": config["history_length"],
        "history_sampling_strategy": config["history_sampling_strategy"],
        "history_source": config["history_source"],
        "history_parallel_mode": config["history_parallel_mode"],
        "history_parallel_workers": config["history_parallel_workers"],
        "tasks": [],
    }

    for (dataset_name, task_name), resource in task_resources.items():
        task_id = f"{dataset_name}__{task_name}"
        frame_path = output_dir / f"{task_id}__train.parquet"
        history_path = output_dir / f"{task_id}__history_candidates.pt"

        resource.frame.to_parquet(frame_path, index=False)
        if (
            resource.history_source == "dataset"
            and resource.history_candidate_offsets is not None
            and resource.history_candidate_indices is not None
        ):
            torch.save(
                {
                    "offsets": resource.history_candidate_offsets,
                    "indices": resource.history_candidate_indices,
                },
                history_path,
            )

        manifest["tasks"].append(
            {
                "dataset": dataset_name,
                "task": task_name,
                "split": resource.split,
                "frame_path": str(frame_path),
                "history_path": str(history_path) if history_path.exists() else None,
                "time_col": resource.time_col,
                "output_col": resource.output_col,
                "entity_col": resource.entity_col,
                "num_rows": len(resource.frame),
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return output_dir


def load_preprocessed_train_data(
    config: dict[str, Any],
    num_workers: int = 0,
) -> tuple[DataLoader, dict[tuple[str, str], TaskResource], list[MixedTaskExampleRef]]:
    config_name = config.get("name", "unnamed")
    input_dir = preprocessed_train_dir(config_name)
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Preprocessed train data not found at {manifest_path}. "
            "Run preprocess_train.py first."
        )

    manifest = json.loads(manifest_path.read_text())
    task_resources: dict[tuple[str, str], TaskResource] = {}
    examples: list[MixedTaskExampleRef] = []

    for task_meta in manifest["tasks"]:
        dataset_name = task_meta["dataset"]
        task_name = task_meta["task"]
        frame = pd.read_parquet(task_meta["frame_path"])
        frame = frame.reset_index(drop=True)
        time_values = torch.as_tensor(frame[task_meta["time_col"]].astype("int64").to_numpy(), dtype=torch.long)

        grouped = frame.groupby(task_meta["entity_col"], sort=False).indices
        entity_to_row_indices: dict[Any, torch.Tensor] = {}
        entity_to_time_values: dict[Any, torch.Tensor] = {}
        for entity_value, row_indices in grouped.items():
            indices = torch.as_tensor(row_indices, dtype=torch.long)
            entity_to_row_indices[entity_value] = indices
            entity_to_time_values[entity_value] = time_values[indices]

        offsets = None
        indices = None
        history_path = task_meta.get("history_path")
        if history_path:
            cached = torch.load(history_path, map_location="cpu")
            offsets = cached["offsets"].to(torch.long)
            indices = cached["indices"].to(torch.long)
        elif config["history_source"] == "dataset":
            raise FileNotFoundError(
                f"Missing preprocessed history candidate file for "
                f"{dataset_name}/{task_name}. Run preprocess_train.py again."
            )

        resource = TaskResource(
            dataset=dataset_name,
            task=task_name,
            split=task_meta["split"],
            frame=frame,
            time_col=task_meta["time_col"],
            output_col=task_meta["output_col"],
            entity_col=task_meta["entity_col"],
            entity_table="",
            history_length=config["history_length"],
            history_sampling_strategy=config["history_sampling_strategy"],
            history_source=config["history_source"],
            history_parallel_mode=config["history_parallel_mode"],
            history_parallel_workers=config["history_parallel_workers"],
            cache_dataset_history_labels=config["cache_dataset_history_labels"],
            cache_train_history_candidates=config["cache_train_history_candidates"],
            task_object=None,
            db=None,  # type: ignore[arg-type]
            task_frame_time_values=time_values,
            task_schedule_timestamps=torch.unique_consecutive(time_values),
            label_horizon_ns=0,
            entity_to_row_indices=entity_to_row_indices,
            entity_to_time_values=entity_to_time_values,
            history_candidate_offsets=offsets,
            history_candidate_indices=indices,
        )
        task_resources[(dataset_name, task_name)] = resource

        for row_index in range(len(frame)):
            examples.append(
                MixedTaskExampleRef(
                    dataset=dataset_name,
                    task=task_name,
                    split=task_meta["split"],
                    row_index=row_index,
                )
            )

    loader = DataLoader(
        MixedTaskTrainDataset(examples, task_resources),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_mixed_batch,
    )
    return loader, task_resources, examples


def summarize_examples(examples: list[MixedTaskExampleRef]) -> None:
    print(f"num_examples={len(examples)}")
    counts: dict[tuple[str, str], int] = {}
    for example in examples:
        key = (example.dataset, example.task)
        counts[key] = counts.get(key, 0) + 1

    for (dataset_name, task_name), count in sorted(counts.items()):
        print(f"  {dataset_name} / {task_name}: {count}")
