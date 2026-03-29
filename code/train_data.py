from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from relbench.base import Database
from relbench.tasks import get_task, get_task_names

from data_config import RelBenchDuckDBConfig
from data_pipeline import RelBenchDuckDBPipeline


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
    task_object: Any
    db: Database
    task_frame_time_values: torch.Tensor
    label_horizon_ns: int
    entity_to_row_indices: dict[Any, torch.Tensor]
    entity_to_time_values: dict[Any, torch.Tensor]
    entity_to_dataset_times: dict[Any, torch.Tensor]


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

    entity_to_dataset_times = build_entity_dataset_time_index(
        db=db,
        entity_col=entity_col,
        entity_table=entity_table,
    )

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
        task_object=task,
        db=db,
        task_frame_time_values=time_values,
        label_horizon_ns=int(task.timedelta.value),
        entity_to_row_indices=entity_to_row_indices,
        entity_to_time_values=entity_to_time_values,
        entity_to_dataset_times=entity_to_dataset_times,
    )


def build_entity_dataset_time_index(
    db: Database,
    entity_col: str,
    entity_table: str,
) -> dict[Any, torch.Tensor]:
    entity_to_times: dict[Any, list[int]] = {}

    for table_name, table in db.table_dict.items():
        if table.time_col is None:
            continue

        candidate_cols: list[str] = []
        if table_name == entity_table and table.pkey_col == entity_col:
            candidate_cols.append(entity_col)

        for fkey_col, pkey_table_name in table.fkey_col_to_pkey_table.items():
            if pkey_table_name == entity_table:
                candidate_cols.append(fkey_col)

        if not candidate_cols:
            continue

        time_ns = table.df[table.time_col].astype("int64")
        for candidate_col in candidate_cols:
            pairs = pd.DataFrame(
                {
                    "entity": table.df[candidate_col],
                    "time_ns": time_ns,
                }
            ).dropna(subset=["entity", "time_ns"])

            grouped = pairs.groupby("entity", sort=False)["time_ns"]
            for entity_value, values in grouped:
                entity_to_times.setdefault(entity_value, []).extend(values.tolist())

    output: dict[Any, torch.Tensor] = {}
    for entity_value, values in entity_to_times.items():
        unique_sorted = sorted(set(int(v) for v in values))
        output[entity_value] = torch.tensor(unique_sorted, dtype=torch.long)
    return output


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


def sample_dataset_history_timestamps_for_example(
    resource: TaskResource,
    row_index: int,
) -> torch.Tensor:
    if resource.history_length <= 0:
        return torch.empty(0, dtype=torch.long)
    if resource.history_sampling_strategy != "random_prior":
        raise ValueError(
            f"Unsupported history sampling strategy: {resource.history_sampling_strategy}"
        )

    row = resource.frame.iloc[row_index]
    entity_value = row[resource.entity_col]
    candidate_times = resource.entity_to_dataset_times.get(entity_value)
    if candidate_times is None or candidate_times.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    current_time_value = torch.tensor(
        int(pd.Timestamp(row[resource.time_col]).value),
        dtype=torch.long,
    )
    effective_cutoff = current_time_value - resource.label_horizon_ns
    cutoff = int(torch.searchsorted(candidate_times, effective_cutoff, right=True).item())
    prior_times = candidate_times[:cutoff]
    if prior_times.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    sample_size = min(resource.history_length, int(prior_times.numel()))
    permutation = torch.randperm(prior_times.numel())[:sample_size]
    chosen = prior_times[permutation]
    order = torch.argsort(chosen)
    return chosen[order]


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
    sampled_times_per_example: list[tuple[TaskResource, Any, torch.Tensor]] = []
    grouped_timestamps: dict[tuple[str, str], set[int]] = {}

    for dataset_name, task_name, row_index in zip(
        batch["dataset"],
        batch["task"],
        batch["row_index"],
    ):
        resource = task_resources[(dataset_name, task_name)]
        row = resource.frame.iloc[row_index]
        entity_value = row[resource.entity_col]
        sampled_times = sample_dataset_history_timestamps_for_example(resource, row_index)
        sampled_times_per_example.append((resource, entity_value, sampled_times))
        grouped_timestamps.setdefault((dataset_name, task_name), set()).update(
            int(x) for x in sampled_times.tolist()
        )

    label_lookup_per_task: dict[tuple[str, str], dict[tuple[Any, int], dict[str, Any]]] = {}
    for key, timestamp_values in grouped_timestamps.items():
        resource = task_resources[key]
        if not timestamp_values:
            label_lookup_per_task[key] = {}
            continue

        timestamps = pd.to_datetime(sorted(timestamp_values))
        label_table = resource.task_object.make_table(resource.db, pd.Series(timestamps))
        label_df = label_table.df
        label_lookup: dict[tuple[Any, int], dict[str, Any]] = {}
        for record in label_df.to_dict(orient="records"):
            label_lookup[
                (
                    record[resource.entity_col],
                    int(pd.Timestamp(record[resource.time_col]).value),
                )
            ] = record
        label_lookup_per_task[key] = label_lookup

    histories: list[list[dict[str, Any]]] = []
    for resource, entity_value, sampled_times in sampled_times_per_example:
        label_lookup = label_lookup_per_task[(resource.dataset, resource.task)]
        history: list[dict[str, Any]] = []
        for time_value in sampled_times.tolist():
            record = label_lookup.get((entity_value, int(time_value)))
            if record is None:
                continue
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


def summarize_examples(examples: list[MixedTaskExampleRef]) -> None:
    print(f"num_examples={len(examples)}")
    counts: dict[tuple[str, str], int] = {}
    for example in examples:
        key = (example.dataset, example.task)
        counts[key] = counts.get(key, 0) + 1

    for (dataset_name, task_name), count in sorted(counts.items()):
        print(f"  {dataset_name} / {task_name}: {count}")
