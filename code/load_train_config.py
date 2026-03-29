from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from relbench.tasks import get_task, get_task_names

from data_config import RelBenchDuckDBConfig
from data_pipeline import RelBenchDuckDBPipeline


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "train_config.py"


def load_config(config_path: Path) -> dict:
    spec = importlib.util.spec_from_file_location("rfm_train_config", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load config file: {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "TRAIN_CONFIG"):
        raise ValueError(f"Config file {config_path} must define TRAIN_CONFIG.")

    config = module.TRAIN_CONFIG
    if not isinstance(config, dict):
        raise ValueError("TRAIN_CONFIG must be a dictionary.")
    return config


def validate_config(config: dict) -> None:
    if "batch_size" not in config:
        raise ValueError("TRAIN_CONFIG must contain 'batch_size'.")
    if "history_length" not in config:
        raise ValueError("TRAIN_CONFIG must contain 'history_length'.")
    if "history_source" not in config:
        raise ValueError("TRAIN_CONFIG must contain 'history_source'.")
    if config["history_source"] not in {"task_table", "dataset"}:
        raise ValueError("TRAIN_CONFIG['history_source'] must be 'task_table' or 'dataset'.")
    if "datasets" not in config or not isinstance(config["datasets"], list):
        raise ValueError("TRAIN_CONFIG must contain a 'datasets' list.")

    for dataset_cfg in config["datasets"]:
        if "name" not in dataset_cfg:
            raise ValueError("Each dataset config must contain a 'name'.")
        if "tasks" not in dataset_cfg or not isinstance(dataset_cfg["tasks"], list):
            raise ValueError(
                f"Dataset config for {dataset_cfg['name']} must contain a 'tasks' list."
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load datasets and tasks specified in a train config file."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to a Python config file. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--no-dataset-download",
        action="store_true",
        help="Use existing dataset cache only and skip RelBench verification/download.",
    )
    parser.add_argument(
        "--no-task-download",
        action="store_true",
        help="Use existing task cache only and skip RelBench verification/download.",
    )
    return parser


def load_dataset(dataset_name: str, config: dict, no_dataset_download: bool) -> None:
    pipeline = RelBenchDuckDBPipeline(
        RelBenchDuckDBConfig(
            dataset_name=dataset_name,
            download=(not no_dataset_download)
            and config.get("dataset_download", True),
            upto_test_timestamp=not config.get("include_future_rows", False),
        )
    )
    db = pipeline.materialize()
    print(f"dataset={dataset_name}")
    print(f"  relbench_cache={pipeline.config.dataset_cache_dir}")
    print(f"  duckdb_path={pipeline.config.duckdb_path}")
    print(f"  num_tables={len(db.table_dict)}")


def load_tasks(dataset_name: str, task_names: list[str], config: dict, no_task_download: bool) -> None:
    available_tasks = set(get_task_names(dataset_name))
    for task_name in task_names:
        if task_name not in available_tasks:
            raise ValueError(
                f"Unknown task '{task_name}' for dataset '{dataset_name}'. "
                f"Available tasks: {sorted(available_tasks)}"
            )

        task = get_task(
            dataset_name,
            task_name,
            download=(not no_task_download) and config.get("task_download", True),
        )
        train_table = task.get_table("train")
        val_table = task.get_table("val")
        test_table = task.get_table("test")

        print(f"  task={task_name}")
        print(f"    train_rows={len(train_table.df)}")
        print(f"    val_rows={len(val_table.df)}")
        print(f"    test_rows={len(test_table.df)}")
        print(f"    columns={list(train_table.df.columns)}")


def main() -> None:
    args = build_parser().parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    validate_config(config)

    print(f"config={config_path}")
    print(f"name={config.get('name', 'unnamed')}")

    for dataset_cfg in config["datasets"]:
        dataset_name = dataset_cfg["name"]
        load_dataset(dataset_name, config, args.no_dataset_download)
        load_tasks(dataset_name, dataset_cfg["tasks"], config, args.no_task_download)


if __name__ == "__main__":
    main()
