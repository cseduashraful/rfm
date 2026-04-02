from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from load_train_config import validate_config


DEFAULT_INFERENCE_CONFIG_PATH = (
    Path(__file__).resolve().parent / "configs" / "inference_config.py"
)


def load_inference_config(config_path: Path) -> dict:
    spec = importlib.util.spec_from_file_location("rfm_inference_config", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load config file: {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "INFERENCE_CONFIG"):
        raise ValueError(f"Config file {config_path} must define INFERENCE_CONFIG.")

    config = module.INFERENCE_CONFIG
    if not isinstance(config, dict):
        raise ValueError("INFERENCE_CONFIG must be a dictionary.")
    return config


def normalize_inference_config(config: dict) -> dict:
    normalized = dict(config)

    dataset_name = normalized.pop("dataset", None)
    task_name = normalized.pop("task", None)
    if dataset_name is None or task_name is None:
        raise ValueError(
            "INFERENCE_CONFIG must contain single 'dataset' and 'task' entries."
        )

    normalized["datasets"] = [{"name": dataset_name, "tasks": [task_name]}]
    return normalized


def load_and_validate_inference_config(config_path: Path) -> dict:
    config = normalize_inference_config(load_inference_config(config_path))
    validate_config(config)
    return config


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_INFERENCE_CONFIG_PATH,
        help=(
            "Path to a Python inference config file. "
            f"Default: {DEFAULT_INFERENCE_CONFIG_PATH}"
        ),
    )
    return parser
