from __future__ import annotations

import argparse
from pathlib import Path

from load_train_config import DEFAULT_CONFIG_PATH, load_config, validate_config
from train_data import preprocess_and_save_train_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess and save train split data and history candidates."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to a Python train config file. Default: {DEFAULT_CONFIG_PATH}",
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


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config.resolve())
    validate_config(config)
    output_dir = preprocess_and_save_train_data(
        config,
        no_dataset_download=args.no_dataset_download,
        no_task_download=args.no_task_download,
    )
    print(f"preprocessed_train_dir={output_dir}")


if __name__ == "__main__":
    main()
