from __future__ import annotations

import argparse
from pathlib import Path

from load_train_config import DEFAULT_CONFIG_PATH, load_config, validate_config
from train_data import (
    build_train_loader_with_resources,
    construct_batch_history,
    summarize_examples,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a mixed-task train batch loader from a train config."
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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of PyTorch DataLoader workers.",
    )
    parser.add_argument(
        "--preview-batches",
        type=int,
        default=1,
        help="Number of batches to print for inspection.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config.resolve())
    validate_config(config)

    loader, task_resources, examples = build_train_loader_with_resources(
        config,
        no_dataset_download=args.no_dataset_download,
        no_task_download=args.no_task_download,
        num_workers=args.num_workers,
    )
    summarize_examples(examples)

    for batch_index, batch in enumerate(loader):
        if batch_index >= args.preview_batches:
            break
        batch_history = construct_batch_history(batch, task_resources)
        print(f"batch_index={batch_index}")
        print(f"  batch_size={len(batch['dataset'])}")
        print(f"  datasets={batch['dataset']}")
        print(f"  tasks={batch['task']}")
        print(f"  row_indices={batch['row_index']}")
        print(f"  timestamps={batch['timestamp']}")
        print(f"  outputs={batch['output']}")
        print(f"  first_train_item={batch['train_item'][0]}")
        print(f"  first_history={batch_history[0]}")
        # breakpoint()


if __name__ == "__main__":
    main()
