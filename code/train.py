from __future__ import annotations

import argparse
from pathlib import Path
import time

from load_train_config import DEFAULT_CONFIG_PATH, load_config, validate_config
from train_data import (
    construct_batch_history,
    load_preprocessed_train_data,
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to iterate over the train loader.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config.resolve())
    validate_config(config)

    loader, task_resources, examples = load_preprocessed_train_data(
        config,
        num_workers=args.num_workers,
    )
    summarize_examples(examples)

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        num_batches = 0

        for batch_index, batch in enumerate(loader):
            batch_history = construct_batch_history(batch, task_resources)
            # if epoch == 0 and batch_index < args.preview_batches:
            #     print(f"epoch={epoch + 1} batch_index={batch_index}")
            #     print(f"  first_train_item={batch['train_item'][0]}")
            #     print(f"  first_history={len(batch_history[0])}")
            
            print(f"epoch={epoch + 1} batch_index={batch_index}")
            print(f"  first_train_item={batch['train_item'][0]}")
            print(f"  first_history={len(batch_history[0])}")
            num_batches += 1
            if batch_index >= 10:
                break

        epoch_time = time.perf_counter() - epoch_start
        print(f"epoch={epoch + 1} num_batches={num_batches} time_sec={epoch_time:.2f}")


if __name__ == "__main__":
    main()
