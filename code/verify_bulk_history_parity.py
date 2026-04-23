from __future__ import annotations

import argparse
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from inference_history import (
    build_history_store_with_options,
    build_inference_histories_bulk,
    build_inference_history,
    build_inference_resource,
)
from load_inference_config import build_parser, load_and_validate_inference_config


def extend_parser() -> argparse.ArgumentParser:
    parser = build_parser(
        "Verify parity between single-request and bulk history construction."
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
        "--dataset",
        type=str,
        default=None,
        help="Override the single dataset defined in the inference config.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Override the single task defined in the inference config.",
    )
    parser.add_argument(
        "--history-sampling-strategy",
        choices=["recent_min_overlap", "most_recent_k", "random_prior"],
        default="most_recent_k",
        help="Override the inference history sampler.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode and cap processing to --max-items examples.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=200,
        help="Maximum eval rows to check when --debug is enabled.",
    )
    parser.add_argument(
        "--force-history-cache-rebuild",
        action="store_true",
        help="Rebuild the zero-shot dataset-history disk cache before verification.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first mismatch.",
    )
    return parser


def _normalize_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return int(value.value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and np.isnan(value):
        return "NaN"
    return value


def _normalize_history_item(item: dict[str, Any], time_col: str) -> tuple:
    normalized_pairs = []
    for key in sorted(item.keys()):
        value = item[key]
        if key == time_col:
            value = pd.Timestamp(value)
        normalized_pairs.append((key, _normalize_value(value)))
    return tuple(normalized_pairs)


def main() -> None:
    args = extend_parser().parse_args()
    config = load_and_validate_inference_config(args.config.resolve())
    if args.dataset is not None:
        config["datasets"][0]["name"] = args.dataset
    if args.task is not None:
        config["datasets"][0]["tasks"] = [args.task]
    if args.history_sampling_strategy is not None:
        config["history_sampling_strategy"] = args.history_sampling_strategy

    resource = build_inference_resource(
        config,
        no_dataset_download=args.no_dataset_download,
        no_task_download=args.no_task_download,
    )
    store, _ = build_history_store_with_options(
        resource,
        force_rebuild=args.force_history_cache_rebuild,
        verbose=False,
    )

    eval_table = resource.task_object.get_table("test", mask_input_cols=False)
    if resource.output_col not in eval_table.df.columns:
        eval_table = resource.task_object.get_table("val", mask_input_cols=False)

    eval_rows = list(eval_table.df.itertuples(index=False))
    if args.debug:
        eval_rows = eval_rows[: args.max_items]

    requests: list[tuple[int, Any, Any]] = []
    for idx, row in enumerate(eval_rows):
        record = row._asdict()
        requests.append((idx, record[resource.entity_col], record[resource.time_col]))

    t0 = perf_counter()
    single_histories: dict[int, list[dict[str, Any]]] = {}
    for request_id, entity_value, cutoff_time in requests:
        single_histories[request_id] = build_inference_history(
            store,
            resource,
            entity_value,
            cutoff_time,
        )
    single_s = perf_counter() - t0

    t0 = perf_counter()
    bulk_histories = build_inference_histories_bulk(store, resource, requests)
    bulk_s = perf_counter() - t0

    mismatches = 0
    for request_id, entity_value, cutoff_time in requests:
        single_items = single_histories.get(request_id, [])
        bulk_items = bulk_histories.get(request_id, [])
        single_norm = [
            _normalize_history_item(item, resource.time_col)
            for item in single_items
        ]
        bulk_norm = [
            _normalize_history_item(item, resource.time_col)
            for item in bulk_items
        ]

        if single_norm != bulk_norm:
            mismatches += 1
            print("\nMismatch detected")
            print(f"request_id={request_id}")
            print(f"entity={entity_value}")
            print(f"cutoff_time={cutoff_time}")
            print(f"single_len={len(single_items)} bulk_len={len(bulk_items)}")
            first_diff_idx = None
            for idx in range(min(len(single_norm), len(bulk_norm))):
                if single_norm[idx] != bulk_norm[idx]:
                    first_diff_idx = idx
                    break
            if first_diff_idx is not None:
                print(f"first_diff_index={first_diff_idx}")
                print(f"single_item={single_items[first_diff_idx]}")
                print(f"bulk_item={bulk_items[first_diff_idx]}")
            else:
                print("history lengths differ but common prefix is identical")
            if args.fail_fast:
                break

    checked = len(requests) if not args.fail_fast else (
        next((i + 1 for i, req in enumerate(requests)
              if single_histories.get(req[0], []) != bulk_histories.get(req[0], [])),
             len(requests))
    )
    print("\nParity summary")
    print(f"checked_requests={checked}")
    print(f"mismatches={mismatches}")
    print(f"single_total_s={single_s:.4f}")
    print(f"bulk_total_s={bulk_s:.4f}")
    if bulk_s > 0:
        print(f"speedup_single_over_bulk={single_s / bulk_s:.3f}x")

    if mismatches > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
