from __future__ import annotations

import argparse
import cProfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Lock
import threading
import pstats
from time import perf_counter
from typing import Any
import warnings

import numpy as np
import pandas as pd
import torch

from load_inference_config import build_parser, load_and_validate_inference_config
from fastdfs_context import build_fastdfs_context_builder
from grag import RelBenchGraphRAGStore, RowFeatureExtractor, build_zero_shot_prompt
from inference_history import (
    build_inference_histories_bulk,
    build_history_store_with_options,
    build_inference_history,
    build_inference_resource,
    validate_history_non_overlap,
)
from zero_shot_llm import LocalLLM, MODEL_PATHS, extract_number


warnings.filterwarnings(
    "ignore",
    message="Could not infer format.*",
    category=UserWarning,
    module=r"woodwork\.type_sys\.utils",
)
warnings.filterwarnings(
    "ignore",
    message="Logical type .* does not match .*",
    category=UserWarning,
    module=r"featuretools\.entityset\.entityset",
)


@dataclass
class PhaseTimer:
    enabled: bool = True
    _phase_durations: dict[str, list[float]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def add(self, phase: str, duration_s: float) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._phase_durations.setdefault(phase, []).append(duration_s)

    def summary_rows(self) -> list[dict[str, Any]]:
        rows = []
        for phase, durations in self._phase_durations.items():
            if not durations:
                continue
            arr = np.array(durations, dtype=float)
            rows.append(
                {
                    "phase": phase,
                    "count": int(arr.size),
                    "total_s": float(arr.sum()),
                    "avg_s": float(arr.mean()),
                    "p50_s": float(np.percentile(arr, 50)),
                    "p95_s": float(np.percentile(arr, 95)),
                    "max_s": float(arr.max()),
                }
            )
        rows.sort(key=lambda row: row["total_s"], reverse=True)
        return rows

    def total_for_phase(self, phase: str) -> float:
        with self._lock:
            durations = self._phase_durations.get(phase, [])
            return float(sum(durations))

    def total_for_prefix(self, prefix: str) -> float:
        with self._lock:
            total = 0.0
            for phase, durations in self._phase_durations.items():
                if phase.startswith(prefix):
                    total += float(sum(durations))
            return total


def extend_parser() -> argparse.ArgumentParser:
    parser = build_parser(
        "Build zero-shot inference histories from a single-task inference config."
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
        "--max-items",
        type=int,
        default=50,
        help="Maximum number of inference items to process when --debug is enabled.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode and cap processing to --max-items examples.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Maximum number of neighbors per hop to include in prompt construction.",
    )
    parser.add_argument(
        "--num-hops",
        type=int,
        default=1,
        help="Number of graph hops to retrieve for each history item.",
    )
    parser.add_argument(
        "--recent-context-k",
        type=int,
        default=2,
        help=(
            "Show DFS and neighborhood context only for the most recent k prompt entries "
            "(history + query)."
        ),
    )
    parser.add_argument(
        "--context-components",
        type=str,
        default="neighbors,dfs,dfs_table",
        help=(
            "Comma-separated prompt context components. Supported: neighbors, dfs, dfs_table, none. "
            "Example: --context-components neighbors,dfs"
        ),
    )
    parser.add_argument(
        "--context-workers",
        type=int,
        default=1,
        help="Number of workers to use when preparing prompts across examples and within each prompt context.",
    )
    parser.add_argument(
        "--frontier-workers",
        type=int,
        default=1,
        help="Number of workers to use when expanding frontier nodes within a hop.",
    )
    parser.add_argument(
        "--context-batch-size",
        type=int,
        default=8,
        help="Number of examples to prepare prompts for before LLM generation.",
    )
    parser.add_argument(
        "--llm-batch-size",
        type=int,
        default=1,
        help="Number of prompts to send to the LLM at once.",
    )
    parser.add_argument(
        "--dfs-batch-size",
        type=int,
        default=64,
        help="When --use-dfs is enabled, number of examples to batch for DFS precompute before prompt building.",
    )
    parser.add_argument(
        "--pipeline-batch-size",
        type=int,
        default=None,
        help=(
            "Prompt/LLM pipeline micro-batch size. If unset, defaults to "
            "--context-batch-size (or legacy behavior when DFS is disabled)."
        ),
    )
    parser.add_argument(
        "--bulk-history-query",
        dest="bulk_history_query",
        action="store_true",
        default=True,
        help="Use bulk history construction for prepare batches when available.",
    )
    parser.add_argument(
        "--no-bulk-history-query",
        dest="bulk_history_query",
        action="store_false",
        help="Disable bulk history construction and use per-request history lookup.",
    )
    parser.add_argument(
        "--overlap-prep-llm",
        dest="overlap_prep_llm",
        action="store_true",
        default=True,
        help="Overlap CPU prompt preparation for next batch with current LLM generation.",
    )
    parser.add_argument(
        "--no-overlap-prep-llm",
        dest="overlap_prep_llm",
        action="store_false",
        help="Disable CPU/GPU overlap and run batches serially.",
    )
    parser.add_argument(
        "--prep-queue-size",
        type=int,
        default=4,
        help="Number of prepared prompt batches to buffer ahead in overlap mode.",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Enable task-agnostic neighbor aggregation summaries in the prompt.",
    )
    parser.add_argument(
        "--semantic-retrieval",
        action="store_true",
        help="Also retrieve generic same-table similar-entity context beyond FK/PK neighbors.",
    )
    parser.add_argument(
        "--use-dfs",
        "--use-fastdfs",
        dest="use_dfs",
        action="store_true",
        help="Enable FastDFS feature extraction and prompt summaries for each example/query timestamp.",
    )
    parser.add_argument(
        "--model-size",
        choices=sorted(MODEL_PATHS),
        default="1b",
        help="Local LLaMA size to use for zero-shot generation.",
    )
    parser.add_argument(
        "--history-sampling-strategy",
        choices=["recent_min_overlap", "most_recent_k", "random_prior"],
        default="most_recent_k",
        help="Override the inference history sampler.",
    )
    parser.add_argument(
        "--self-example-max-count",
        type=int,
        default=None,
        help=(
            "Maximum number of same-entity (self) history examples to keep per query. "
            "If omitted, keeps all available self examples up to history_length."
        ),
    )
    parser.add_argument(
        "--other-example-max-count",
        type=int,
        default=None,
        help=(
            "Maximum number of other-entity history examples to include per query. "
            "If omitted, uses legacy backfill behavior up to history_length."
        ),
    )
    parser.add_argument(
        "--other-neighbor-entity-count",
        type=int,
        default=5,
        help=(
            "n: number of nearby same-type neighbor entities (from PK-FK graph) "
            "to use for other examples."
        ),
    )
    parser.add_argument(
        "--other-neighbor-history-count",
        type=int,
        default=3,
        help=(
            "h: number of most recent history rows per selected nearby neighbor entity."
        ),
    )
    parser.add_argument(
        "--other-neighbor-search-hops",
        type=int,
        default=4,
        help=(
            "Hop depth used specifically for mining nearby same-type neighbor entities "
            "for other examples."
        ),
    )
    parser.add_argument(
        "--print-log",
        action="store_true",
        help="Print detailed zero-shot inspection logs.",
    )
    parser.add_argument(
        "--pred-only",
        action="store_true",
        help="With --print-log, print only prediction and ground truth per item.",
    )
    parser.add_argument(
        "--force-history-cache-rebuild",
        action="store_true",
        help="Rebuild the zero-shot dataset-history disk cache before inference.",
    )
    parser.add_argument(
        "--profile",
        dest="profile",
        action="store_true",
        default=True,
        help="Enable lightweight phase timing and summary output.",
    )
    parser.add_argument(
        "--no-profile",
        dest="profile",
        action="store_false",
        help="Disable lightweight profiling for clean runs.",
    )
    parser.add_argument(
        "--profile-print-items",
        action="store_true",
        help="Print per-item timing details during inference.",
    )
    parser.add_argument(
        "--profile-slowest-items",
        type=int,
        default=10,
        help="Number of slowest items to print in the final profile summary.",
    )
    parser.add_argument(
        "--enable-cprofile",
        action="store_true",
        help="Enable Python cProfile and print top functions at the end.",
    )
    parser.add_argument(
        "--cprofile-sort",
        choices=["cumulative", "tottime", "calls"],
        default="cumulative",
        help="Sort key for cProfile function stats.",
    )
    parser.add_argument(
        "--cprofile-top-n",
        type=int,
        default=40,
        help="How many cProfile rows to print.",
    )
    parser.add_argument(
        "--cprofile-output",
        type=str,
        default=None,
        help="Optional output path to save raw cProfile stats (.prof).",
    )
    parser.add_argument(
        "--enable-torch-profiler",
        action="store_true",
        help="Save PyTorch profiler traces for non-LLM prepare_item work only.",
    )
    parser.add_argument(
        "--torch-profiler-dir",
        type=str,
        default="artifacts/torch_profile",
        help="Directory where prepare-item PyTorch traces will be saved.",
    )
    parser.add_argument(
        "--torch-profiler-max-items",
        type=int,
        default=10,
        help="Maximum number of prepare_item traces to export.",
    )
    parser.add_argument(
        "--torch-profiler-record-shapes",
        action="store_true",
        help="Record tensor shapes in PyTorch profiler traces.",
    )
    parser.add_argument(
        "--torch-profiler-profile-memory",
        action="store_true",
        help="Record memory events in PyTorch profiler traces.",
    )
    parser.add_argument(
        "--torch-profiler-with-stack",
        action="store_true",
        help="Record Python stack traces in PyTorch profiler output.",
    )
    return parser


def _parse_context_components(raw: str) -> set[str]:
    allowed = {"neighbors", "dfs", "dfs_table", "none"}
    selected = {part.strip().lower() for part in raw.split(",") if part.strip()}
    if not selected:
        return {"neighbors", "dfs", "dfs_table"}
    invalid = sorted(selected - allowed)
    if invalid:
        raise ValueError(
            "Invalid --context-components value(s): "
            + ", ".join(invalid)
            + ". Allowed: neighbors, dfs, dfs_table, none."
        )
    if "none" in selected:
        return set()
    return selected


def run(args: argparse.Namespace) -> None:
    if args.self_example_max_count is not None and args.self_example_max_count < 0:
        raise ValueError("--self-example-max-count must be >= 0.")
    if args.other_example_max_count is not None and args.other_example_max_count < 0:
        raise ValueError("--other-example-max-count must be >= 0.")
    if args.other_neighbor_entity_count < 0:
        raise ValueError("--other-neighbor-entity-count must be >= 0.")
    if args.other_neighbor_history_count < 0:
        raise ValueError("--other-neighbor-history-count must be >= 0.")
    if args.other_neighbor_search_hops < 1:
        raise ValueError("--other-neighbor-search-hops must be >= 1.")

    config = load_and_validate_inference_config(args.config.resolve())
    if args.dataset is not None:
        config["datasets"][0]["name"] = args.dataset
    if args.task is not None:
        config["datasets"][0]["tasks"] = [args.task]
    if args.history_sampling_strategy is not None:
        config["history_sampling_strategy"] = args.history_sampling_strategy

    timers = PhaseTimer(enabled=args.profile)
    prepare_item_stats: list[dict[str, Any]] = []
    prepare_item_lock = Lock()
    global_start = perf_counter()
    torch_profile_reservation_lock = Lock()
    torch_profile_saved_lock = Lock()
    torch_profile_saved_paths: list[str] = []
    torch_profile_next_slot = 0

    t0 = perf_counter()
    resource = build_inference_resource(
        config,
        no_dataset_download=args.no_dataset_download,
        no_task_download=args.no_task_download,
    )
    timers.add("setup.build_inference_resource", perf_counter() - t0)

    t0 = perf_counter()
    store, _ = build_history_store_with_options(
        resource,
        force_rebuild=args.force_history_cache_rebuild,
        verbose=args.print_log,
    )
    timers.add("setup.build_history_store", perf_counter() - t0)

    graph_store = RelBenchGraphRAGStore()
    t0 = perf_counter()
    graph_store.build_base(resource.db, resource.task_object)
    timers.add("setup.graph_store.build_base", perf_counter() - t0)
    graph_store.parallel_frontier_workers = max(1, args.frontier_workers)
    t0 = perf_counter()
    extractor = RowFeatureExtractor(resource.db, graph_store)
    timers.add("setup.row_feature_extractor", perf_counter() - t0)
    context_components = _parse_context_components(args.context_components)
    include_neighbors = "neighbors" in context_components
    include_dfs_summary = "dfs" in context_components
    include_dfs_table = "dfs_table" in context_components
    effective_use_dfs = args.use_dfs and (include_dfs_summary or include_dfs_table)
    dfs_context_builder = None
    if effective_use_dfs:
        t0 = perf_counter()
        dfs_context_builder = build_fastdfs_context_builder(resource)
        timers.add("setup.fastdfs_context_builder", perf_counter() - t0)

    t0 = perf_counter()
    llm = LocalLLM(MODEL_PATHS[args.model_size], print_log=args.print_log)
    timers.add("setup.load_llm", perf_counter() - t0)

    t0 = perf_counter()
    eval_table = resource.task_object.get_table("test", mask_input_cols=False)
    timers.add("setup.load_eval_table_test", perf_counter() - t0)
    if resource.output_col not in eval_table.df.columns:
        if args.print_log:
            print("using val for evaluation since test has no labels")
        t0 = perf_counter()
        eval_table = resource.task_object.get_table("val", mask_input_cols=False)
        timers.add("setup.load_eval_table_val", perf_counter() - t0)

    eval_max_time = eval_table.df[resource.time_col].max()
    if resource.db.max_timestamp < eval_max_time:
        raise ValueError(
            "The materialized DB does not include enough future rows for inference. "
            f"db.max_timestamp={resource.db.max_timestamp}, eval_max_timestamp={eval_max_time}. "
            "Set INFERENCE_CONFIG['use_full_raw_db'] = True and rerun without "
            "--no-dataset-download if needed."
        )

    if args.print_log:
        print(f"dataset={resource.dataset}")
        print(f"task={resource.task}")
        print(f"history_source={resource.history_source}")
        print(f"history_sampling_strategy={resource.history_sampling_strategy}")
        print(f"model_size={args.model_size}")
        print(f"eval_rows={len(eval_table.df)}")
        print(f"use_dfs={args.use_dfs}")
        if effective_use_dfs and dfs_context_builder is not None:
            print(f"dfs_training_window={dfs_context_builder.training_window}")
            print(f"dfs_training_window_key={dfs_context_builder.training_window_key}")
        print(f"context_components={','.join(sorted(context_components)) or 'none'}")
        print(f"recent_context_k={args.recent_context_k}")
        print(f"profile={args.profile}")
        print(f"enable_cprofile={args.enable_cprofile}")
        print(f"enable_torch_profiler={args.enable_torch_profiler}")
        print(f"overlap_prep_llm={args.overlap_prep_llm}")
        print(f"bulk_history_query={args.bulk_history_query}")
        print(f"pipeline_batch_size={args.pipeline_batch_size}")
        print(f"self_example_max_count={args.self_example_max_count}")
        print(f"other_example_max_count={args.other_example_max_count}")
        print(f"other_neighbor_entity_count={args.other_neighbor_entity_count}")
        print(f"other_neighbor_history_count={args.other_neighbor_history_count}")
        print(f"other_neighbor_search_hops={args.other_neighbor_search_hops}")

    torch_profile_enabled = bool(args.enable_torch_profiler)
    torch_profile_limit = max(0, int(args.torch_profiler_max_items))
    torch_profile_dir = Path(args.torch_profiler_dir)
    if torch_profile_enabled and torch_profile_limit > 0:
        torch_profile_dir.mkdir(parents=True, exist_ok=True)
    elif torch_profile_enabled and torch_profile_limit == 0:
        print(
            "torch profiler is enabled but --torch-profiler-max-items=0, "
            "so no traces will be collected."
        )

    predictions = []
    targets = []

    eval_rows = list(eval_table.df.itertuples(index=False))
    if args.debug:
        eval_rows = eval_rows[: args.max_items]

    target_example_count = max(1, int(getattr(resource, "history_length", 10)))
    self_example_max_count = args.self_example_max_count
    other_example_max_count = args.other_example_max_count
    other_neighbor_entity_count = max(0, int(args.other_neighbor_entity_count))
    other_neighbor_history_count = max(0, int(args.other_neighbor_history_count))
    other_neighbor_search_hops = max(1, int(args.other_neighbor_search_hops))
    history_window_ns = int(getattr(resource, "label_horizon_ns", 0) or 0)
    if history_window_ns <= 0:
        task_delta = getattr(resource.task_object, "timedelta", None)
        if task_delta is not None:
            history_window_ns = int(pd.Timedelta(task_delta).value)
    pool_frames = []
    for split_name in ("train", "val", "test"):
        try:
            split_table = resource.task_object.get_table(split_name, mask_input_cols=False)
        except Exception:
            continue
        split_df = split_table.df
        required_cols = {resource.entity_col, resource.time_col, resource.output_col}
        if not required_cols.issubset(set(split_df.columns)):
            continue
        pool_frames.append(split_df.copy())
    if pool_frames:
        example_pool_df = pd.concat(pool_frames, ignore_index=True)
        example_pool_df[resource.time_col] = pd.to_datetime(
            example_pool_df[resource.time_col], errors="coerce"
        )
        example_pool_df = example_pool_df.dropna(subset=[resource.time_col]).reset_index(drop=True)
        example_pool_df = (
            example_pool_df.sort_values(by=[resource.time_col], kind="stable")
            .drop_duplicates(subset=[resource.entity_col, resource.time_col], keep="first")
            .reset_index(drop=True)
        )
    else:
        example_pool_df = pd.DataFrame()

    context_batch_size = max(1, args.context_batch_size)
    llm_batch_size = max(1, args.llm_batch_size)
    dfs_batch_size = max(1, args.dfs_batch_size)
    pipeline_batch_size = (
        max(1, int(args.pipeline_batch_size))
        if args.pipeline_batch_size is not None
        else context_batch_size
    )
    dfs_super_batch_size = dfs_batch_size if effective_use_dfs else pipeline_batch_size

    def _reserve_torch_profile_slot() -> int | None:
        nonlocal torch_profile_next_slot
        if not torch_profile_enabled or torch_profile_limit <= 0:
            return None
        with torch_profile_reservation_lock:
            if torch_profile_next_slot >= torch_profile_limit:
                return None
            reserved_slot = torch_profile_next_slot
            torch_profile_next_slot += 1
            return reserved_slot

    def _dfs_cache_key(row_dict: dict[str, Any]) -> tuple[str, int]:
        return (
            str(row_dict[resource.entity_col]),
            int(pd.Timestamp(row_dict[resource.time_col]).value),
        )

    def _history_key(record: dict[str, Any]) -> tuple[str, int]:
        return (
            str(record[resource.entity_col]),
            int(pd.Timestamp(record[resource.time_col]).value),
        )

    def _entity_key(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (np.integer, int)):
            return str(int(value))
        if isinstance(value, (np.floating, float)):
            if np.isnan(value):
                return ""
            value_f = float(value)
            if value_f.is_integer():
                return str(int(value_f))
            return str(value_f)
        return str(value).strip()

    def _select_self_history_by_window(
        rows: list[dict[str, Any]],
        query_time: Any,
        window_ns: int,
    ) -> list[dict[str, Any]]:
        if window_ns <= 0 or len(rows) <= 1:
            return rows
        query_ts_ns = int(pd.Timestamp(query_time).value)
        selected_by_bucket: dict[int, dict[str, Any]] = {}
        for row in sorted(rows, key=lambda r: pd.Timestamp(r[resource.time_col]), reverse=True):
            ts_ns = int(pd.Timestamp(row[resource.time_col]).value)
            if ts_ns > query_ts_ns:
                continue
            bucket = int((query_ts_ns - ts_ns) // window_ns)
            if bucket not in selected_by_bucket:
                # rows are traversed newest->oldest, so first seen is most recent in its window
                selected_by_bucket[bucket] = row
        return sorted(selected_by_bucket.values(), key=lambda r: pd.Timestamp(r[resource.time_col]))

    def _select_other_entity_examples(
        entity_value: Any,
        cutoff_time: Any,
        max_rows: int,
        neighbor_entity_count: int,
        neighbor_history_count: int,
    ) -> list[dict[str, Any]]:
        if (
            max_rows <= 0
            or neighbor_entity_count <= 0
            or neighbor_history_count <= 0
            or example_pool_df.empty
        ):
            return []
        cutoff_ts = pd.Timestamp(cutoff_time)
        query_entity_key = _entity_key(entity_value)
        pool = example_pool_df[example_pool_df[resource.time_col] <= cutoff_ts].copy()
        pool["__entity_key"] = pool[resource.entity_col].map(_entity_key)
        pool = pool[pool["__entity_key"] != query_entity_key]
        if pool.empty:
            return []

        # Rank nearby same-type entities via PK-FK graph.
        neighbor_rank_by_entity: dict[str, tuple[int, int]] = {}
        start_node_id = graph_store.node_id_map.get((resource.entity_table, entity_value))
        if start_node_id is not None:
            graph_neighbors = graph_store.get_multihop_neighbors_before(
                start_node=int(start_node_id),
                cutoff_time=cutoff_ts,
                num_hops=other_neighbor_search_hops,
                top_k=max(int(args.top_k), int(neighbor_entity_count) * 24),
            )
            for nbr in graph_neighbors:
                dst_id = int(nbr["dst"])
                dst_table, dst_pk = graph_store.rev_node_id[dst_id]
                if dst_table != resource.entity_table or dst_pk == entity_value:
                    continue
                entity_key = _entity_key(dst_pk)
                hop = int(nbr.get("hop", 10**9))
                ts_val = int(nbr.get("ts", -1))
                prev = neighbor_rank_by_entity.get(entity_key)
                if prev is None or hop < prev[0] or (hop == prev[0] and ts_val > prev[1]):
                    neighbor_rank_by_entity[entity_key] = (hop, ts_val)

        if not neighbor_rank_by_entity:
            return []

        ranked_neighbor_entities = sorted(
            neighbor_rank_by_entity.items(),
            key=lambda item: (item[1][0], -item[1][1], item[0]),
        )
        selected_entity_keys = [key for key, _ in ranked_neighbor_entities[:neighbor_entity_count]]
        if not selected_entity_keys:
            return []

        records: list[dict[str, Any]] = []
        for entity_key in selected_entity_keys:
            entity_rows = pool[pool["__entity_key"] == entity_key]
            if entity_rows.empty:
                continue
            recent_rows = (
                entity_rows.sort_values(by=[resource.time_col], ascending=False, kind="stable")
                .head(neighbor_history_count)
                .sort_values(by=[resource.time_col], kind="stable")
            )
            for rec in recent_rows.to_dict(orient="records"):
                rec.pop("__entity_key", None)
                records.append(rec)

        records.sort(key=lambda rec: pd.Timestamp(rec[resource.time_col]))
        if len(records) > max_rows:
            records = records[-max_rows:]
        for record in records:
            record["__example_scope"] = "other"
        return records

    def precompute_histories(indexed_prepare_rows):
        if not indexed_prepare_rows:
            return {}, {}

        unique_requests: dict[tuple[str, int], tuple[Any, Any]] = {}
        for _, row in indexed_prepare_rows:
            record = row._asdict()
            key = _history_key(record)
            if key not in unique_requests:
                unique_requests[key] = (
                    record[resource.entity_col],
                    record[resource.time_col],
                )

        history_by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
        timing_by_key: dict[tuple[str, int], dict[str, float]] = {}
        request_items = list(unique_requests.items())
        bulk_requests = [
            (request_id, entity_value, cutoff_time)
            for request_id, (_, (entity_value, cutoff_time)) in enumerate(request_items)
        ]
        request_id_to_key = {
            request_id: key for request_id, (key, _) in enumerate(request_items)
        }

        if args.bulk_history_query:
            t0 = perf_counter()
            histories_by_request_id = build_inference_histories_bulk(
                store,
                resource,
                bulk_requests,
            )
            bulk_elapsed = perf_counter() - t0
            if args.profile and request_items:
                per_request = bulk_elapsed / len(request_items)
                for _ in request_items:
                    timers.add("prepare.build_inference_history", per_request)

            for request_id, history in histories_by_request_id.items():
                key = request_id_to_key[request_id]
                history_by_key[key] = history
                timing_by_key[key] = {
                    "history_s": bulk_elapsed / max(1, len(request_items)),
                    "validate_s": 0.0,
                }
            for key, _ in request_items:
                history_by_key.setdefault(key, [])
                timing_by_key.setdefault(key, {"history_s": 0.0, "validate_s": 0.0})
            return history_by_key, timing_by_key

        def fetch_one(item):
            key, (entity_value, cutoff_time) = item
            t0 = perf_counter()
            history = build_inference_history(store, resource, entity_value, cutoff_time)
            history_s = perf_counter() - t0
            t0 = perf_counter()
            validate_history_non_overlap(history, resource, cutoff_time)
            validate_s = perf_counter() - t0
            return key, history, history_s, validate_s
        if args.context_workers > 1 and len(request_items) > 1:
            max_workers = min(args.context_workers, len(request_items))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(fetch_one, item) for item in request_items]
                for future in as_completed(futures):
                    key, history, history_s, validate_s = future.result()
                    history_by_key[key] = history
                    timing_by_key[key] = {
                        "history_s": history_s,
                        "validate_s": validate_s,
                    }
                    if args.profile:
                        timers.add("prepare.build_inference_history", history_s)
                        timers.add("prepare.validate_history_non_overlap", validate_s)
        else:
            for item in request_items:
                key, history, history_s, validate_s = fetch_one(item)
                history_by_key[key] = history
                timing_by_key[key] = {
                    "history_s": history_s,
                    "validate_s": validate_s,
                }
                if args.profile:
                    timers.add("prepare.build_inference_history", history_s)
                    timers.add("prepare.validate_history_non_overlap", validate_s)
        return history_by_key, timing_by_key

    def precompute_dfs_summaries(base_items):
        if not effective_use_dfs or dfs_context_builder is None or not base_items:
            return {}
        all_rows: list[dict[str, Any]] = []
        for item in base_items:
            all_rows.extend(item["entry_rows"])
        t0 = perf_counter()
        summaries = dfs_context_builder.summarize_rows(all_rows)
        precompute_s = perf_counter() - t0
        if args.profile:
            timers.add("dfs.precompute_summaries", precompute_s)

        cached: dict[tuple[str, int], list[str]] = {}
        for row_dict, summary_lines in zip(all_rows, summaries):
            cached[_dfs_cache_key(row_dict)] = summary_lines
        return cached

    def precompute_dfs_feature_dicts(base_items):
        if not (effective_use_dfs and include_dfs_table) or dfs_context_builder is None or not base_items:
            return {}
        all_rows: list[dict[str, Any]] = []
        for item in base_items:
            all_rows.extend(item["entry_rows"])
        t0 = perf_counter()
        feature_dicts = dfs_context_builder.feature_dicts_for_rows(all_rows)
        precompute_s = perf_counter() - t0
        if args.profile:
            timers.add("dfs.precompute_feature_table_rows", precompute_s)

        cached: dict[tuple[str, int], dict[str, str]] = {}
        for row_dict, feature_dict in zip(all_rows, feature_dicts):
            cached[_dfs_cache_key(row_dict)] = feature_dict
        return cached

    def finalize_prompt(base_item, dfs_summary_cache, dfs_feature_cache):
        precomputed_entry_dfs_summaries = None
        precomputed_entry_dfs_feature_dicts = None
        if effective_use_dfs:
            if dfs_context_builder is not None:
                precomputed_entry_dfs_summaries = [
                    dfs_summary_cache.get(
                        _dfs_cache_key(row_dict),
                        ["- DFS summary unavailable for this timestamp."],
                    )
                    for row_dict in base_item["entry_rows"]
                ]
                if include_dfs_table:
                    precomputed_entry_dfs_feature_dicts = [
                        dfs_feature_cache.get(_dfs_cache_key(row_dict), {})
                        for row_dict in base_item["entry_rows"]
                    ]
            else:
                precomputed_entry_dfs_summaries = [
                    ["- DFS context requested but FastDFS context builder is unavailable."]
                    for _ in base_item["entry_rows"]
                ]

        prompt_s = 0.0
        item_index = int(base_item["item_index"])
        torch_profile_slot = _reserve_torch_profile_slot()
        if torch_profile_slot is not None:
            trace_path = torch_profile_dir / f"prepare_item_{item_index:06d}.json"
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=args.torch_profiler_record_shapes,
                profile_memory=args.torch_profiler_profile_memory,
                with_stack=args.torch_profiler_with_stack,
            ) as torch_profiler:
                with torch.profiler.record_function("prepare/build_zero_shot_prompt"):
                    t0 = perf_counter()
                    prompt = build_zero_shot_prompt(
                        store=graph_store,
                        extractor=extractor,
                        resource=resource,
                        query_row=base_item["record"],
                        history_rows=base_item["history"],
                        top_k=args.top_k,
                        num_hops=args.num_hops,
                        include_hop_aggregation=args.aggregate,
                        include_semantic_retrieval=args.semantic_retrieval,
                        use_dfs=effective_use_dfs,
                        dfs_context_builder=dfs_context_builder,
                        context_workers=max(1, args.context_workers),
                        precomputed_entry_dfs_summaries=precomputed_entry_dfs_summaries,
                        precomputed_entry_dfs_feature_dicts=precomputed_entry_dfs_feature_dicts,
                        recent_context_k=max(0, args.recent_context_k),
                        include_neighbors=include_neighbors,
                        include_dfs_summary=include_dfs_summary,
                        include_dfs_table=include_dfs_table,
                        other_neighbor_entity_count=other_neighbor_entity_count,
                        other_neighbor_history_count=other_neighbor_history_count,
                    )
                    prompt_s = perf_counter() - t0
            torch_profiler.export_chrome_trace(str(trace_path))
            with torch_profile_saved_lock:
                torch_profile_saved_paths.append(str(trace_path))
        else:
            t0 = perf_counter()
            prompt = build_zero_shot_prompt(
                store=graph_store,
                extractor=extractor,
                resource=resource,
                query_row=base_item["record"],
                history_rows=base_item["history"],
                top_k=args.top_k,
                num_hops=args.num_hops,
                include_hop_aggregation=args.aggregate,
                include_semantic_retrieval=args.semantic_retrieval,
                use_dfs=effective_use_dfs,
                dfs_context_builder=dfs_context_builder,
                context_workers=max(1, args.context_workers),
                precomputed_entry_dfs_summaries=precomputed_entry_dfs_summaries,
                precomputed_entry_dfs_feature_dicts=precomputed_entry_dfs_feature_dicts,
                recent_context_k=max(0, args.recent_context_k),
                include_neighbors=include_neighbors,
                include_dfs_summary=include_dfs_summary,
                include_dfs_table=include_dfs_table,
                other_neighbor_entity_count=other_neighbor_entity_count,
                other_neighbor_history_count=other_neighbor_history_count,
            )
            prompt_s = perf_counter() - t0

        history_s = float(base_item["timing"]["history_s"])
        validate_s = float(base_item["timing"]["validate_s"])
        prepare_total_s = history_s + validate_s + prompt_s
        if args.profile:
            timers.add("prepare.build_zero_shot_prompt", prompt_s)
            timers.add("prepare.total", prepare_total_s)
            with prepare_item_lock:
                prepare_item_stats.append(
                    {
                        "item_index": item_index,
                        "prepare_total_s": prepare_total_s,
                        "history_s": history_s,
                        "validate_s": validate_s,
                        "prompt_s": prompt_s,
                    }
                )

        return {
            **base_item,
            "prompt": prompt,
            "timing": {
                "prepare_total_s": prepare_total_s,
                "history_s": history_s,
                "validate_s": validate_s,
                "prompt_s": prompt_s,
            },
        }

    def build_base_items_for_super_batch(indexed_prepare_rows):
        batch_start = perf_counter()
        history_by_key, history_timing_by_key = precompute_histories(indexed_prepare_rows)
        if args.profile:
            timers.add("prepare.history_precompute_batch", perf_counter() - batch_start)

        base_items = []
        for item_index, row in indexed_prepare_rows:
            record = row._asdict()
            key = _history_key(record)
            full_self_history = [dict(history_row) for history_row in history_by_key.get(key, [])]
            full_self_history = _select_self_history_by_window(
                full_self_history,
                record[resource.time_col],
                history_window_ns,
            )
            if self_example_max_count is None:
                self_history = full_self_history
            elif self_example_max_count == 0:
                self_history = []
            else:
                self_history = full_self_history[-self_example_max_count:]
            for history_row in self_history:
                history_row["__example_scope"] = "self"

            default_other_limit = other_neighbor_entity_count * other_neighbor_history_count
            if other_example_max_count is None:
                other_limit = default_other_limit
            else:
                other_limit = min(other_example_max_count, default_other_limit)

            if other_limit > 0:
                if self_history:
                    latest_self_ts = pd.Timestamp(self_history[-1][resource.time_col])
                else:
                    latest_self_ts = pd.Timestamp(record[resource.time_col])
                other_history = _select_other_entity_examples(
                    record[resource.entity_col],
                    latest_self_ts,
                    other_limit,
                    other_neighbor_entity_count,
                    other_neighbor_history_count,
                )
            else:
                other_history = []

            history = self_history + other_history
            history.sort(key=lambda item: pd.Timestamp(item[resource.time_col]))
            timing = history_timing_by_key.get(key, {"history_s": 0.0, "validate_s": 0.0})
            query_row = dict(record)
            query_row["__example_scope"] = "query"
            entry_rows = [dict(history_row) for history_row in history] + [query_row]
            base_items.append(
                {
                    "item_index": item_index,
                    "row": row,
                    "record": record,
                    "entity_value": record[resource.entity_col],
                    "cutoff_time": record[resource.time_col],
                    "history": history,
                    "entry_rows": entry_rows,
                    "target": record.get(resource.output_col),
                    "timing": {
                        "prepare_total_s": float(timing["history_s"]) + float(timing["validate_s"]),
                        "history_s": float(timing["history_s"]),
                        "validate_s": float(timing["validate_s"]),
                        "prompt_s": 0.0,
                    },
                }
            )

        base_items.sort(key=lambda item: int(item["item_index"]))
        dfs_summary_cache = precompute_dfs_summaries(base_items)
        dfs_feature_cache = precompute_dfs_feature_dicts(base_items)
        return base_items, dfs_summary_cache, dfs_feature_cache

    def finalize_prompt_batch(base_items_batch, dfs_summary_cache, dfs_feature_cache):
        prompt_items = []
        if args.context_workers > 1 and len(base_items_batch) > 1:
            max_workers = min(args.context_workers, len(base_items_batch))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(finalize_prompt, item, dfs_summary_cache, dfs_feature_cache)
                    for item in base_items_batch
                ]
                for future in as_completed(futures):
                    prompt_items.append(future.result())
        else:
            for item in base_items_batch:
                prompt_items.append(finalize_prompt(item, dfs_summary_cache, dfs_feature_cache))
        prompt_items.sort(key=lambda item: int(item["item_index"]))
        return prompt_items

    def consume_llm_items(llm_items):
        if not llm_items:
            return
        # breakpoint()  # for debugging any issues in prompt preparation before LLM generation
        llm_start = perf_counter()
        # if args.print_log and not args.pred_only:
        #     for item in llm_items:
        #         print(item['prompt']) 
        #         breakpoint()
        outputs = llm.generate_batch([item["prompt"] for item in llm_items])
        llm_total_s = perf_counter() - llm_start
        timers.add("llm.generate_batch_total", llm_total_s)
        timers.add("llm.generate_batch_per_item", llm_total_s / max(1, len(llm_items)))
        for item, output in zip(llm_items, outputs):
            parse_start = perf_counter()
            prediction = extract_number(output)
            parse_s = perf_counter() - parse_start
            timers.add("llm.extract_number", parse_s)
            target = item["target"]
            target_float = float(target) if target is not None else None
            large_gap = (
                target_float is not None and abs(float(prediction) - target_float) > 10.0
            )

            if args.print_log:
                print(f"item_index={item['item_index']}")
                print(f"  entity={item['entity_value']}")
                print(f"  timestamp={item['cutoff_time']}")
                print(f"  prediction={prediction} | ground_truth={target}")
                if large_gap:
                    print(
                        f"  large_gap_detected=True | abs_gap={abs(float(prediction) - target_float):.4f}"
                    )
                if not args.pred_only:
                    print(f"  history_len={len(item['history'])}")
                    print(f"  raw_generation={output!r}")
                    # if item["history"]:
                    #     print(item["history"])
                    #     print(item["row"])
                    print("\nPROMPT\n")
                    print(item["prompt"])
                    breakpoint()
                elif large_gap:
                    print(f"  history_len={len(item['history'])}")
                    print(f"  raw_generation={output!r}")
                    print("\nPROMPT\n")
                    print(item["prompt"])
                    breakpoint()

            predictions.append(prediction)
            if target is not None:
                targets.append(target_float)

            if args.profile and args.profile_print_items:
                timing = item.get("timing", {})
                prepare_total = timing.get("prepare_total_s", 0.0)
                history_s = timing.get("history_s", 0.0)
                validate_s = timing.get("validate_s", 0.0)
                prompt_s = timing.get("prompt_s", 0.0)
                llm_share_s = llm_total_s / max(1, len(llm_items))
                total_s = prepare_total + llm_share_s + parse_s
                print(
                    "[profile:item] "
                    f"idx={item['item_index']} "
                    f"prepare={prepare_total:.4f}s "
                    f"(history={history_s:.4f}s, validate={validate_s:.4f}s, prompt={prompt_s:.4f}s) "
                    f"llm_share={llm_share_s:.4f}s "
                    f"parse={parse_s:.4f}s "
                    f"total~={total_s:.4f}s"
                )

    super_batches = []
    for super_start in range(0, len(eval_rows), dfs_super_batch_size):
        super_rows = eval_rows[super_start : super_start + dfs_super_batch_size]
        indexed_super_rows = [
            (super_start + item_offset, row)
            for item_offset, row in enumerate(super_rows)
        ]
        super_batches.append(indexed_super_rows)

    def consume_prompt_items(prompt_items):
        llm_queue = []
        for item in prompt_items:
            llm_queue.append(item)
            if len(llm_queue) >= llm_batch_size:
                consume_llm_items(llm_queue[:llm_batch_size])
                llm_queue = llm_queue[llm_batch_size:]
        consume_llm_items(llm_queue)

    if args.overlap_prep_llm:
        queue_size = max(1, int(args.prep_queue_size))
        prompt_batch_queue: Queue[Any] = Queue(maxsize=queue_size)
        sentinel = object()
        producer_error: list[BaseException] = []

        def producer() -> None:
            try:
                for indexed_super_rows in super_batches:
                    base_items, dfs_summary_cache, dfs_feature_cache = build_base_items_for_super_batch(
                        indexed_super_rows
                    )
                    for micro_start in range(0, len(base_items), pipeline_batch_size):
                        base_chunk = base_items[micro_start : micro_start + pipeline_batch_size]
                        prompt_items = finalize_prompt_batch(
                            base_chunk, dfs_summary_cache, dfs_feature_cache
                        )
                        prompt_batch_queue.put(prompt_items)
            except BaseException as exc:  # pragma: no cover
                producer_error.append(exc)
            finally:
                prompt_batch_queue.put(sentinel)

        producer_thread = threading.Thread(
            target=producer,
            name="prompt-prep-producer",
            daemon=True,
        )
        producer_thread.start()
        while True:
            payload = prompt_batch_queue.get()
            if payload is sentinel:
                break
            consume_prompt_items(payload)
        producer_thread.join()
        if producer_error:
            raise producer_error[0]
    else:
        for indexed_super_rows in super_batches:
            base_items, dfs_summary_cache, dfs_feature_cache = build_base_items_for_super_batch(
                indexed_super_rows
            )
            for micro_start in range(0, len(base_items), pipeline_batch_size):
                base_chunk = base_items[micro_start : micro_start + pipeline_batch_size]
                prompt_items = finalize_prompt_batch(
                    base_chunk, dfs_summary_cache, dfs_feature_cache
                )
                consume_prompt_items(prompt_items)

    timers.add("pipeline.total_runtime", perf_counter() - global_start)

    if predictions and len(predictions) == len(targets):
        mae = np.mean(np.abs(np.array(predictions, dtype=float) - np.array(targets, dtype=float)))
        print(f"\nFinal MAE: {mae}")
    elif predictions:
        print("\nPredictions generated, but no labels were available for MAE.")
    else:
        print("\nNo predictions generated.")

    if args.profile:
        print("\nTiming Summary (seconds)")
        print("phase,count,total,avg,p50,p95,max")
        for row in timers.summary_rows():
            print(
                f"{row['phase']},{row['count']},{row['total_s']:.4f},{row['avg_s']:.4f},"
                f"{row['p50_s']:.4f},{row['p95_s']:.4f},{row['max_s']:.4f}"
            )

        pipeline_total_s = timers.total_for_phase("pipeline.total_runtime")
        llm_generate_s = timers.total_for_phase("llm.generate_batch_total")
        llm_parse_s = timers.total_for_phase("llm.extract_number")
        llm_total_s = llm_generate_s + llm_parse_s
        non_llm_visible_s = max(0.0, pipeline_total_s - llm_total_s)
        setup_total_s = timers.total_for_prefix("setup.")
        prepare_total_s = timers.total_for_phase("prepare.total")
        dfs_total_s = timers.total_for_phase("dfs.precompute_summaries")
        non_llm_actual_total_s = setup_total_s + dfs_total_s + prepare_total_s
        # Estimated serial runtime if prep and LLM are not overlapped in the pipeline.
        no_pipeline_overlap_expected_s = non_llm_actual_total_s + llm_total_s
        processed_items = max(1, len(predictions))
        per_item_total_inference_s = pipeline_total_s / processed_items

        print("\nRuntime Summary (seconds)")
        print(f"total_time={pipeline_total_s:.4f}")
        print(f"pipeline_total={pipeline_total_s:.4f}")
        print(f"llm_time={llm_total_s:.4f}")
        print(f"non_llm_time={non_llm_visible_s:.4f}")
        print(f"non_llm_actual_total={non_llm_actual_total_s:.4f}")
        print(f"without_pipeline_expected_total={no_pipeline_overlap_expected_s:.4f}")
        print(f"per_item_total_inference_time={per_item_total_inference_s:.4f}")

        if prepare_item_stats:
            sorted_items = sorted(
                prepare_item_stats,
                key=lambda x: x["prepare_total_s"],
                reverse=True,
            )
            top_n = max(0, min(args.profile_slowest_items, len(sorted_items)))
            if top_n > 0:
                print(f"\nTop {top_n} Slowest prepare_item calls")
                print("item_index,prepare_total,history,validate,prompt")
                for item in sorted_items[:top_n]:
                    print(
                        f"{int(item['item_index'])},{item['prepare_total_s']:.4f},"
                        f"{item['history_s']:.4f},{item['validate_s']:.4f},{item['prompt_s']:.4f}"
                    )

    if torch_profile_enabled:
        print("\nPyTorch prepare-item traces")
        print(f"requested_max_items={torch_profile_limit}")
        print(f"saved_traces={len(torch_profile_saved_paths)}")
        print(f"trace_dir={torch_profile_dir}")
        if torch_profile_saved_paths and args.print_log:
            for trace_path in torch_profile_saved_paths:
                print(f"trace={trace_path}")


def main() -> None:
    args = extend_parser().parse_args()
    if not args.enable_cprofile:
        run(args)
        return

    profiler = cProfile.Profile()
    try:
        profiler.enable()
        run(args)
    finally:
        profiler.disable()
        print("\nPython cProfile Summary")
        stats = pstats.Stats(profiler).sort_stats(args.cprofile_sort)
        stats.print_stats(max(1, args.cprofile_top_n))
        if args.cprofile_output:
            profiler.dump_stats(args.cprofile_output)
            print(f"cProfile stats saved to {args.cprofile_output}")


if __name__ == "__main__":
    main()
