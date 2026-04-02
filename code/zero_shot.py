from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from load_inference_config import build_parser, load_and_validate_inference_config
from grag import RelBenchGraphRAGStore, RowFeatureExtractor, build_zero_shot_prompt
from inference_history import (
    build_history_store,
    build_inference_history,
    build_inference_resource,
    validate_history_non_overlap,
)
from zero_shot_llm import LocalLLM, MODEL_PATHS, extract_number


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
        "--max-items",
        type=int,
        default=10,
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
        default=2,
        help="Number of graph hops to retrieve for each history item.",
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
        default=4,
        help="Number of examples to prepare prompts for before LLM generation.",
    )
    parser.add_argument(
        "--llm-batch-size",
        type=int,
        default=1,
        help="Number of prompts to send to the LLM at once.",
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
        "--print-log",
        action="store_true",
        help="Print detailed zero-shot inspection logs.",
    )
    parser.add_argument(
        "--pred-only",
        action="store_true",
        help="With --print-log, print only prediction and ground truth per item.",
    )
    return parser


def main() -> None:
    args = extend_parser().parse_args()
    config = load_and_validate_inference_config(args.config.resolve())
    if args.history_sampling_strategy is not None:
        config["history_sampling_strategy"] = args.history_sampling_strategy

    resource = build_inference_resource(
        config,
        no_dataset_download=args.no_dataset_download,
        no_task_download=args.no_task_download,
    )
    store, _ = build_history_store(resource)
    graph_store = RelBenchGraphRAGStore()
    graph_store.build_base(resource.db, resource.task_object)
    graph_store.parallel_frontier_workers = max(1, args.frontier_workers)
    extractor = RowFeatureExtractor(resource.db, graph_store)
    llm = LocalLLM(MODEL_PATHS[args.model_size], print_log=args.print_log)

    eval_table = resource.task_object.get_table("test", mask_input_cols=False)
    if resource.output_col not in eval_table.df.columns:
        if args.print_log:
            print("using val for evaluation since test has no labels")
        eval_table = resource.task_object.get_table("val", mask_input_cols=False)

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

    predictions = []
    targets = []

    eval_rows = list(eval_table.df.itertuples(index=False))
    if args.debug:
        eval_rows = eval_rows[: args.max_items]

    context_batch_size = max(1, args.context_batch_size)
    llm_batch_size = max(1, args.llm_batch_size)

    def prepare_item(item_index_and_row):
        item_index, row = item_index_and_row
        record = row._asdict()
        entity_value = record[resource.entity_col]
        cutoff_time = record[resource.time_col]
        history = build_inference_history(store, resource, entity_value, cutoff_time)
        validate_history_non_overlap(history, resource, cutoff_time)
        prompt = build_zero_shot_prompt(
            store=graph_store,
            extractor=extractor,
            resource=resource,
            query_row=record,
            history_rows=history,
            top_k=args.top_k,
            num_hops=args.num_hops,
            include_hop_aggregation=args.aggregate,
            include_semantic_retrieval=args.semantic_retrieval,
            context_workers=max(1, args.context_workers),
        )


        return {
            "item_index": item_index,
            "row": row,
            "record": record,
            "entity_value": entity_value,
            "cutoff_time": cutoff_time,
            "history": history,
            "prompt": prompt,
            "target": record.get(resource.output_col),
        }

    def consume_llm_items(llm_items):
        if not llm_items:
            return
        # breakpoint()  # for debugging any issues in prompt preparation before LLM generation
        outputs = llm.generate_batch([item["prompt"] for item in llm_items])
        for item, output in zip(llm_items, outputs):
            prediction = extract_number(output)
            target = item["target"]

            if args.print_log:
                print(f"item_index={item['item_index']}")
                print(f"  entity={item['entity_value']}")
                print(f"  timestamp={item['cutoff_time']}")
                print(f"  prediction={prediction} | ground_truth={target}")
                if not args.pred_only:
                    print(f"  history_len={len(item['history'])}")
                    print(f"  raw_generation={output!r}")
                    if item["history"]:
                        print(item["history"])
                        print(item["row"])
                    print("\nPROMPT\n")
                    print(item["prompt"])

            predictions.append(prediction)
            if target is not None:
                targets.append(float(target))

    for context_start in range(0, len(eval_rows), context_batch_size):
        context_rows = eval_rows[context_start : context_start + context_batch_size]
        indexed_context_rows = [
            (context_start + item_offset, row)
            for item_offset, row in enumerate(context_rows)
        ]

        llm_queue = []
        if args.context_workers > 1 and len(indexed_context_rows) > 1:
            max_workers = min(args.context_workers, len(indexed_context_rows))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(prepare_item, item) for item in indexed_context_rows]
                for future in as_completed(futures):
                    llm_queue.append(future.result())
                    if len(llm_queue) >= llm_batch_size:
                        consume_llm_items(llm_queue[:llm_batch_size])
                        llm_queue = llm_queue[llm_batch_size:]
        else:
            for item in indexed_context_rows:
                llm_queue.append(prepare_item(item))
                if len(llm_queue) >= llm_batch_size:
                    consume_llm_items(llm_queue[:llm_batch_size])
                    llm_queue = llm_queue[llm_batch_size:]

        consume_llm_items(llm_queue)

    if predictions and len(predictions) == len(targets):
        mae = np.mean(np.abs(np.array(predictions, dtype=float) - np.array(targets, dtype=float)))
        print(f"\nFinal MAE: {mae}")
    elif predictions:
        print("\nPredictions generated, but no labels were available for MAE.")
    else:
        print("\nNo predictions generated.")


if __name__ == "__main__":
    main()
