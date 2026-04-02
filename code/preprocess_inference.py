from __future__ import annotations

from load_inference_config import build_parser, load_and_validate_inference_config
from train_data import preprocess_and_save_train_data


def extend_parser():
    parser = build_parser(
        "Preprocess and save inference data and history candidates for a single task."
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
    args = extend_parser().parse_args()
    config = load_and_validate_inference_config(args.config.resolve())
    output_dir = preprocess_and_save_train_data(
        config,
        no_dataset_download=args.no_dataset_download,
        no_task_download=args.no_task_download,
    )
    print(f"preprocessed_inference_dir={output_dir}")


if __name__ == "__main__":
    main()
