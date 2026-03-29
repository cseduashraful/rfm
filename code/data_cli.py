from __future__ import annotations

import argparse
from pathlib import Path

from data_config import RelBenchDuckDBConfig
from data_pipeline import RelBenchDuckDBPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize RelBench datasets and expose them through DuckDB."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="RelBench dataset name, e.g. rel-amazon, rel-event, rel-trial.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Override the root directory used for RelBench parquet caches.",
    )
    parser.add_argument(
        "--duckdb-path",
        type=Path,
        default=None,
        help="Override the DuckDB file path.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Force RelBench download/verification. This is enabled by default.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Use the existing RelBench cache only and skip download/verification.",
    )
    parser.add_argument(
        "--include-future-rows",
        action="store_true",
        help="Disable truncation at the dataset test timestamp.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("materialize", help="Build/cache the RelBench parquet tables.")
    subparsers.add_parser("tables", help="Print table metadata.")

    sql_parser = subparsers.add_parser("sql", help="Run a SQL query against DuckDB views.")
    sql_parser.add_argument("--query", required=True, help="SQL query to execute.")
    return parser


def make_pipeline(args: argparse.Namespace) -> RelBenchDuckDBPipeline:
    config = RelBenchDuckDBConfig(
        dataset_name=args.dataset,
        cache_root=args.cache_root,
        duckdb_path=args.duckdb_path,
        download=not args.no_download,
        upto_test_timestamp=not args.include_future_rows,
    )
    return RelBenchDuckDBPipeline(config)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    pipeline = make_pipeline(args)

    if args.command == "materialize":
        db = pipeline.materialize()
        print(f"Materialized {len(db.table_dict)} tables into {pipeline.config.parquet_dir}")
        return

    if args.command == "tables":
        for info in pipeline.list_tables():
            print(f"[{info.name}]")
            print(f"  rows: {info.rows}")
            print(f"  columns: {', '.join(info.columns)}")
            print(f"  parquet: {info.parquet_path}")
            print(f"  primary_key: {info.primary_key}")
            print(f"  time_column: {info.time_column}")
            print(f"  foreign_keys: {info.foreign_keys}")
        return

    if args.command == "sql":
        result = pipeline.query(args.query)
        print(result.to_string(index=False))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
