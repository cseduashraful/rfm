from __future__ import annotations

from typing import Any, Callable

import duckdb
import pandas as pd


TaskHistoryQueryFn = Callable[[Any, Any, pd.Timestamp], pd.DataFrame]
TaskHistoryBulkQueryFn = Callable[[Any, pd.DataFrame], pd.DataFrame]
TaskHistoryFeatureHint = dict[str, dict[Any, int]]


def _register_rel_amazon_tables(conn: duckdb.DuckDBPyConnection, db: Any) -> None:
    conn.register("review", db.table_dict["review"].df)
    conn.register("product", db.table_dict["product"].df)
    conn.register("customer", db.table_dict["customer"].df)


def _register_rel_f1_tables(conn: duckdb.DuckDBPyConnection, db: Any) -> None:
    conn.register("results", db.table_dict["results"].df)
    conn.register("drivers", db.table_dict["drivers"].df)
    conn.register("races", db.table_dict["races"].df)


def _raw_dataset_timestamps_before(db: Any, cutoff_time: pd.Timestamp) -> pd.DataFrame:
    timestamp_series: list[pd.Series] = []
    for table in db.table_dict.values():
        time_col = table.time_col
        if time_col is None or time_col not in table.df.columns:
            continue
        series = pd.to_datetime(table.df[time_col], errors="coerce").dropna()
        if not series.empty:
            timestamp_series.append(series)

    if not timestamp_series:
        return pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns]")})

    timestamps = pd.concat(timestamp_series, ignore_index=True)
    timestamps = timestamps[timestamps < cutoff_time]
    if timestamps.empty:
        return pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns]")})

    timestamps = pd.Series(timestamps.sort_values().unique(), name="timestamp")
    return timestamps.to_frame()


def _fetch_rel_amazon_item_ltv(resource: Any, entity_value: Any, effective_cutoff: pd.Timestamp) -> pd.DataFrame:
    horizon_days = max(1, int(resource.task_object.timedelta / pd.Timedelta(days=1)))
    with duckdb.connect() as conn:
        _register_rel_amazon_tables(conn, resource.db)
        conn.register("timestamp_df", _raw_dataset_timestamps_before(resource.db, effective_cutoff))
        return conn.execute(
            f"""
            WITH candidate_timestamps AS (
                SELECT timestamp
                FROM timestamp_df
            ),
            entity_reviews AS (
                SELECT
                    review.review_time,
                    product.price
                FROM review
                JOIN product
                  ON review.product_id = product.product_id
                WHERE review.product_id = ?
            )
            SELECT
                candidate_timestamps.timestamp AS timestamp,
                CAST(? AS BIGINT) AS product_id,
                COALESCE(SUM(entity_reviews.price), 0) AS ltv
            FROM candidate_timestamps
            JOIN entity_reviews
              ON entity_reviews.review_time > candidate_timestamps.timestamp
             AND entity_reviews.review_time <= candidate_timestamps.timestamp + INTERVAL '{horizon_days} days'
            GROUP BY candidate_timestamps.timestamp, product_id
            ORDER BY candidate_timestamps.timestamp
            """,
            [entity_value, entity_value],
        ).fetch_df()


def _fetch_rel_amazon_user_ltv(resource: Any, entity_value: Any, effective_cutoff: pd.Timestamp) -> pd.DataFrame:
    horizon_days = max(1, int(resource.task_object.timedelta / pd.Timedelta(days=1)))
    with duckdb.connect() as conn:
        _register_rel_amazon_tables(conn, resource.db)
        conn.register("timestamp_df", _raw_dataset_timestamps_before(resource.db, effective_cutoff))
        return conn.execute(
            f"""
            WITH candidate_timestamps AS (
                SELECT timestamp
                FROM timestamp_df
            ),
            entity_reviews AS (
                SELECT
                    review.review_time,
                    product.price
                FROM review
                JOIN product
                  ON review.product_id = product.product_id
                WHERE review.customer_id = ?
            )
            SELECT
                candidate_timestamps.timestamp AS timestamp,
                CAST(? AS BIGINT) AS customer_id,
                COALESCE(SUM(future_reviews.price), 0) AS ltv
            FROM candidate_timestamps
            LEFT JOIN entity_reviews AS future_reviews
              ON future_reviews.review_time > candidate_timestamps.timestamp
             AND future_reviews.review_time <= candidate_timestamps.timestamp + INTERVAL '{horizon_days} days'
            WHERE EXISTS (
                SELECT 1
                FROM entity_reviews AS past_reviews
                WHERE past_reviews.review_time > candidate_timestamps.timestamp - INTERVAL '{horizon_days} days'
                  AND past_reviews.review_time <= candidate_timestamps.timestamp
            )
            GROUP BY candidate_timestamps.timestamp, customer_id
            ORDER BY candidate_timestamps.timestamp
            """,
            [entity_value, entity_value],
        ).fetch_df()


def _fetch_rel_f1_driver_position(resource: Any, entity_value: Any, effective_cutoff: pd.Timestamp) -> pd.DataFrame:
    horizon_days = max(1, int(resource.task_object.timedelta / pd.Timedelta(days=1)))
    with duckdb.connect() as conn:
        _register_rel_f1_tables(conn, resource.db)
        conn.register("timestamp_df", _raw_dataset_timestamps_before(resource.db, effective_cutoff))
        return conn.execute(
            f"""
            WITH candidate_timestamps AS (
                SELECT timestamp
                FROM timestamp_df
            )
            SELECT
                candidate_timestamps.timestamp AS date,
                CAST(? AS BIGINT) AS driverId,
                AVG(re.positionOrder) AS position
            FROM candidate_timestamps
            LEFT JOIN results AS re
              ON re.date <= candidate_timestamps.timestamp + INTERVAL '{horizon_days} days'
             AND re.date > candidate_timestamps.timestamp
             AND re.driverId = ?
            WHERE EXISTS (
                SELECT 1
                FROM results AS prior_results
                WHERE prior_results.driverId = ?
                  AND prior_results.date > candidate_timestamps.timestamp - INTERVAL '1 year'
            )
            GROUP BY candidate_timestamps.timestamp, driverId
            HAVING COUNT(re.positionOrder) > 0
            ORDER BY candidate_timestamps.timestamp
            """,
            [entity_value, entity_value, entity_value],
        ).fetch_df()


def _fetch_rel_f1_driver_position_bulk(resource: Any, request_frame: pd.DataFrame) -> pd.DataFrame:
    if request_frame.empty:
        return pd.DataFrame(
            {
                "request_id": pd.Series(dtype="int64"),
                "date": pd.Series(dtype="datetime64[ns]"),
                "driverId": pd.Series(dtype="int64"),
                "position": pd.Series(dtype="float64"),
            }
        )

    horizon_days = max(1, int(resource.task_object.timedelta / pd.Timedelta(days=1)))
    request_frame = request_frame.copy()
    request_frame["effective_cutoff"] = pd.to_datetime(
        request_frame["effective_cutoff"]
    )
    max_cutoff = request_frame["effective_cutoff"].max()

    with duckdb.connect() as conn:
        _register_rel_f1_tables(conn, resource.db)
        conn.register("request_df", request_frame)
        conn.register(
            "timestamp_df",
            _raw_dataset_timestamps_before(resource.db, max_cutoff),
        )
        return conn.execute(
            f"""
            WITH request_rows AS (
                SELECT
                    CAST(request_id AS BIGINT) AS request_id,
                    CAST(entity_value AS BIGINT) AS driverId,
                    effective_cutoff
                FROM request_df
            ),
            candidate_timestamps AS (
                SELECT
                    r.request_id,
                    r.driverId,
                    ts.timestamp
                FROM request_rows AS r
                JOIN timestamp_df AS ts
                  ON ts.timestamp < r.effective_cutoff
            )
            SELECT
                candidate_timestamps.request_id AS request_id,
                candidate_timestamps.timestamp AS date,
                candidate_timestamps.driverId AS driverId,
                AVG(re.positionOrder) AS position
            FROM candidate_timestamps
            LEFT JOIN results AS re
              ON re.date <= candidate_timestamps.timestamp + INTERVAL '{horizon_days} days'
             AND re.date > candidate_timestamps.timestamp
             AND re.driverId = candidate_timestamps.driverId
            WHERE EXISTS (
                SELECT 1
                FROM results AS prior_results
                WHERE prior_results.driverId = candidate_timestamps.driverId
                  AND prior_results.date > candidate_timestamps.timestamp - INTERVAL '1 year'
            )
            GROUP BY
                candidate_timestamps.request_id,
                candidate_timestamps.timestamp,
                candidate_timestamps.driverId
            HAVING COUNT(re.positionOrder) > 0
            ORDER BY candidate_timestamps.request_id, candidate_timestamps.timestamp
            """
        ).fetch_df()


TASK_HISTORY_QUERY_REGISTRY: dict[tuple[str, str], TaskHistoryQueryFn] = {
    ("rel-amazon", "item-ltv"): _fetch_rel_amazon_item_ltv,
    ("rel-amazon", "user-ltv"): _fetch_rel_amazon_user_ltv,
    ("rel-f1", "driver-position"): _fetch_rel_f1_driver_position,
}

TASK_HISTORY_BULK_QUERY_REGISTRY: dict[tuple[str, str], TaskHistoryBulkQueryFn] = {
    ("rel-f1", "driver-position"): _fetch_rel_f1_driver_position_bulk,
}

TASK_HISTORY_FEATURE_HINTS_REGISTRY: dict[tuple[str, str], TaskHistoryFeatureHint] = {
    # Derived from task SQL:
    # AVG(results.positionOrder) grouped by candidate timestamp and driver,
    # filtered by results.date windows and results.driverId equality.
    ("rel-f1", "driver-position"): {
        "tables": {
            "results": 4,
            "drivers": 1,
            "races": 1,
        },
        "columns": {
            ("results", "positionOrder"): 8,
            ("results", "date"): 5,
            ("results", "driverId"): 5,
            ("results", "position"): 3,
            ("results", "rank"): 2,
            ("results", "grid"): 2,
            ("results", "laps"): 2,
        },
    },
    # Derived from task SQL:
    # SUM(product.price) over review.review_time windows keyed by product/customer entities.
    ("rel-amazon", "item-ltv"): {
        "tables": {
            "review": 4,
            "product": 4,
            "customer": 1,
        },
        "columns": {
            ("product", "price"): 8,
            ("review", "review_time"): 5,
            ("review", "product_id"): 5,
            ("review", "customer_id"): 3,
        },
    },
    ("rel-amazon", "user-ltv"): {
        "tables": {
            "review": 4,
            "product": 4,
            "customer": 1,
        },
        "columns": {
            ("product", "price"): 8,
            ("review", "review_time"): 5,
            ("review", "customer_id"): 5,
            ("review", "product_id"): 3,
        },
    },
}


def get_task_history_query(dataset_name: str, task_name: str) -> TaskHistoryQueryFn | None:
    return TASK_HISTORY_QUERY_REGISTRY.get((dataset_name, task_name))


def get_task_history_query_bulk(
    dataset_name: str,
    task_name: str,
) -> TaskHistoryBulkQueryFn | None:
    return TASK_HISTORY_BULK_QUERY_REGISTRY.get((dataset_name, task_name))


def get_task_history_feature_hints(
    dataset_name: str,
    task_name: str,
) -> TaskHistoryFeatureHint | None:
    return TASK_HISTORY_FEATURE_HINTS_REGISTRY.get((dataset_name, task_name))
