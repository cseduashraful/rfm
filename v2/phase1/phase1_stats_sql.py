from __future__ import annotations

import itertools
import time
from typing import Any

import numpy as np
import pandas as pd

from phase1_data_access import Phase1DataAccess, TableMetadata
from phase1_utils import (
    cramers_v,
    eta_squared,
    is_boolean_dtype_name,
    is_numeric_dtype_name,
    is_text_dtype_name,
    quote_ident,
    safe_mode,
)


def build_table_stats(
    *,
    access: Phase1DataAccess,
    table_meta: TableMetadata,
    config: Any,
) -> tuple[dict[str, Any], float]:
    t0 = time.perf_counter()
    filtered_rows = access.count_rows(table_meta.table_name, filtered=True)
    raw_rows = access.count_rows(table_meta.table_name, filtered=False)
    invalid_time_rows = access.count_invalid_time_rows(table_meta.table_name)

    max_rows_exact_profile = int(
        getattr(config, "max_rows_exact_profile", getattr(config, "sampling_row_threshold", 200_000))
    )
    profile_sample_size = min(
        int(getattr(config, "sample_size", 100_000)),
        filtered_rows,
    )
    profiling_mode = "exact" if filtered_rows <= max_rows_exact_profile else "sampled"
    sampled = profiling_mode != "exact"

    profiled_relation_sql = access.relation_sql(table_meta.table_name, filtered=True)
    if sampled and profile_sample_size > 0:
        profiled_relation_sql = access.sampled_relation_sql(
            table_meta.table_name,
            filtered=True,
            sample_size=profile_sample_size,
        )
    profiled_row_count = filtered_rows if not sampled else profile_sample_size

    missing_counts = _missingness_counts(access, table_meta, filtered=True)
    distinct_estimates = _distinct_count_estimates(
        access=access,
        table_meta=table_meta,
        relation_sql=profiled_relation_sql,
        exact=(not sampled),
        max_text_cardinality=int(getattr(config, "max_cardinality_for_topk", 5_000)),
    )

    columns: dict[str, Any] = {}
    for column_name in table_meta.columns:
        dtype_name = table_meta.dtypes.get(column_name, "")
        missing_count = int(missing_counts.get(column_name, 0))
        non_null_count = max(filtered_rows - missing_count, 0)
        base = {
            "missingness": (float(missing_count) / float(filtered_rows)) if filtered_rows else None,
            "non_null_count": non_null_count,
            "sampled": sampled,
            "profile_sample_size": int(profiled_row_count),
            "profiling_mode": profiling_mode,
            "distinct_count_method": distinct_estimates.get(column_name, {}).get("method", "unknown"),
        }
        if is_numeric_dtype_name(dtype_name) and not is_boolean_dtype_name(dtype_name):
            base["numeric"] = _numeric_stats_for_column(
                access=access,
                relation_sql=profiled_relation_sql,
                column_name=column_name,
                histogram_bins=int(getattr(config, "histogram_bins", 20)),
            )
        else:
            base["categorical"] = _categorical_stats_for_column(
                access=access,
                relation_sql=profiled_relation_sql,
                column_name=column_name,
                distinct_estimate=distinct_estimates.get(column_name, {}).get("value"),
                topk=int(getattr(config, "categorical_topk", 20)),
                skip_high_cardinality_at=int(getattr(config, "max_cardinality_for_topk", 5_000)),
            )
        distinct_value = distinct_estimates.get(column_name, {}).get("value")
        if distinct_value is not None:
            base["distinct_count"] = int(distinct_value)
        columns[column_name] = base

    corr_columns = _select_correlation_columns(table_meta, config)
    sample_for_corr = pd.DataFrame(columns=corr_columns)
    corr_sample_size = 0
    if corr_columns and not bool(getattr(config, "skip_correlations", False)):
        corr_sample_size = min(
            filtered_rows,
            int(getattr(config, "correlation_sample_size", getattr(config, "sample_size", 100_000))),
        )
        if corr_sample_size > 0:
            sample_for_corr = access.sample_df(
                table_meta.table_name,
                filtered=True,
                sample_size=corr_sample_size,
                columns=corr_columns,
            )
    correlations = _build_correlations(
        sample_df=sample_for_corr,
        table_name=table_meta.table_name,
        sampled=bool(corr_columns) and (not sample_for_corr.empty) and corr_sample_size < filtered_rows,
        config=config,
    )

    payload = {
        "table_name": table_meta.table_name,
        "row_count": filtered_rows,
        "sampled": sampled,
        "profiling_mode": profiling_mode,
        "rows_before_cutoff_filter": raw_rows,
        "rows_after_cutoff_filter": filtered_rows,
        "rows_dropped_invalid_time": invalid_time_rows,
        "profile_sample_size": int(profiled_row_count),
        "temporal_coverage": _temporal_coverage(access, table_meta),
        "columns": columns,
        "correlations": correlations,
    }
    return payload, time.perf_counter() - t0


def _missingness_counts(
    access: Phase1DataAccess,
    table_meta: TableMetadata,
    *,
    filtered: bool,
) -> dict[str, int]:
    relation_sql = access.relation_sql(table_meta.table_name, filtered=filtered)
    exprs = [
        f"SUM(CASE WHEN {quote_ident(col)} IS NULL THEN 1 ELSE 0 END) AS {quote_ident(col)}"
        for col in table_meta.columns
    ]
    row = access.fetchdf(f"SELECT {', '.join(exprs)} FROM ({relation_sql}) rel")
    if row.empty:
        return {col: 0 for col in table_meta.columns}
    values = row.iloc[0].to_dict()
    return {col: int(values.get(col, 0) or 0) for col in table_meta.columns}


def _distinct_count_estimates(
    *,
    access: Phase1DataAccess,
    table_meta: TableMetadata,
    relation_sql: str,
    exact: bool,
    max_text_cardinality: int,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for column_name in table_meta.columns:
        dtype_name = table_meta.dtypes.get(column_name, "")
        method = "count_distinct" if exact else "approx_count_distinct"
        expr = f"COUNT(DISTINCT {quote_ident(column_name)})"
        if not exact:
            expr = f"APPROX_COUNT_DISTINCT({quote_ident(column_name)})"
        if is_text_dtype_name(dtype_name) and not exact:
            method = "sampled_topk_only"
            out[column_name] = {"value": None, "method": method}
            continue
        try:
            value = access.fetch_scalar(f"SELECT {expr} FROM ({relation_sql}) rel")
            out[column_name] = {"value": int(value) if value is not None else 0, "method": method}
        except Exception:
            out[column_name] = {"value": None, "method": "failed"}
    return out


def _numeric_stats_for_column(
    *,
    access: Phase1DataAccess,
    relation_sql: str,
    column_name: str,
    histogram_bins: int,
) -> dict[str, Any]:
    qcol = quote_ident(column_name)
    try:
        row = access.fetchdf(
            "SELECT "
            f"MIN(CAST({qcol} AS DOUBLE)) AS min_value, "
            f"MAX(CAST({qcol} AS DOUBLE)) AS max_value, "
            f"AVG(CAST({qcol} AS DOUBLE)) AS mean_value, "
            f"STDDEV_SAMP(CAST({qcol} AS DOUBLE)) AS std_value, "
            f"MEDIAN(CAST({qcol} AS DOUBLE)) AS median_value, "
            f"QUANTILE_CONT(CAST({qcol} AS DOUBLE), 0.10) AS p10, "
            f"QUANTILE_CONT(CAST({qcol} AS DOUBLE), 0.25) AS p25, "
            f"QUANTILE_CONT(CAST({qcol} AS DOUBLE), 0.50) AS p50, "
            f"QUANTILE_CONT(CAST({qcol} AS DOUBLE), 0.75) AS p75, "
            f"QUANTILE_CONT(CAST({qcol} AS DOUBLE), 0.90) AS p90 "
            f"FROM ({relation_sql}) rel WHERE {qcol} IS NOT NULL"
        )
    except Exception:
        row = pd.DataFrame()
    if row.empty:
        return {}
    numeric_row = row.iloc[0].to_dict()

    series_df = access.fetchdf(
        f"SELECT CAST({qcol} AS DOUBLE) AS value FROM ({relation_sql}) rel "
        f"WHERE {qcol} IS NOT NULL"
    )
    if series_df.empty:
        return {}
    values = pd.to_numeric(series_df["value"], errors="coerce").dropna()
    if values.empty:
        return {}
    hist_counts, hist_edges = np.histogram(values.to_numpy(), bins=histogram_bins)
    return {
        "min": _safe_float(numeric_row.get("min_value")),
        "max": _safe_float(numeric_row.get("max_value")),
        "mean": _safe_float(numeric_row.get("mean_value")),
        "std": _safe_float(numeric_row.get("std_value"), default=0.0),
        "median": _safe_float(numeric_row.get("median_value")),
        "mode": safe_mode(values),
        "quantiles": {
            "p10": _safe_float(numeric_row.get("p10")),
            "p25": _safe_float(numeric_row.get("p25")),
            "p50": _safe_float(numeric_row.get("p50")),
            "p75": _safe_float(numeric_row.get("p75")),
            "p90": _safe_float(numeric_row.get("p90")),
        },
        "histogram": {
            "counts": hist_counts.tolist(),
            "bin_edges": hist_edges.tolist(),
        },
    }


def _categorical_stats_for_column(
    *,
    access: Phase1DataAccess,
    relation_sql: str,
    column_name: str,
    distinct_estimate: int | None,
    topk: int,
    skip_high_cardinality_at: int,
) -> dict[str, Any]:
    qcol = quote_ident(column_name)
    if distinct_estimate is not None and distinct_estimate > skip_high_cardinality_at:
        return {
            "cardinality": int(distinct_estimate),
            "mode": None,
            "top_values": [],
            "top_values_skipped": True,
        }
    try:
        top_values = access.fetchdf(
            f"SELECT CAST({qcol} AS VARCHAR) AS value, COUNT(*) AS count "
            f"FROM ({relation_sql}) rel WHERE {qcol} IS NOT NULL "
            f"GROUP BY 1 ORDER BY count DESC, value ASC LIMIT {int(topk)}"
        )
    except Exception:
        top_values = pd.DataFrame(columns=["value", "count"])
    rows = top_values.to_dict(orient="records")
    return {
        "cardinality": int(distinct_estimate) if distinct_estimate is not None else None,
        "mode": rows[0]["value"] if rows else None,
        "top_values": [
            {"value": str(row["value"]), "count": int(row["count"])}
            for row in rows
        ],
    }


def _temporal_coverage(access: Phase1DataAccess, table_meta: TableMetadata) -> dict[str, Any]:
    if table_meta.time_column is None:
        return {"is_static": True}
    qcol = quote_ident(table_meta.time_column)
    relation_sql = access.relation_sql(table_meta.table_name, filtered=True)
    bounds = access.fetchdf(
        f"SELECT MIN(TRY_CAST({qcol} AS TIMESTAMP)) AS min_time, "
        f"MAX(TRY_CAST({qcol} AS TIMESTAMP)) AS max_time "
        f"FROM ({relation_sql}) rel"
    )
    if bounds.empty or pd.isna(bounds.iloc[0].get("min_time")):
        return {"is_static": False, "empty": True}
    monthly = access.fetchdf(
        f"SELECT STRFTIME(TRY_CAST({qcol} AS TIMESTAMP), '%Y-%m') AS bucket, COUNT(*) AS count "
        f"FROM ({relation_sql}) rel "
        f"WHERE TRY_CAST({qcol} AS TIMESTAMP) IS NOT NULL "
        "GROUP BY 1 ORDER BY 1"
    )
    return {
        "is_static": False,
        "min_time": pd.Timestamp(bounds.iloc[0]["min_time"]),
        "max_time": pd.Timestamp(bounds.iloc[0]["max_time"]),
        "coverage_by_month": {
            str(row["bucket"]): int(row["count"])
            for row in monthly.to_dict(orient="records")
            if row.get("bucket") is not None
        },
    }


def _select_correlation_columns(table_meta: TableMetadata, config: Any) -> list[str]:
    max_corr_columns = int(getattr(config, "max_corr_columns", 30))
    selected: list[str] = []
    for column_name in table_meta.columns:
        dtype_name = table_meta.dtypes.get(column_name, "")
        if is_text_dtype_name(dtype_name) and column_name.lower().endswith(("text", "body", "description")):
            continue
        selected.append(column_name)
        if len(selected) >= max_corr_columns:
            break
    return selected


def _build_correlations(
    *,
    sample_df: pd.DataFrame,
    table_name: str,
    sampled: bool,
    config: Any,
) -> list[dict[str, Any]]:
    if sample_df.empty or bool(getattr(config, "skip_correlations", False)):
        return []
    correlations: list[dict[str, Any]] = []

    numeric_cols = [c for c in sample_df.columns if pd.api.types.is_numeric_dtype(sample_df[c])]
    cat_cols = [c for c in sample_df.columns if c not in numeric_cols]

    num_subset = numeric_cols[: min(len(numeric_cols), int(getattr(config, "max_corr_numeric_columns", 20)))]
    if len(num_subset) >= 2:
        num_df = sample_df[num_subset].apply(pd.to_numeric, errors="coerce")
        corr_mat = num_df.corr(method="spearman", min_periods=30)
        notna = num_df.notna().astype(np.int8)
        valid_counts = notna.T.dot(notna)
        for i, col_a in enumerate(num_subset):
            for j in range(i + 1, len(num_subset)):
                col_b = num_subset[j]
                score = corr_mat.loc[col_a, col_b]
                if pd.isna(score):
                    continue
                sample_size = int(valid_counts.loc[col_a, col_b])
                if sample_size < 30:
                    continue
                correlations.append(
                    {
                        "type": "numeric_numeric_spearman",
                        "table": table_name,
                        "col_a": col_a,
                        "col_b": col_b,
                        "score": float(score),
                        "sampled": sampled,
                        "sample_size": sample_size,
                    }
                )

    max_cat_columns = int(getattr(config, "max_corr_categorical_columns", 15))
    for col_a, col_b in itertools.combinations(cat_cols[:max_cat_columns], 2):
        pair = sample_df[[col_a, col_b]].dropna()
        if len(pair) < 50:
            continue
        if pair[col_a].nunique(dropna=True) > 200 or pair[col_b].nunique(dropna=True) > 200:
            continue
        score = cramers_v(pair[col_a], pair[col_b])
        if score is None:
            continue
        correlations.append(
            {
                "type": "categorical_categorical_cramers_v",
                "table": table_name,
                "col_a": col_a,
                "col_b": col_b,
                "score": float(score),
                "sampled": sampled,
                "sample_size": int(len(pair)),
            }
        )

    max_cross_columns = int(getattr(config, "max_corr_cross_columns", 15))
    for num_col in numeric_cols[:max_cross_columns]:
        for cat_col in cat_cols[:max_cross_columns]:
            pair = sample_df[[num_col, cat_col]].dropna()
            if len(pair) < 50:
                continue
            if pair[cat_col].nunique(dropna=True) > 200:
                continue
            score = eta_squared(pair[num_col], pair[cat_col])
            if score is None:
                continue
            correlations.append(
                {
                    "type": "numeric_categorical_eta_squared",
                    "table": table_name,
                    "col_a": num_col,
                    "col_b": cat_col,
                    "score": float(score),
                    "sampled": sampled,
                    "sample_size": int(len(pair)),
                }
            )

    correlations.sort(key=lambda row: abs(float(row["score"])), reverse=True)
    return correlations[: int(getattr(config, "correlation_topk_per_table", 30))]


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    try:
        return float(value)
    except Exception:
        return default
