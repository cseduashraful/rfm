from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ALLOWED_TABLE_ROLES = {"entity", "event", "lookup", "bridge", "static"}
ALLOWED_COLUMN_ROLES = {"id", "timestamp", "measure", "status", "category", "text", "other"}
NUMERIC_DTYPE_TOKENS = ("tinyint", "smallint", "integer", "bigint", "hugeint", "float", "double", "decimal")
TEMPORAL_DTYPE_TOKENS = ("timestamp", "date", "time")
TEXT_DTYPE_TOKENS = ("varchar", "string", "text")
BOOLEAN_DTYPE_TOKENS = ("bool",)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        if pd.isna(value):
            return None
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=json_default)


def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def is_numeric_dtype_name(dtype_name: str) -> bool:
    low = str(dtype_name).lower()
    return any(tok in low for tok in NUMERIC_DTYPE_TOKENS)


def is_temporal_dtype_name(dtype_name: str) -> bool:
    low = str(dtype_name).lower()
    return any(tok in low for tok in TEMPORAL_DTYPE_TOKENS)


def is_text_dtype_name(dtype_name: str) -> bool:
    low = str(dtype_name).lower()
    return any(tok in low for tok in TEXT_DTYPE_TOKENS)


def is_boolean_dtype_name(dtype_name: str) -> bool:
    low = str(dtype_name).lower()
    return any(tok in low for tok in BOOLEAN_DTYPE_TOKENS)


def quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def quote_qualified(schema: str, name: str) -> str:
    return f"{quote_ident(schema)}.{quote_ident(name)}"


def sql_str_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def sql_timestamp_literal(value: Any) -> str:
    ts = pd.Timestamp(value)
    return f"TIMESTAMP {sql_str_literal(ts.isoformat())}"


def normalize_identifier_tokens(name: str) -> set[str]:
    low = "".join(ch if ch.isalnum() else " " for ch in str(name).lower()).strip()
    toks = {tok for tok in low.split() if len(tok) >= 2}
    expanded = set(toks)
    for tok in list(toks):
        if tok.endswith("s") and len(tok) > 3:
            expanded.add(tok[:-1])
        if tok.endswith("es") and len(tok) > 4:
            expanded.add(tok[:-2])
    return expanded


def singularize_identifier(name: str) -> str:
    low = str(name).lower()
    if low.endswith("ies") and len(low) > 4:
        return low[:-3] + "y"
    if low.endswith("ses") and len(low) > 4:
        return low[:-2]
    if low.endswith("s") and len(low) > 3:
        return low[:-1]
    return low


def to_hashable_cell(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return ("__ndarray__", tuple(to_hashable_cell(v) for v in value.tolist()))
    if isinstance(value, (list, tuple)):
        return ("__seq__", tuple(to_hashable_cell(v) for v in value))
    if isinstance(value, set):
        items = sorted((to_hashable_cell(v) for v in value), key=lambda x: repr(x))
        return ("__set__", tuple(items))
    if isinstance(value, dict):
        items = sorted(
            ((str(k), to_hashable_cell(v)) for k, v in value.items()),
            key=lambda x: x[0],
        )
        return ("__dict__", tuple(items))
    try:
        hash(value)
        return value
    except Exception:
        return repr(value)


def to_hashable_series(series: pd.Series) -> pd.Series:
    return series.map(to_hashable_cell)


def safe_nunique(series: pd.Series, dropna: bool = True) -> int:
    try:
        return int(series.nunique(dropna=dropna))
    except TypeError:
        clean = series.dropna() if dropna else series
        hashed = to_hashable_series(clean)
        return int(hashed.nunique(dropna=False))


def safe_mode(series: pd.Series) -> Any:
    try:
        mode = series.mode(dropna=True)
        if mode.empty:
            return None
        return mode.iloc[0]
    except Exception:
        return None


def cramers_v(a: pd.Series, b: pd.Series) -> float | None:
    try:
        contingency = pd.crosstab(to_hashable_series(a), to_hashable_series(b))
        if contingency.empty:
            return None
        observed = contingency.to_numpy(dtype=float)
        n = observed.sum()
        if n <= 0:
            return None
        row_sum = observed.sum(axis=1, keepdims=True)
        col_sum = observed.sum(axis=0, keepdims=True)
        expected = row_sum @ col_sum / n
        if (expected == 0).any():
            return None
        chi2 = ((observed - expected) ** 2 / expected).sum()
        r, c = observed.shape
        denom = min(r - 1, c - 1)
        if denom <= 0:
            return None
        return float(math.sqrt((chi2 / n) / denom))
    except Exception:
        return None


def eta_squared(num: pd.Series, cat: pd.Series) -> float | None:
    try:
        valid = pd.DataFrame({"num": num, "cat": to_hashable_series(cat)}).dropna()
        if valid.empty:
            return None
        grand_mean = valid["num"].mean()
        grouped = valid.groupby("cat")["num"]
        ss_between = sum(len(vals) * (vals.mean() - grand_mean) ** 2 for _, vals in grouped)
        ss_total = ((valid["num"] - grand_mean) ** 2).sum()
        if ss_total <= 0:
            return None
        return float(ss_between / ss_total)
    except Exception:
        return None
