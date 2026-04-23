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
        contingency = pd.crosstab(a, b)
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
        valid = pd.DataFrame({"num": num, "cat": cat}).dropna()
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
