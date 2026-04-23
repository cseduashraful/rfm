from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import os
import re
from threading import Lock
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


@contextmanager
def _suppress_tqdm_output():
    previous_disable = os.environ.get("TQDM_DISABLE")
    os.environ["TQDM_DISABLE"] = "1"

    tqdm_module = None
    original_tqdm = None
    try:
        import tqdm as tqdm_module  # type: ignore

        original_tqdm = tqdm_module.tqdm

        def _quiet_tqdm(*args, **kwargs):
            kwargs["disable"] = True
            return original_tqdm(*args, **kwargs)

        tqdm_module.tqdm = _quiet_tqdm
    except Exception:
        tqdm_module = None
        original_tqdm = None

    try:
        yield
    finally:
        if tqdm_module is not None and original_tqdm is not None:
            tqdm_module.tqdm = original_tqdm
        if previous_disable is None:
            os.environ.pop("TQDM_DISABLE", None)
        else:
            os.environ["TQDM_DISABLE"] = previous_disable


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    return False


def _format_value(value: Any) -> str:
    if _is_missing(value):
        return "missing"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        value_f = float(value)
        if abs(value_f) >= 1e6 or (abs(value_f) > 0 and abs(value_f) < 1e-3):
            return f"{value_f:.3e}"
        return f"{value_f:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, pd.Timestamp):
        return str(value)
    text = str(value)
    if len(text) > 80:
        return text[:77] + "..."
    return text


def _is_identifier(text: str) -> bool:
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", text))


def _parse_outer_function(expr: str) -> tuple[str, str] | None:
    expr = expr.strip()
    open_idx = expr.find("(")
    if open_idx <= 0 or not expr.endswith(")"):
        return None

    fn_name = expr[:open_idx].strip()
    if not _is_identifier(fn_name):
        return None

    depth = 0
    for idx, ch in enumerate(expr[open_idx:], start=open_idx):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and idx != len(expr) - 1:
                return None
    if depth != 0:
        return None
    return fn_name, expr[open_idx + 1 : -1].strip()


def _strip_root_prefix(column_name: str, root_table: str) -> str:
    prefix = f"{root_table}."
    if column_name.startswith(prefix):
        return column_name[len(prefix) :]
    if "." in column_name:
        return column_name.split(".", 1)[1]
    return column_name


@dataclass
class FastDFSContextBuilder:
    fastdfs_module: Any
    rdb: Any
    dfs_config: Any
    entity_col: str
    time_col: str
    root_table: str
    key_mappings: Dict[str, str]
    max_metapath_groups: int = 6
    max_features_per_metapath: int = 4
    strict_cutoff: bool = True
    training_window: pd.Timedelta | None = None
    training_window_key: str | None = None
    summary_cache: Dict[Tuple[str, int], List[str]] = field(default_factory=dict)
    feature_cache: Dict[Tuple[str, int], Dict[str, str]] = field(default_factory=dict)
    compute_lock: Any = field(default_factory=Lock, repr=False)
    table_names: set[str] = field(default_factory=set)

    def summarize_rows(self, rows: list[dict[str, Any]]) -> list[list[str]]:
        if not rows:
            return []

        output: list[list[str] | None] = [None] * len(rows)
        missing_keys: list[Tuple[str, int]] = []
        missing_rows: list[dict[str, Any]] = []
        seen_missing: set[Tuple[str, int]] = set()

        for row in rows:
            cache_key = self._cache_key(row)
            if cache_key in self.summary_cache or cache_key in seen_missing:
                continue
            seen_missing.add(cache_key)
            missing_keys.append(cache_key)
            missing_rows.append(row)

        if missing_rows:
            computed = self._compute_batch_summaries(missing_rows)
            for cache_key, summary_lines in zip(missing_keys, computed):
                self.summary_cache[cache_key] = summary_lines

        for idx, row in enumerate(rows):
            cache_key = self._cache_key(row)
            output[idx] = self.summary_cache.get(
                cache_key,
                ["- DFS summary unavailable for this timestamp."],
            )

        return [item if item is not None else [] for item in output]

    def feature_dicts_for_rows(self, rows: list[dict[str, Any]]) -> list[Dict[str, str]]:
        if not rows:
            return []

        output: list[Dict[str, str] | None] = [None] * len(rows)
        missing_keys: list[Tuple[str, int]] = []
        missing_rows: list[dict[str, Any]] = []
        seen_missing: set[Tuple[str, int]] = set()

        for row in rows:
            cache_key = self._cache_key(row)
            if cache_key in self.feature_cache or cache_key in seen_missing:
                continue
            seen_missing.add(cache_key)
            missing_keys.append(cache_key)
            missing_rows.append(row)

        if missing_rows:
            feature_df = self._compute_feature_frame(missing_rows).reset_index(drop=True)
            for cache_key, (_, feature_row) in zip(missing_keys, feature_df.iterrows()):
                self.feature_cache[cache_key] = self._feature_row_to_dict(feature_row)

        for idx, row in enumerate(rows):
            cache_key = self._cache_key(row)
            output[idx] = self.feature_cache.get(cache_key, {})

        return [item if item is not None else {} for item in output]

    def _cache_key(self, row: dict[str, Any]) -> Tuple[str, int]:
        entity_val = row[self.entity_col]
        ts = int(pd.Timestamp(row[self.time_col]).value)
        return str(entity_val), ts

    def _effective_cutoff(self, timestamp: Any) -> pd.Timestamp:
        cutoff = pd.Timestamp(timestamp)
        if not self.strict_cutoff:
            return cutoff
        try:
            return cutoff - pd.Timedelta(nanoseconds=1)
        except Exception:
            return cutoff

    def _compute_batch_summaries(self, rows: list[dict[str, Any]]) -> list[list[str]]:
        feature_df = self._compute_feature_frame(rows).reset_index(drop=True)

        summaries: list[list[str]] = []
        for idx in range(len(rows)):
            summaries.append(self._summarize_feature_row(feature_df.iloc[idx]))
        return summaries

    def _compute_feature_frame(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        target_records = []
        for row in rows:
            target_records.append(
                {
                    self.entity_col: row[self.entity_col],
                    self.time_col: self._effective_cutoff(row[self.time_col]),
                }
            )
        target_df = pd.DataFrame(target_records)

        with self.compute_lock:
            with _suppress_tqdm_output():
                feature_df = self.fastdfs_module.compute_dfs_features(
                    self.rdb,
                    target_df,
                    key_mappings=self.key_mappings,
                    cutoff_time_column=self.time_col,
                    config=self.dfs_config,
                )
        return feature_df

    def _feature_row_to_dict(self, feature_row: pd.Series) -> Dict[str, str]:
        output: Dict[str, str] = {}
        for feature_name, value in feature_row.items():
            if feature_name in {self.entity_col, self.time_col}:
                continue
            if _is_missing(value):
                continue
            output[str(feature_name)] = _format_value(value)
        return output

    def _summarize_feature_row(self, feature_row: pd.Series) -> list[str]:
        grouped: Dict[Tuple[str, ...], List[Tuple[str, Any, str]]] = {}

        for feature_name, value in feature_row.items():
            if feature_name in {self.entity_col, self.time_col}:
                continue
            if _is_missing(value):
                continue

            metapath, description = self._feature_to_metapath_and_description(feature_name)
            grouped.setdefault(metapath, []).append((description, value, feature_name))

        if not grouped:
            return ["- No DFS-derived signals available before this timestamp."]

        def _strip_within_suffixes(text: str) -> str:
            current = text.strip()
            while True:
                match = re.match(r"^(?P<head>.+) within [A-Za-z_][A-Za-z0-9_]*$", current)
                if match is None:
                    return current
                current = match.group("head").strip()

        def _simplify_metric_label(label: str) -> str:
            base = _strip_within_suffixes(label)
            while True:
                match = re.match(r"^(?P<agg>[a-z_]+) of (?P<target>.+)$", base)
                if match is None:
                    break
                agg = match.group("agg")
                target = match.group("target").strip()
                if agg == "count" and re.match(r"^rows in [A-Za-z_][A-Za-z0-9_]*$", target):
                    return "rows"
                if agg in {"max", "min", "mean", "std", "sum"}:
                    base = target
                    continue
                break
            return base

        lines: list[str] = []
        preferred_stat_order = ["max", "min", "mean", "std"]
        sorted_paths = sorted(grouped.keys(), key=lambda path: (len(path), path))
        for path in sorted_paths[: self.max_metapath_groups]:
            items = sorted(grouped[path], key=lambda item: item[2])
            lines.append(f"- Meta-path {' -> '.join(path)}:")

            rows_value: str | None = None
            per_column_stats: Dict[str, Dict[str, str]] = {}
            other_items: list[str] = []
            limited_items = items[: self.max_features_per_metapath]

            for description, value, _ in limited_items:
                match = re.match(r"^(?P<agg>[a-z_]+) of (?P<target>.+)$", description)
                if match is None:
                    other_items.append(f"{description}={_format_value(value)}")
                    continue

                agg = match.group("agg")
                target = match.group("target")

                if agg == "count" and re.match(r"^rows in [A-Za-z_][A-Za-z0-9_]*$", target):
                    rows_value = _format_value(value)
                    continue

                col_match = re.match(r"^(?P<col>.+) within [A-Za-z_][A-Za-z0-9_]*$", target)
                if col_match is None:
                    other_items.append(f"{description}={_format_value(value)}")
                    continue

                col_name = col_match.group("col").strip()
                if not col_name:
                    other_items.append(f"{description}={_format_value(value)}")
                    continue

                per_column_stats.setdefault(col_name, {})[agg] = _format_value(value)

            if rows_value is not None:
                lines.append(f"  - rows={rows_value}")

            for col_name in sorted(per_column_stats):
                stats = per_column_stats[col_name]
                rendered_parts = [f"{stat}={stats.get(stat, '?')}" for stat in preferred_stat_order]
                rendered_parts.extend(
                    f"{stat}={stats[stat]}" for stat in sorted(stats) if stat not in preferred_stat_order
                )
                simple_label = _simplify_metric_label(col_name)
                if simple_label == "rows":
                    row_stats = [stats.get(stat, "?") for stat in preferred_stat_order]
                    known_values = [val for val in row_stats if val != "?"]
                    if known_values and len(set(known_values)) == 1:
                        lines.append(f"  - rows = {known_values[0]}")
                    else:
                        lines.append(f"  - rows:[{', '.join(rendered_parts)}]")
                else:
                    lines.append(f"  - {simple_label}:[{', '.join(rendered_parts)}]")

            for other in other_items:
                lines.append(f"  - other: {other}")

            if not (rows_value is not None or per_column_stats or other_items):
                lines.append("  - No DFS features retained for this meta-path.")

            omitted = len(items) - len(limited_items)
            if omitted > 0:
                lines.append(f"  - ... {omitted} additional DFS features")

        omitted_paths = len(sorted_paths) - self.max_metapath_groups
        if omitted_paths > 0:
            lines.append(f"- ... {omitted_paths} additional meta-path groups")
        return lines

    def _feature_to_metapath_and_description(self, feature_name: str) -> tuple[Tuple[str, ...], str]:
        expr = _strip_root_prefix(feature_name, self.root_table)
        path_tables = self._extract_path_tables(expr)
        metapath = tuple([self.root_table] + path_tables) if path_tables else (self.root_table,)
        description = self._describe_expression(expr)
        return metapath, description

    def _extract_path_tables(self, expr: str) -> list[str]:
        collected: list[str] = []

        def walk(fragment: str) -> None:
            fragment = fragment.strip()
            if not fragment:
                return

            fn = _parse_outer_function(fragment)
            if fn is not None:
                _, inner = fn
                if inner in self.table_names:
                    collected.append(inner)
                    return
                walk(inner)
                return

            if "." in fragment:
                head, tail = fragment.split(".", 1)
                if head in self.table_names:
                    collected.append(head)
                walk(tail)

        walk(expr)

        deduped: list[str] = []
        for table_name in collected:
            if table_name not in deduped:
                deduped.append(table_name)
        return deduped

    def _describe_expression(self, expr: str) -> str:
        expr = expr.strip()
        if not expr:
            return "feature"

        fn = _parse_outer_function(expr)
        if fn is not None:
            fn_name, inner = fn
            return f"{fn_name.lower()} of {self._describe_expression(inner)}"

        if expr in self.table_names:
            return f"rows in {expr}"

        if "." in expr:
            head, tail = expr.split(".", 1)
            if head in self.table_names:
                return f"{self._describe_expression(tail)} within {head}"

        return expr.replace("_", " ")


def build_fastdfs_context_builder(
    resource,
    *,
    max_depth: int = 2,
    max_metapath_groups: int = 5,
    max_features_per_metapath: int = 100,
):
    try:
        import fastdfs
        from fastdfs import DFSConfig
    except ImportError as exc:
        raise ImportError(
            "fastdfs is required for --use-dfs. Install fastdfs (for example via the rfm env) "
            "and run zero_shot with that Python interpreter."
        ) from exc

    tables: Dict[str, pd.DataFrame] = {}
    primary_keys: Dict[str, str] = {}
    time_columns: Dict[str, str] = {}
    foreign_keys: list[tuple[str, str, str, str]] = []

    for table_name, table in resource.db.table_dict.items():
        df = table.df.copy()
        tables[table_name] = df

        if table.pkey_col is not None and table.pkey_col in df.columns:
            primary_keys[table_name] = table.pkey_col

        if table.time_col is not None and table.time_col in df.columns:
            time_columns[table_name] = table.time_col

        for fk_col, parent_table in table.fkey_col_to_pkey_table.items():
            if fk_col not in df.columns:
                continue
            parent = resource.db.table_dict.get(parent_table)
            if parent is None or parent.pkey_col is None:
                continue
            foreign_keys.append((table_name, fk_col, parent_table, parent.pkey_col))

    if resource.entity_table not in resource.db.table_dict:
        raise ValueError(
            f"Entity table {resource.entity_table!r} not found in DB table_dict."
        )

    entity_table_pk = resource.db.table_dict[resource.entity_table].pkey_col
    if entity_table_pk is None:
        raise ValueError(
            f"Entity table {resource.entity_table!r} has no primary key; cannot map DFS keys."
        )

    rdb = fastdfs.create_rdb(
        tables=tables,
        name=f"{resource.dataset}_{resource.task}",
        primary_keys=primary_keys,
        foreign_keys=foreign_keys,
        time_columns=time_columns,
    )

    # Use cumulative DFS features by default (no time-bounded training window).
    training_window = None
    active_window_key: str | None = None
    base_kwargs = {"max_depth": max_depth, "use_cutoff_time": True}
    dfs_config = DFSConfig(**base_kwargs)

    key_mappings = {resource.entity_col: f"{resource.entity_table}.{entity_table_pk}"}

    return FastDFSContextBuilder(
        fastdfs_module=fastdfs,
        rdb=rdb,
        dfs_config=dfs_config,
        entity_col=resource.entity_col,
        time_col=resource.time_col,
        root_table=resource.entity_table,
        key_mappings=key_mappings,
        max_metapath_groups=max_metapath_groups,
        max_features_per_metapath=max_features_per_metapath,
        strict_cutoff=True,
        training_window=training_window,
        training_window_key=active_window_key,
        table_names=set(resource.db.table_dict.keys()),
    )
