from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Ensure sibling project modules under RFM/code are importable when running from RFM/v2.
_RFM_ROOT = Path(__file__).resolve().parents[2]
_RFM_CODE_DIR = _RFM_ROOT / "code"
if _RFM_CODE_DIR.exists():
    sys.path.insert(0, str(_RFM_CODE_DIR))

from train_data import maybe_materialize_dataset
from relbench.tasks import get_task, get_task_names
from zero_shot_llm import LocalLLM, MODEL_PATHS
from phase1_utils import (
    ALLOWED_COLUMN_ROLES,
    ALLOWED_TABLE_ROLES,
    cramers_v as _cramers_v,
    eta_squared as _eta_squared,
    is_numeric as _is_numeric,
    safe_mode as _safe_mode,
    write_json as _write_json,
)
from phase1_fk import infer_fk_candidates
from phase1_semantic_graph import build_semantic_context_graph


def _to_hashable_cell(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return ("__ndarray__", tuple(_to_hashable_cell(v) for v in value.tolist()))
    if isinstance(value, (list, tuple)):
        return ("__seq__", tuple(_to_hashable_cell(v) for v in value))
    if isinstance(value, set):
        items = sorted((_to_hashable_cell(v) for v in value), key=lambda x: repr(x))
        return ("__set__", tuple(items))
    if isinstance(value, dict):
        items = sorted(
            ((str(k), _to_hashable_cell(v)) for k, v in value.items()),
            key=lambda x: x[0],
        )
        return ("__dict__", tuple(items))
    try:
        hash(value)
        return value
    except Exception:
        return repr(value)


def _safe_nunique(series: pd.Series, dropna: bool = True) -> int:
    try:
        return int(series.nunique(dropna=dropna))
    except TypeError:
        clean = series.dropna() if dropna else series
        hashed = clean.map(_to_hashable_cell)
        return int(hashed.nunique(dropna=False))


@dataclass
class Phase1Config:
    dataset: str = "rel-f1"
    output_dir: Path = Path("artifacts/v2/phase1")
    max_path_depth: int = 8
    sampling_enabled: bool = True
    sampling_row_threshold: int = 200_000
    sample_size: int = 100_000
    correlation_topk_per_table: int = 30
    correlation_global_cap: int = 500
    histogram_bins: int = 20
    categorical_topk: int = 20
    random_seed: int = 42
    artifact_version: str = "phase1.v1"
    max_paths: int = 50_000
    max_frontier_per_start: int = 5_000
    skip_correlations: bool = False
    use_llm_semantics: bool = False
    llm_model_size: str = "8b"
    llm_semantics_confidence_threshold: float = 0.6
    llm_semantics_use_ensemble: bool = True
    llm_semantics_max_new_tokens: int = 768
    use_llm_path_scoring: bool = False
    llm_path_top_k: int = 200
    compute_path_catalog: bool = False


class Phase1Pipeline:
    def __init__(self, config: Phase1Config):
        self.config = config
        self._llm: LocalLLM | None = None

    def run(self) -> dict[str, Path]:
        print("[phase1] loading dataset and computing global cutoff...")
        db = maybe_materialize_dataset(
            dataset_name=self.config.dataset,
            config={"dataset_download": True, "include_future_rows": False},
            no_dataset_download=False,
        )
        cutoff = self._compute_global_val_start(self.config.dataset)
        print(f"[phase1] global cutoff (val_start) = {cutoff}")
        filtered_tables = self._build_cutoff_filtered_tables(db, cutoff)

        print("[phase1] building schema artifact...")
        schema = self._build_schema_artifact(db, filtered_tables, cutoff)
        print("[phase1] building stats artifact...")
        stats = self._build_stats_artifact(db, filtered_tables, cutoff)
        print("[phase1] building semantics artifact...")
        semantics = self._build_semantics_artifact(db, filtered_tables, cutoff)
        print("[phase1] building semantic context graph artifact...")
        semantic_context_graph = self._build_semantic_context_graph_artifact(
            db=db,
            filtered=filtered_tables,
            cutoff=cutoff,
            schema_artifact=schema,
            stats_artifact=stats,
            semantics_artifact=semantics,
        )
        path_catalog: dict[str, Any] | None = None
        if self.config.compute_path_catalog:
            print("[phase1] building path catalog artifact...")
            path_catalog = self._build_path_catalog_artifact(db, filtered_tables, cutoff)
        else:
            print("[phase1] skipping path catalog artifact (enable with --compute-path-catalog).")
        print("[phase1] building safety artifact...")
        safety = self._build_safety_artifact(cutoff)

        base = self.config.output_dir / self.config.dataset
        paths = {
            "schema": base / "schema.json",
            "stats": base / "stats.json",
            "semantics": base / "semantics.json",
            "semantic_context_graph": base / "semantic_context_graph.json",
            "safety_rules": base / "safety_rules.json",
        }
        if self.config.compute_path_catalog:
            paths["path_catalog"] = base / "path_catalog.json"
        _write_json(paths["schema"], schema)
        _write_json(paths["stats"], stats)
        _write_json(paths["semantics"], semantics)
        _write_json(paths["semantic_context_graph"], semantic_context_graph)
        if self.config.compute_path_catalog and path_catalog is not None:
            _write_json(paths["path_catalog"], path_catalog)
        _write_json(paths["safety_rules"], safety)
        return paths

    def _get_llm(self) -> LocalLLM:
        if self._llm is None:
            model_path = MODEL_PATHS[self.config.llm_model_size]
            print(f"[phase1] loading LLM semantics model: size={self.config.llm_model_size}")
            self._llm = LocalLLM(model_path, print_log=True)
        return self._llm

    def _artifact_header(self, cutoff: pd.Timestamp) -> dict[str, Any]:
        return {
            "artifact_version": self.config.artifact_version,
            "dataset_name": self.config.dataset,
            "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "global_cutoff_time": cutoff,
        }

    def _compute_global_val_start(self, dataset: str) -> pd.Timestamp:
        val_starts: list[pd.Timestamp] = []
        for task_name in get_task_names(dataset):
            try:
                task = get_task(dataset, task_name, download=True)
                val_table = task.get_table("val", mask_input_cols=False)
                time_col = getattr(task, "time_col", val_table.time_col)
                if time_col is None or time_col not in val_table.df.columns:
                    continue
                ts = pd.to_datetime(val_table.df[time_col], errors="coerce").dropna()
                if not ts.empty:
                    val_starts.append(pd.Timestamp(ts.min()))
            except Exception:
                continue
        if not val_starts:
            raise RuntimeError(
                f"Could not determine global val_start cutoff for dataset={dataset}."
            )
        return min(val_starts)

    def _build_cutoff_filtered_tables(
        self,
        db: Any,
        cutoff: pd.Timestamp,
    ) -> dict[str, pd.DataFrame]:
        filtered: dict[str, pd.DataFrame] = {}
        for table_name, table in db.table_dict.items():
            frame = table.df.copy()
            time_col = table.time_col
            if time_col is not None and time_col in frame.columns:
                ts = pd.to_datetime(frame[time_col], errors="coerce")
                mask = ts.notna() & (ts < cutoff)
                frame = frame.loc[mask].copy()
            filtered[table_name] = frame
        return filtered

    def _infer_pk_candidates(self, frame: pd.DataFrame, table_name: str) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        n = len(frame)
        if n == 0:
            return candidates
        for col in frame.columns:
            non_null = frame[col].notna().sum()
            unique = _safe_nunique(frame[col], dropna=True)
            if unique == n and non_null == n:
                candidates.append({
                    "columns": [col],
                    "confidence": "high",
                    "reason": "unique_and_non_null",
                })
            elif unique >= int(0.995 * n) and non_null >= int(0.995 * n):
                candidates.append({
                    "columns": [col],
                    "confidence": "medium",
                    "reason": "near_unique",
                })
        id_like = [c for c in frame.columns if c.lower().endswith("id")]
        for c1, c2 in itertools.combinations(id_like[:12], 2):
            combo = frame[[c1, c2]].dropna()
            if combo.empty:
                continue
            unique = len(combo.drop_duplicates())
            if unique == len(combo) and len(combo) >= int(0.95 * n):
                candidates.append({
                    "columns": [c1, c2],
                    "confidence": "medium",
                    "reason": "composite_unique",
                })
        return candidates

    def _infer_fk_candidates(self, db: Any, filtered: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
        return infer_fk_candidates(db, filtered)

    def _build_schema_artifact(self, db: Any, filtered: dict[str, pd.DataFrame], cutoff: pd.Timestamp) -> dict[str, Any]:
        header = self._artifact_header(cutoff)
        inferred_fk = self._infer_fk_candidates(db, filtered)
        tables = []
        for table_name, table in db.table_dict.items():
            frame = filtered[table_name]
            cols = []
            for col in frame.columns:
                cols.append(
                    {
                        "name": col,
                        "dtype": str(frame[col].dtype),
                        "nullable": bool(frame[col].isna().any()),
                        "n_unique": _safe_nunique(frame[col], dropna=True),
                    }
                )
            tables.append(
                {
                    "table_name": table_name,
                    "row_count": int(len(frame)),
                    "provided_primary_key": table.pkey_col,
                    "provided_foreign_keys": dict(table.fkey_col_to_pkey_table),
                    "time_column": table.time_col,
                    "columns": cols,
                    "inferred_pk_candidates": self._infer_pk_candidates(frame, table_name),
                }
            )
        return {
            **header,
            "tables": tables,
            "inferred_foreign_key_candidates": inferred_fk,
        }

    def _sample_if_needed(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        if (not self.config.sampling_enabled) or len(frame) <= self.config.sampling_row_threshold:
            return frame, False
        sample_n = min(self.config.sample_size, len(frame))
        sampled = frame.sample(n=sample_n, random_state=self.config.random_seed)
        return sampled, True

    def _numeric_stats(self, series: pd.Series) -> dict[str, Any]:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if clean.empty:
            return {}
        q = clean.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        hist_counts, hist_edges = np.histogram(clean.to_numpy(), bins=self.config.histogram_bins)
        return {
            "min": float(clean.min()),
            "max": float(clean.max()),
            "mean": float(clean.mean()),
            "std": float(clean.std(ddof=1)) if len(clean) > 1 else 0.0,
            "median": float(clean.median()),
            "mode": _safe_mode(clean),
            "quantiles": {
                "p10": float(q.loc[0.1]),
                "p25": float(q.loc[0.25]),
                "p50": float(q.loc[0.5]),
                "p75": float(q.loc[0.75]),
                "p90": float(q.loc[0.9]),
            },
            "histogram": {
                "counts": hist_counts.tolist(),
                "bin_edges": hist_edges.tolist(),
            },
        }

    def _categorical_stats(self, series: pd.Series) -> dict[str, Any]:
        clean = series.dropna()
        if clean.empty:
            return {}
        vc = clean.value_counts(dropna=True).head(self.config.categorical_topk)
        return {
            "cardinality": _safe_nunique(clean, dropna=True),
            "mode": _safe_mode(clean),
            "top_values": [{"value": str(idx), "count": int(val)} for idx, val in vc.items()],
        }

    def _table_temporal_coverage(self, frame: pd.DataFrame, time_col: str | None) -> dict[str, Any]:
        if time_col is None or time_col not in frame.columns:
            return {"is_static": True}
        ts = pd.to_datetime(frame[time_col], errors="coerce").dropna()
        if ts.empty:
            return {"is_static": False, "empty": True}
        monthly = ts.dt.to_period("M").astype(str).value_counts().sort_index()
        return {
            "is_static": False,
            "min_time": pd.Timestamp(ts.min()),
            "max_time": pd.Timestamp(ts.max()),
            "coverage_by_month": {k: int(v) for k, v in monthly.items()},
        }

    def _build_correlations_for_table(self, frame: pd.DataFrame, table_name: str) -> list[dict[str, Any]]:
        if self.config.skip_correlations:
            return []
        sampled_df, sampled = self._sample_if_needed(frame)
        correlations: list[dict[str, Any]] = []

        numeric_cols = [c for c in sampled_df.columns if _is_numeric(sampled_df[c])]
        cat_cols = [c for c in sampled_df.columns if not _is_numeric(sampled_df[c])]

        for a, b in itertools.combinations(numeric_cols[:40], 2):
            pair = sampled_df[[a, b]].dropna()
            if len(pair) < 30:
                continue
            val = pair[a].corr(pair[b], method="spearman")
            if pd.isna(val):
                continue
            correlations.append(
                {
                    "type": "numeric_numeric_spearman",
                    "table": table_name,
                    "col_a": a,
                    "col_b": b,
                    "score": float(val),
                    "sampled": sampled,
                    "sample_size": int(len(pair)),
                }
            )

        for a, b in itertools.combinations(cat_cols[:25], 2):
            pair = sampled_df[[a, b]].dropna()
            if len(pair) < 50:
                continue
            if _safe_nunique(pair[a], dropna=True) > 200 or _safe_nunique(pair[b], dropna=True) > 200:
                continue
            val = _cramers_v(pair[a], pair[b])
            if val is None:
                continue
            correlations.append(
                {
                    "type": "categorical_categorical_cramers_v",
                    "table": table_name,
                    "col_a": a,
                    "col_b": b,
                    "score": float(val),
                    "sampled": sampled,
                    "sample_size": int(len(pair)),
                }
            )

        for num_col in numeric_cols[:20]:
            for cat_col in cat_cols[:20]:
                pair = sampled_df[[num_col, cat_col]].dropna()
                if len(pair) < 50:
                    continue
                if _safe_nunique(pair[cat_col], dropna=True) > 200:
                    continue
                val = _eta_squared(pair[num_col], pair[cat_col])
                if val is None:
                    continue
                correlations.append(
                    {
                        "type": "numeric_categorical_eta_squared",
                        "table": table_name,
                        "col_a": num_col,
                        "col_b": cat_col,
                        "score": float(val),
                        "sampled": sampled,
                        "sample_size": int(len(pair)),
                    }
                )

        correlations.sort(key=lambda x: abs(float(x["score"])), reverse=True)
        return correlations[: self.config.correlation_topk_per_table]

    def _build_stats_artifact(self, db: Any, filtered: dict[str, pd.DataFrame], cutoff: pd.Timestamp) -> dict[str, Any]:
        header = self._artifact_header(cutoff)
        tables = []
        all_corrs: list[dict[str, Any]] = []

        for table_name, table in db.table_dict.items():
            frame = filtered[table_name]
            sampled_frame, sampled = self._sample_if_needed(frame)
            col_stats: dict[str, Any] = {}
            for col in frame.columns:
                full_col = frame[col]
                prof_col = sampled_frame[col] if col in sampled_frame.columns else full_col
                base = {
                    "missingness": float(full_col.isna().mean()) if len(full_col) else None,
                    "non_null_count": int(full_col.notna().sum()),
                    "sampled": sampled,
                    "profile_sample_size": int(len(prof_col)),
                }
                if _is_numeric(full_col):
                    base["numeric"] = self._numeric_stats(prof_col)
                else:
                    base["categorical"] = self._categorical_stats(prof_col)
                col_stats[col] = base

            table_corrs = self._build_correlations_for_table(frame, table_name)
            all_corrs.extend(table_corrs)

            tables.append(
                {
                    "table_name": table_name,
                    "row_count": int(len(frame)),
                    "sampled": sampled,
                    "temporal_coverage": self._table_temporal_coverage(frame, table.time_col),
                    "columns": col_stats,
                    "correlations": table_corrs,
                }
            )

        all_corrs.sort(key=lambda x: abs(float(x["score"])), reverse=True)
        all_corrs = all_corrs[: self.config.correlation_global_cap]

        return {
            **header,
            "tables": tables,
            "global_top_correlations": all_corrs,
        }

    def _table_role(self, table_name: str, table_obj: Any) -> str:
        lower = table_name.lower()
        if table_obj.time_col is None:
            return "static"
        if lower.endswith("s") and lower not in {"status", "news"}:
            return "event"
        return "entity"

    def _column_role(self, col: str, dtype: str) -> str:
        c = col.lower()
        if c.endswith("id") or c == "id":
            return "id"
        if "date" in c or "time" in c or "timestamp" in c:
            return "timestamp"
        if any(tok in c for tok in ["status", "type", "category"]):
            return "category"
        if any(tok in c for tok in ["count", "mean", "std", "sum", "max", "min", "score", "value", "amount", "points"]):
            return "measure"
        if "object" in dtype or "string" in dtype:
            return "category"
        return "measure"

    def _build_semantics_artifact(self, db: Any, filtered: dict[str, pd.DataFrame], cutoff: pd.Timestamp) -> dict[str, Any]:
        header = self._artifact_header(cutoff)
        tables = []
        leakage_patterns = ["label", "target", "future", "leak", "outcome"]
        llm_used = False
        llm_failures = 0
        llm_diagnostics: list[dict[str, Any]] = []
        llm = self._get_llm() if self.config.use_llm_semantics else None
        for table_name, table in db.table_dict.items():
            frame = filtered[table_name]
            table_role = self._table_role(table_name, table)
            columns = []
            llm_result: dict[str, Any] | None = None
            if llm is not None:
                llm_result, diag = self._annotate_table_semantics_with_llm(
                    llm=llm,
                    table_name=table_name,
                    frame=frame,
                    provided_time_col=table.time_col,
                )
                llm_diagnostics.append(diag)
                if llm_result is not None:
                    llm_used = True
                else:
                    llm_failures += 1

            if llm_result is not None:
                llm_table_role = self._normalize_table_role(llm_result.get("table_role"), table_role)
                table_role_conf = float(llm_result.get("table_role_confidence", 0.0))
                table_role = llm_table_role if table_role_conf >= self.config.llm_semantics_confidence_threshold else table_role
                llm_columns = llm_result.get("columns", {})
                for col in frame.columns:
                    col_info = llm_columns.get(col, {})
                    heuristic_role = self._column_role(col, str(frame[col].dtype))
                    llm_role = self._normalize_column_role(col_info.get("semantic_role"), heuristic_role)
                    llm_role = self._semantic_rule_override(col, llm_role, heuristic_role)
                    llm_conf = float(col_info.get("confidence", 0.0))
                    llm_abstain = bool(col_info.get("abstain", False))
                    if llm_abstain or llm_conf < self.config.llm_semantics_confidence_threshold:
                        role = heuristic_role
                        confidence = "merged_low_conf_fallback"
                    else:
                        role = llm_role
                        confidence = "llm"
                    leakage_risk = bool(col_info.get("leakage_risk", any(p in col.lower() for p in leakage_patterns)))
                    columns.append(
                        {
                            "name": col,
                            "semantic_role": role,
                            "leakage_risk": leakage_risk,
                            "confidence": confidence,
                            "llm_confidence": llm_conf,
                            "llm_abstain": llm_abstain,
                            "heuristic_prior_role": heuristic_role,
                        }
                    )
            else:
                for col in frame.columns:
                    role = self._column_role(col, str(frame[col].dtype))
                    col_l = col.lower()
                    leakage_risk = any(p in col_l for p in leakage_patterns)
                    columns.append(
                        {
                            "name": col,
                            "semantic_role": role,
                            "leakage_risk": leakage_risk,
                            "confidence": "heuristic",
                            "llm_confidence": None,
                            "llm_abstain": None,
                            "heuristic_prior_role": role,
                        }
                    )
            tables.append(
                {
                    "table_name": table_name,
                    "table_role": table_role,
                    "time_column": table.time_col,
                    "columns": columns,
                }
            )
        validation = self._validate_semantics_tables(tables)
        return {
            **header,
            "annotation_model": {
                "name": f"local-llama-{self.config.llm_model_size}" if llm_used else "heuristic-baseline",
                "llm_used": llm_used,
                "llm_failures": llm_failures,
                "determinism": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "do_sample": False,
                    "max_retries_on_parse_failure": 2,
                },
                "notes": (
                    "LLM semantic annotation enabled with deterministic generation and JSON parsing."
                    if llm_used
                    else "Heuristic semantic baseline. Enable --use-llm-semantics to use local Llama."
                ),
            },
            "llm_diagnostics": llm_diagnostics,
            "validation": validation,
            "tables": tables,
        }

    def _validate_semantics_tables(self, tables: list[dict[str, Any]]) -> dict[str, Any]:
        issues: list[dict[str, Any]] = []
        for table in tables:
            tname = table.get("table_name", "")
            trole = str(table.get("table_role", ""))
            if trole not in ALLOWED_TABLE_ROLES:
                issues.append({"type": "invalid_table_role", "table": tname, "value": trole})
            for col in table.get("columns", []):
                cname = col.get("name", "")
                role = str(col.get("semantic_role", ""))
                if role not in ALLOWED_COLUMN_ROLES:
                    issues.append({"type": "invalid_column_role", "table": tname, "column": cname, "value": role})
                low = str(cname).lower()
                if any(tok in low for tok in ["date", "time", "timestamp"]) and role != "timestamp":
                    issues.append({"type": "timestamp_name_mismatch", "table": tname, "column": cname, "value": role})
                if low.endswith("id") and role != "id":
                    issues.append({"type": "id_name_mismatch", "table": tname, "column": cname, "value": role})
        return {
            "issue_count": len(issues),
            "issues": issues[:500],
            "passed": len(issues) == 0,
        }

    def _extract_json_object(self, text: str) -> dict[str, Any] | None:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        chunk = text[start : end + 1]
        try:
            return json.loads(chunk)
        except Exception:
            return None

    def _normalize_table_role(self, role: Any, fallback: str) -> str:
        if role is None:
            return fallback
        role_s = str(role).strip().lower()
        if role_s in ALLOWED_TABLE_ROLES:
            return role_s
        # Repair common malformed cases like "entity|event".
        for token in re.split(r"[^a-z]+", role_s):
            if token in ALLOWED_TABLE_ROLES:
                return token
        return fallback

    def _normalize_column_role(self, role: Any, fallback: str) -> str:
        if role is None:
            return fallback
        role_s = str(role).strip().lower()
        if role_s in ALLOWED_COLUMN_ROLES:
            return role_s
        for token in re.split(r"[^a-z]+", role_s):
            if token in ALLOWED_COLUMN_ROLES:
                return token
        return fallback

    def _column_prior_info(self, series: pd.Series, col_name: str) -> dict[str, Any]:
        non_null = int(series.notna().sum())
        n = len(series)
        unique = _safe_nunique(series, dropna=True)
        prior_role = self._column_role(col_name, str(series.dtype))
        top_values = []
        try:
            top = series.dropna().astype(str).value_counts().head(5)
            top_values = [{"value": k, "count": int(v)} for k, v in top.items()]
        except Exception:
            top_values = []
        return {
            "name": col_name,
            "dtype": str(series.dtype),
            "non_null": non_null,
            "missing_ratio": float(series.isna().mean()) if n else None,
            "unique": unique,
            "unique_ratio": float(unique / max(1, non_null)) if non_null else 0.0,
            "heuristic_prior_role": prior_role,
            "top_values": top_values,
        }

    def _semantic_rule_override(self, col_name: str, role: str, heuristic_role: str) -> str:
        c = col_name.lower()
        # Hard safety-style overrides for obvious tokens.
        if c.endswith("id") and role != "id":
            return "id"
        if any(tok in c for tok in ["date", "time", "timestamp"]) and role != "timestamp":
            return "timestamp"
        if role not in ALLOWED_COLUMN_ROLES:
            return heuristic_role
        return role

    def _build_semantics_prompt(
        self,
        *,
        table_name: str,
        provided_time_col: str | None,
        col_specs: list[dict[str, Any]],
        variant: str = "primary",
    ) -> str:
        lead = "Use heuristic priors as guidance, not absolute truth."
        if variant == "alt":
            lead = "Focus on consistency between names, dtypes, and priors. If uncertain, abstain."
        return (
            "You are labeling relational database semantics.\n"
            f"{lead}\n"
            "Return STRICT JSON only with exactly this structure:\n"
            "{\n"
            '  "table_role": "entity|event|lookup|bridge|static",\n'
            '  "table_role_confidence": <float 0..1>,\n'
            '  "columns": {\n'
            '    "<col_name>": {\n'
            '      "semantic_role": "id|timestamp|measure|status|category|text|other",\n'
            '      "leakage_risk": true|false,\n'
            '      "confidence": <float 0..1>,\n'
            '      "abstain": true|false\n'
            "    }\n"
            "  }\n"
            "}\n"
            "No markdown, no prose.\n\n"
            f"Table name: {table_name}\n"
            f"Provided time column: {provided_time_col}\n"
            f"Columns with priors:\n{json.dumps(col_specs, ensure_ascii=True)}\n"
        )

    def _build_semantics_critic_prompt(self, candidate_json: dict[str, Any], table_name: str) -> str:
        return (
            "You are a critic for relational semantic labels.\n"
            "Check internal consistency and repair invalid enums.\n"
            "Allowed table_role: entity,event,lookup,bridge,static.\n"
            "Allowed semantic_role: id,timestamp,measure,status,category,text,other.\n"
            "Return STRICT JSON in the same schema as input. No markdown.\n\n"
            f"Table: {table_name}\n"
            f"Candidate JSON:\n{json.dumps(candidate_json, ensure_ascii=True)}\n"
        )

    def _sanitize_llm_semantics_result(
        self,
        *,
        parsed: dict[str, Any],
        table_name: str,
        frame: pd.DataFrame,
        heuristic_table_role: str,
    ) -> tuple[dict[str, Any], list[str]]:
        issues: list[str] = []
        out: dict[str, Any] = {"columns": {}}

        out["table_role"] = self._normalize_table_role(parsed.get("table_role"), heuristic_table_role)
        if out["table_role"] != str(parsed.get("table_role", "")).strip().lower():
            issues.append(f"{table_name}: repaired_table_role")
        trc = parsed.get("table_role_confidence", 0.0)
        try:
            out["table_role_confidence"] = max(0.0, min(1.0, float(trc)))
        except Exception:
            out["table_role_confidence"] = 0.0
            issues.append(f"{table_name}: invalid_table_role_confidence")

        llm_columns = parsed.get("columns", {})
        if not isinstance(llm_columns, dict):
            llm_columns = {}
            issues.append(f"{table_name}: invalid_columns_object")

        for col in frame.columns:
            heuristic_role = self._column_role(col, str(frame[col].dtype))
            cobj = llm_columns.get(col, {})
            if not isinstance(cobj, dict):
                cobj = {}
                issues.append(f"{table_name}.{col}: invalid_column_payload")
            role = self._normalize_column_role(cobj.get("semantic_role"), heuristic_role)
            role = self._semantic_rule_override(col, role, heuristic_role)
            try:
                conf = float(cobj.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            conf = max(0.0, min(1.0, conf))
            abstain = bool(cobj.get("abstain", False))
            leakage_risk = bool(cobj.get("leakage_risk", any(p in col.lower() for p in ["label", "target", "future", "leak", "outcome"])))
            out["columns"][col] = {
                "semantic_role": role,
                "confidence": conf,
                "abstain": abstain,
                "leakage_risk": leakage_risk,
            }
        return out, issues

    def _annotate_table_semantics_with_llm(
        self,
        *,
        llm: LocalLLM,
        table_name: str,
        frame: pd.DataFrame,
        provided_time_col: str | None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        diagnostics: dict[str, Any] = {
            "table_name": table_name,
            "llm_solver_ok": False,
            "llm_critic_ok": False,
            "llm_ensemble_used": False,
            "issues": [],
        }
        heuristic_table_role = self._table_role(table_name, type("T", (), {"time_col": provided_time_col})())
        col_specs = [self._column_prior_info(frame[col], col) for col in frame.columns]

        def _run_prompt(prompt: str) -> dict[str, Any] | None:
            for _ in range(3):
                try:
                    out = llm.generate_batch([prompt], max_new_tokens=self.config.llm_semantics_max_new_tokens)[0]
                    parsed = self._extract_json_object(out)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
            return None

        # Pass 1: solver
        primary_prompt = self._build_semantics_prompt(
            table_name=table_name,
            provided_time_col=provided_time_col,
            col_specs=col_specs,
            variant="primary",
        )
        parsed_primary = _run_prompt(primary_prompt)
        if not isinstance(parsed_primary, dict):
            diagnostics["issues"].append("solver_parse_failed")
            return None, diagnostics
        diagnostics["llm_solver_ok"] = True
        primary_sanitized, primary_issues = self._sanitize_llm_semantics_result(
            parsed=parsed_primary,
            table_name=table_name,
            frame=frame,
            heuristic_table_role=heuristic_table_role,
        )
        diagnostics["issues"].extend(primary_issues)

        # Pass 2: critic repair
        critic_prompt = self._build_semantics_critic_prompt(primary_sanitized, table_name)
        parsed_critic = _run_prompt(critic_prompt)
        final_sanitized = primary_sanitized
        if isinstance(parsed_critic, dict):
            diagnostics["llm_critic_ok"] = True
            critic_sanitized, critic_issues = self._sanitize_llm_semantics_result(
                parsed=parsed_critic,
                table_name=table_name,
                frame=frame,
                heuristic_table_role=heuristic_table_role,
            )
            diagnostics["issues"].extend(critic_issues)
            final_sanitized = critic_sanitized

        # Light ensemble only for ambiguous cases.
        ambiguity_score = 0
        for col in frame.columns:
            h = self._column_role(col, str(frame[col].dtype))
            r = final_sanitized["columns"][col]["semantic_role"]
            c = float(final_sanitized["columns"][col]["confidence"])
            abstain = bool(final_sanitized["columns"][col]["abstain"])
            if abstain or c < self.config.llm_semantics_confidence_threshold or r != h:
                ambiguity_score += 1
        if (
            self.config.llm_semantics_use_ensemble
            and ambiguity_score > max(2, int(0.2 * max(1, len(frame.columns))))
        ):
            alt_prompt = self._build_semantics_prompt(
                table_name=table_name,
                provided_time_col=provided_time_col,
                col_specs=col_specs,
                variant="alt",
            )
            parsed_alt = _run_prompt(alt_prompt)
            if isinstance(parsed_alt, dict):
                diagnostics["llm_ensemble_used"] = True
                alt_sanitized, alt_issues = self._sanitize_llm_semantics_result(
                    parsed=parsed_alt,
                    table_name=table_name,
                    frame=frame,
                    heuristic_table_role=heuristic_table_role,
                )
                diagnostics["issues"].extend(alt_issues)
                # Majority vote: primary/critic result + alt + heuristic tie-breaker.
                role_votes = [
                    final_sanitized.get("table_role", heuristic_table_role),
                    alt_sanitized.get("table_role", heuristic_table_role),
                    heuristic_table_role,
                ]
                final_sanitized["table_role"] = Counter(role_votes).most_common(1)[0][0]
                for col in frame.columns:
                    votes = [
                        final_sanitized["columns"][col]["semantic_role"],
                        alt_sanitized["columns"][col]["semantic_role"],
                        self._column_role(col, str(frame[col].dtype)),
                    ]
                    voted = Counter(votes).most_common(1)[0][0]
                    final_sanitized["columns"][col]["semantic_role"] = voted
                    final_sanitized["columns"][col]["confidence"] = max(
                        float(final_sanitized["columns"][col].get("confidence", 0.0)),
                        float(alt_sanitized["columns"][col].get("confidence", 0.0)),
                    )
                    final_sanitized["columns"][col]["abstain"] = bool(
                        final_sanitized["columns"][col].get("abstain", False)
                    ) and bool(alt_sanitized["columns"][col].get("abstain", False))

        return final_sanitized, diagnostics

    def _build_path_catalog_artifact(self, db: Any, filtered: dict[str, pd.DataFrame], cutoff: pd.Timestamp) -> dict[str, Any]:
        header = self._artifact_header(cutoff)

        fk_edges: list[dict[str, Any]] = []
        for child_name, table in db.table_dict.items():
            child_df = filtered[child_name]
            for fk_col, parent_name in table.fkey_col_to_pkey_table.items():
                if fk_col not in child_df.columns:
                    continue
                parent = db.table_dict.get(parent_name)
                if parent is None or parent.pkey_col is None:
                    continue
                parent_df = filtered[parent_name]
                if parent.pkey_col not in parent_df.columns:
                    continue
                cvals = set(child_df[fk_col].dropna().unique().tolist())
                pvals = set(parent_df[parent.pkey_col].dropna().unique().tolist())
                overlap = len(cvals & pvals)
                ratio = overlap / max(1, len(cvals)) if cvals else 0.0
                confidence = "high" if ratio >= 0.98 else ("medium" if ratio >= 0.9 else "low")
                fk_edges.append(
                    {
                        "child_table": child_name,
                        "child_column": fk_col,
                        "parent_table": parent_name,
                        "parent_column": parent.pkey_col,
                        "source": "provided",
                        "join_types": ["inner", "left"],
                        "confidence": confidence,
                        "recommended_by_default": confidence != "low",
                        "overlap_ratio": ratio,
                    }
                )

        inferred_edges = self._infer_fk_candidates(db, filtered)
        existing_keys = {
            (
                e["child_table"],
                e["child_column"],
                e["parent_table"],
                e["parent_column"],
            )
            for e in fk_edges
        }
        for edge in inferred_edges:
            edge_key = (
                edge["child_table"],
                edge["child_column"],
                edge["parent_table"],
                edge["parent_column"],
            )
            if edge_key in existing_keys:
                continue
            fk_edges.append(
                {
                    **edge,
                    "source": "inferred",
                    "join_types": ["inner", "left"],
                }
            )

        graph: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for edge in fk_edges:
            graph.setdefault(edge["child_table"], []).append((edge["parent_table"], edge))

        paths = []
        truncated = False
        for start in sorted(graph.keys()):
            frontier = [([start], [])]
            expanded = 0
            while frontier:
                nodes, used_edges = frontier.pop()
                expanded += 1
                if expanded > self.config.max_frontier_per_start:
                    truncated = True
                    break
                if len(nodes) - 1 >= self.config.max_path_depth:
                    continue
                last = nodes[-1]
                for nxt, edge in graph.get(last, []):
                    if nxt in nodes:
                        continue
                    new_nodes = nodes + [nxt]
                    new_edges = used_edges + [edge]
                    fanouts = []
                    for e in new_edges:
                        child_rows = len(filtered[e["child_table"]])
                        child_unique = max(1, _safe_nunique(filtered[e["child_table"]][e["child_column"]], dropna=True))
                        fanouts.append(child_rows / child_unique)
                    estimated_multiplier = float(np.prod(fanouts)) if fanouts else 1.0
                    recommended = all(e.get("recommended_by_default", True) for e in new_edges)
                    paths.append(
                        {
                            "path_tables": new_nodes,
                            "path_edges": new_edges,
                            "depth": len(new_nodes) - 1,
                            "estimated_row_multiplier": estimated_multiplier,
                            "recommended": recommended,
                            "temporal_valid": True,
                        }
                    )
                    if len(paths) >= self.config.max_paths:
                        truncated = True
                        break
                    frontier.append((new_nodes, new_edges))
                if truncated:
                    break
            if truncated:
                break

        paths.sort(key=lambda x: (x["depth"], x["estimated_row_multiplier"], x["path_tables"]))

        path_scoring_report = {
            "llm_used": False,
            "scored_count": 0,
            "top_k": self.config.llm_path_top_k,
            "failures": 0,
        }
        if self.config.use_llm_path_scoring and self.config.use_llm_semantics and paths:
            llm = self._get_llm()
            score_count = min(self.config.llm_path_top_k, len(paths))
            scored = 0
            failures = 0
            for idx in range(score_count):
                path_obj = paths[idx]
                scored_obj = self._score_path_with_llm(llm, path_obj)
                if scored_obj is None:
                    failures += 1
                    continue
                path_obj.update(scored_obj)
                scored += 1
            path_scoring_report = {
                "llm_used": True,
                "scored_count": scored,
                "top_k": self.config.llm_path_top_k,
                "failures": failures,
            }

        return {
            **header,
            "max_path_depth": self.config.max_path_depth,
            "max_paths_cap": self.config.max_paths,
            "truncated": truncated,
            "path_scoring_report": path_scoring_report,
            "foreign_key_edges": fk_edges,
            "paths": paths,
        }

    def _score_path_with_llm(self, llm: LocalLLM, path_obj: dict[str, Any]) -> dict[str, Any] | None:
        prompt = (
            "You are scoring relational join paths for dataset-aware retrieval quality.\n"
            "Return STRICT JSON only with fields:\n"
            "{\n"
            '  "semantic_relevance": <float 0..1>,\n'
            '  "temporal_risk": <float 0..1>,\n'
            '  "cardinality_risk": <float 0..1>,\n'
            '  "recommended": true|false,\n'
            '  "rationale_short": "<short text>"\n'
            "}\n"
            "No markdown.\n\n"
            f"Path tables: {json.dumps(path_obj.get('path_tables', []), ensure_ascii=True)}\n"
            f"Depth: {path_obj.get('depth')}\n"
            f"Estimated row multiplier: {path_obj.get('estimated_row_multiplier')}\n"
            f"Current recommended flag: {path_obj.get('recommended')}\n"
        )
        try:
            out = llm.generate_batch([prompt], max_new_tokens=192)[0]
            parsed = self._extract_json_object(out)
            if not isinstance(parsed, dict):
                return None
            sem = float(parsed.get("semantic_relevance", 0.0))
            tr = float(parsed.get("temporal_risk", 1.0))
            cr = float(parsed.get("cardinality_risk", 1.0))
            rec = bool(parsed.get("recommended", False))
            rationale = str(parsed.get("rationale_short", ""))[:240]
            return {
                "llm_semantic_relevance": max(0.0, min(1.0, sem)),
                "llm_temporal_risk": max(0.0, min(1.0, tr)),
                "llm_cardinality_risk": max(0.0, min(1.0, cr)),
                "llm_recommended": rec,
                "llm_rationale_short": rationale,
            }
        except Exception:
            return None

    def _build_semantic_context_graph_artifact(
        self,
        *,
        db: Any,
        filtered: dict[str, pd.DataFrame],
        cutoff: pd.Timestamp,
        schema_artifact: dict[str, Any],
        stats_artifact: dict[str, Any],
        semantics_artifact: dict[str, Any],
    ) -> dict[str, Any]:
        header = self._artifact_header(cutoff)
        graph = build_semantic_context_graph(
            db=db,
            filtered=filtered,
            schema_artifact=schema_artifact,
            stats_artifact=stats_artifact,
            semantics_artifact=semantics_artifact,
            infer_fk_candidates_fn=infer_fk_candidates,
            table_role_fn=self._table_role,
        )
        return {**header, **graph}
    def _build_safety_artifact(self, cutoff: pd.Timestamp) -> dict[str, Any]:
        header = self._artifact_header(cutoff)
        return {
            **header,
            "rules": {
                "global_temporal_cutoff": {
                    "description": "No rows at or beyond val_start are visible in Phase 1.",
                    "operator": "<",
                    "cutoff_time": cutoff,
                },
                "leakage_detection": {
                    "name_patterns": ["label", "target", "future", "leak", "outcome"],
                    "policy": "flag_for_review",
                },
                "manual_overrides": {
                    "allowed": False,
                    "description": "No manual denylist/allowlist in Phase 1.",
                },
            },
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 1 DB intelligence extraction pipeline")
    parser.add_argument("--dataset", type=str, default="rel-f1")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v2/phase1"))
    parser.add_argument("--max-path-depth", type=int, default=8)
    parser.add_argument("--sampling-row-threshold", type=int, default=200_000)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--correlation-topk-per-table", type=int, default=30)
    parser.add_argument("--correlation-global-cap", type=int, default=500)
    parser.add_argument("--disable-sampling", action="store_true")
    parser.add_argument("--max-paths", type=int, default=50_000)
    parser.add_argument("--max-frontier-per-start", type=int, default=5_000)
    parser.add_argument("--skip-correlations", action="store_true")
    parser.add_argument("--use-llm-semantics", action="store_true")
    parser.add_argument("--llm-model-size", choices=sorted(MODEL_PATHS), default="8b")
    parser.add_argument("--use-llm-path-scoring", action="store_true")
    parser.add_argument("--llm-path-top-k", type=int, default=200)
    parser.add_argument("--compute-path-catalog", action="store_true")
    parser.add_argument("--llm-semantics-confidence-threshold", type=float, default=0.6)
    parser.add_argument("--disable-llm-semantics-ensemble", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = Phase1Config(
        dataset=args.dataset,
        output_dir=args.output_dir,
        max_path_depth=args.max_path_depth,
        sampling_enabled=not args.disable_sampling,
        sampling_row_threshold=args.sampling_row_threshold,
        sample_size=args.sample_size,
        correlation_topk_per_table=args.correlation_topk_per_table,
        correlation_global_cap=args.correlation_global_cap,
        max_paths=args.max_paths,
        max_frontier_per_start=args.max_frontier_per_start,
        skip_correlations=args.skip_correlations,
        use_llm_semantics=args.use_llm_semantics,
        llm_model_size=args.llm_model_size,
        use_llm_path_scoring=args.use_llm_path_scoring,
        llm_path_top_k=args.llm_path_top_k,
        compute_path_catalog=args.compute_path_catalog or args.use_llm_path_scoring,
        llm_semantics_confidence_threshold=args.llm_semantics_confidence_threshold,
        llm_semantics_use_ensemble=not args.disable_llm_semantics_ensemble,
    )
    pipeline = Phase1Pipeline(cfg)
    paths = pipeline.run()
    print("Phase 1 completed. Artifacts:")
    for key, path in paths.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
