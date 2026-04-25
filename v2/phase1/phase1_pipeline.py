from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# Ensure sibling project modules under RFM/code are importable when running from RFM/v2.
_RFM_ROOT = Path(__file__).resolve().parents[2]
_RFM_CODE_DIR = _RFM_ROOT / "code"
if _RFM_CODE_DIR.exists():
    sys.path.insert(0, str(_RFM_CODE_DIR))

from relbench.tasks import get_task, get_task_names
from zero_shot_llm import LocalLLM, MODEL_PATHS

from phase1_data_access import Phase1DataAccess, TableMetadata
from phase1_fk import infer_fk_candidates, score_provided_foreign_keys
from phase1_semantic_graph import build_semantic_context_graph
from phase1_stats_sql import build_table_stats
from phase1_utils import (
    ALLOWED_COLUMN_ROLES,
    ALLOWED_TABLE_ROLES,
    is_boolean_dtype_name,
    is_numeric_dtype_name,
    is_temporal_dtype_name,
    is_text_dtype_name,
    normalize_identifier_tokens,
    write_json as _write_json,
)


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
    correlation_sample_size: int = 50_000
    histogram_bins: int = 20
    categorical_topk: int = 20
    random_seed: int = 42
    artifact_version: str = "phase1.v2"
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
    stats_workers: int = 1
    stats_checkpoint_resume: bool = True
    max_rows_exact_profile: int = 200_000
    max_corr_columns: int = 30
    max_corr_numeric_columns: int = 20
    max_corr_categorical_columns: int = 15
    max_corr_cross_columns: int = 15
    max_fk_pairs_exact: int = 250
    fk_overlap_sample_size: int = 50_000
    max_fk_null_ratio: float = 0.98
    max_cardinality_for_topk: int = 5_000
    max_text_profile_rows: int = 5_000
    include_low_confidence_fks: bool = True
    strict_cutoff_task_loading: bool = True
    cutoff_task_load_retries: int = 2
    max_composite_key_pairs: int = 20


class Phase1Pipeline:
    def __init__(self, config: Phase1Config):
        self.config = config
        self._llm: LocalLLM | None = None

    def run(self) -> dict[str, Path]:
        with Phase1DataAccess(
            dataset_name=self.config.dataset,
            random_seed=self.config.random_seed,
            download=True,
        ) as access:
            print("[phase1] computing global cutoff...")
            cutoff_info = self._compute_global_val_start(self.config.dataset)
            cutoff = pd.Timestamp(cutoff_info["cutoff"])
            access.set_cutoff(cutoff)
            print(f"[phase1] global cutoff (val_start) = {cutoff}")

            print("[phase1] scoring provided/inferred foreign keys...")
            fk_catalog = self._build_fk_catalog(access)

            print("[phase1] building stats artifact...")
            stats = self._build_stats_artifact(access, cutoff)

            print("[phase1] building schema artifact...")
            schema = self._build_schema_artifact(access, stats, cutoff, fk_catalog)

            print("[phase1] building semantics artifact...")
            semantics = self._build_semantics_artifact(schema, stats, cutoff)

            print("[phase1] building semantic context graph artifact...")
            semantic_context_graph = self._build_semantic_context_graph_artifact(
                cutoff=cutoff,
                schema_artifact=schema,
                stats_artifact=stats,
                semantics_artifact=semantics,
                fk_catalog=fk_catalog,
            )

            path_catalog: dict[str, Any] | None = None
            if self.config.compute_path_catalog:
                print("[phase1] building lightweight path catalog placeholder...")
                path_catalog = self._build_path_catalog_artifact(cutoff, fk_catalog)
            else:
                print("[phase1] skipping path catalog artifact.")

            print("[phase1] building safety artifact...")
            safety = self._build_safety_artifact(
                cutoff=cutoff,
                access=access,
                cutoff_info=cutoff_info,
                schema_artifact=schema,
                stats_artifact=stats,
                semantics_artifact=semantics,
                fk_catalog=fk_catalog,
            )

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

    def _compute_global_val_start(self, dataset: str) -> dict[str, Any]:
        task_reports: list[dict[str, Any]] = []
        errors: list[str] = []
        warnings: list[str] = []
        val_starts: list[tuple[str, pd.Timestamp]] = []
        for task_name in get_task_names(dataset):
            try:
                task = self._load_task_for_cutoff(dataset, task_name)
                val_table = task.get_table("val", mask_input_cols=False)
                time_col = getattr(task, "time_col", val_table.time_col)
                if time_col is None or time_col not in val_table.df.columns:
                    warnings.append(f"task={task_name}: missing validation time column")
                    task_reports.append(
                        {"task_name": task_name, "status": "missing_time_column", "time_col": time_col}
                    )
                    continue
                ts = pd.to_datetime(val_table.df[time_col], errors="coerce").dropna()
                if ts.empty:
                    warnings.append(f"task={task_name}: empty validation timestamps")
                    task_reports.append(
                        {"task_name": task_name, "status": "empty_validation_timestamps", "time_col": time_col}
                    )
                    continue
                val_start = pd.Timestamp(ts.min())
                val_starts.append((task_name, val_start))
                task_reports.append(
                    {
                        "task_name": task_name,
                        "status": "ok",
                        "time_col": time_col,
                        "val_start": val_start,
                    }
                )
            except Exception as exc:
                msg = f"task={task_name}: {exc}"
                errors.append(msg)
                task_reports.append({"task_name": task_name, "status": "error", "error": str(exc)})
        unique_cutoffs = sorted({pd.Timestamp(ts) for _, ts in val_starts})
        recoverable_errors = [
            msg
            for msg in errors
            if self._is_retriable_task_load_error(msg)
        ]
        if (
            errors
            and self.config.strict_cutoff_task_loading
            and not (unique_cutoffs and len(unique_cutoffs) == 1 and len(recoverable_errors) == len(errors))
        ):
            raise RuntimeError(
                "Failed to compute cutoff because one or more task validation tables could not be loaded: "
                + "; ".join(errors[:10])
            )
        if not val_starts:
            raise RuntimeError(f"Could not determine global val_start cutoff for dataset={dataset}.")
        if errors and unique_cutoffs and len(unique_cutoffs) == 1 and len(recoverable_errors) == len(errors):
            warnings.append(
                "One or more task artifacts could not be downloaded/verified, "
                "but all successfully loaded tasks agreed on a single dataset cutoff. "
                "Proceeding with that cutoff."
            )
        selected_task, cutoff = min(val_starts, key=lambda item: item[1])
        return {
            "cutoff": cutoff,
            "selected_from_task": selected_task,
            "task_reports": task_reports,
            "warnings": warnings,
            "errors": errors,
            "unique_successful_cutoffs": [str(ts) for ts in unique_cutoffs],
        }

    def _load_task_for_cutoff(self, dataset: str, task_name: str) -> Any:
        retries = max(0, int(getattr(self.config, "cutoff_task_load_retries", 2)))
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return get_task(dataset, task_name, download=True)
            except Exception as exc:
                last_exc = exc
                message = str(exc)
                retriable = self._is_retriable_task_load_error(message)
                if (not retriable) or attempt >= retries:
                    raise
                wait_s = min(2.0, 0.5 * (attempt + 1))
                print(
                    f"[phase1][cutoff] retrying task load dataset={dataset} task={task_name} "
                    f"attempt={attempt + 1}/{retries} after error: {message}"
                )
                time.sleep(wait_s)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"Unexpected task load failure for dataset={dataset} task={task_name}")

    def _is_retriable_task_load_error(self, message: str) -> bool:
        low = str(message).lower()
        retry_markers = [
            "sha256 hash of downloaded file",
            "downloaded file may have been corrupted",
            "connection reset",
            "connection aborted",
            "temporarily unavailable",
            "timed out",
            "timeout",
            "remote end closed connection",
            "incomplete read",
        ]
        return any(marker in low for marker in retry_markers)

    def _build_fk_catalog(self, access: Phase1DataAccess) -> list[dict[str, Any]]:
        provided = score_provided_foreign_keys(access=access, config=self.config)
        inferred = infer_fk_candidates(access=access, config=self.config)
        by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for candidate in provided + inferred:
            key = (
                str(candidate.get("child_table", "")),
                str(candidate.get("child_column", "")),
                str(candidate.get("parent_table", "")),
                str(candidate.get("parent_column", "")),
            )
            if not all(key):
                continue
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = candidate
                continue
            if str(candidate.get("source", "")) == "provided":
                by_key[key] = candidate
                continue
            if float(candidate.get("overlap_ratio", 0.0)) > float(existing.get("overlap_ratio", 0.0)):
                by_key[key] = candidate
        return sorted(
            by_key.values(),
            key=lambda row: (
                str(row.get("source", "")) != "provided",
                -float(row.get("overlap_ratio", 0.0)),
                str(row.get("child_table", "")),
                str(row.get("child_column", "")),
            ),
        )

    def _stats_checkpoint_signature(
        self,
        *,
        access: Phase1DataAccess,
        cutoff: pd.Timestamp,
    ) -> dict[str, Any]:
        manifest_json = json.dumps(access.dataset_manifest(), sort_keys=True, default=str)
        manifest_hash = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
        return {
            "artifact_version": self.config.artifact_version,
            "dataset_name": self.config.dataset,
            "global_cutoff_time": str(cutoff),
            "sampling_enabled": bool(self.config.sampling_enabled),
            "sampling_row_threshold": int(self.config.sampling_row_threshold),
            "sample_size": int(self.config.sample_size),
            "correlation_topk_per_table": int(self.config.correlation_topk_per_table),
            "correlation_global_cap": int(self.config.correlation_global_cap),
            "skip_correlations": bool(self.config.skip_correlations),
            "histogram_bins": int(self.config.histogram_bins),
            "categorical_topk": int(self.config.categorical_topk),
            "max_rows_exact_profile": int(self.config.max_rows_exact_profile),
            "max_corr_columns": int(self.config.max_corr_columns),
            "max_fk_pairs_exact": int(self.config.max_fk_pairs_exact),
            "max_cardinality_for_topk": int(self.config.max_cardinality_for_topk),
            "random_seed": int(self.config.random_seed),
            "manifest_hash": manifest_hash,
        }

    def _build_stats_artifact(
        self,
        access: Phase1DataAccess,
        cutoff: pd.Timestamp,
    ) -> dict[str, Any]:
        header = self._artifact_header(cutoff)
        table_order = list(access.table_names)
        total_tables = len(table_order)
        tables_by_name: dict[str, dict[str, Any]] = {}
        all_corrs: list[dict[str, Any]] = []

        stats_dir = self.config.output_dir / self.config.dataset
        stats_dir.mkdir(parents=True, exist_ok=True)
        ckpt_tables_path = stats_dir / "stats_tables_checkpoint.ndjson"
        ckpt_meta_path = stats_dir / "stats_checkpoint_meta.json"
        ckpt_progress_path = stats_dir / "stats_progress.json"
        signature = self._stats_checkpoint_signature(access=access, cutoff=cutoff)

        resume_enabled = bool(self.config.stats_checkpoint_resume)
        if resume_enabled and ckpt_meta_path.exists() and ckpt_tables_path.exists():
            try:
                existing_meta = json.loads(ckpt_meta_path.read_text(encoding="utf-8"))
            except Exception:
                existing_meta = {}
            if existing_meta == signature:
                with ckpt_tables_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except Exception:
                            continue
                        table_name = str(payload.get("table_name", ""))
                        if table_name in access.table_names:
                            tables_by_name[table_name] = payload
                if tables_by_name:
                    print(
                        f"[phase1][stats] resumed {len(tables_by_name)}/{total_tables} tables from checkpoint."
                    )
            else:
                ckpt_tables_path.unlink(missing_ok=True)
                ckpt_progress_path.unlink(missing_ok=True)
        else:
            ckpt_tables_path.unlink(missing_ok=True)
            ckpt_progress_path.unlink(missing_ok=True)

        ckpt_meta_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
        for resumed in tables_by_name.values():
            all_corrs.extend(resumed.get("correlations", []))

        pending_tables = [name for name in table_order if name not in tables_by_name]
        print(f"[phase1][stats] computing {len(pending_tables)} pending tables.")
        for table_name in pending_tables:
            table_payload, elapsed = build_table_stats(
                access=access,
                table_meta=access.table_meta(table_name),
                config=self.config,
            )
            tables_by_name[table_name] = table_payload
            all_corrs.extend(table_payload.get("correlations", []))
            with ckpt_tables_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(table_payload, default=str))
                handle.write("\n")
            _write_json(
                ckpt_progress_path,
                {
                    "dataset_name": self.config.dataset,
                    "global_cutoff_time": str(cutoff),
                    "completed_tables": len(tables_by_name),
                    "total_tables": total_tables,
                    "last_table_name": table_name,
                    "last_table_elapsed_sec": float(elapsed),
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            print(
                f"[phase1][stats] {len(tables_by_name)}/{total_tables} "
                f"table={table_name} done in {elapsed:.2f}s"
            )

        tables = [tables_by_name[name] for name in table_order if name in tables_by_name]
        all_corrs.sort(key=lambda row: abs(float(row["score"])), reverse=True)
        mode_counts = {
            "exact": sum(1 for table in tables if table.get("profiling_mode") == "exact"),
            "sampled": sum(1 for table in tables if table.get("profiling_mode") == "sampled"),
            "approx": sum(1 for table in tables if table.get("profiling_mode") == "approx"),
            "skipped": sum(1 for table in tables if table.get("profiling_mode") == "skipped"),
        }
        return {
            **header,
            "profiling_summary": {
                "mode_counts": mode_counts,
                "checkpoint_resume_enabled": bool(self.config.stats_checkpoint_resume),
                "sampling_enabled": bool(self.config.sampling_enabled),
                "sample_size": int(self.config.sample_size),
                "max_rows_exact_profile": int(self.config.max_rows_exact_profile),
            },
            "tables": tables,
            "global_top_correlations": all_corrs[: self.config.correlation_global_cap],
        }

    def _infer_pk_candidates(
        self,
        access: Phase1DataAccess,
        table_meta: TableMetadata,
        stats_table: dict[str, Any],
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        row_count = int(stats_table.get("row_count", 0))
        if row_count <= 0:
            return candidates
        columns_stats = stats_table.get("columns", {})
        for column_name in table_meta.columns:
            meta = columns_stats.get(column_name, {})
            distinct_count = meta.get("distinct_count")
            missingness = meta.get("missingness")
            non_null_count = int(meta.get("non_null_count", 0) or 0)
            if distinct_count is None:
                continue
            if int(distinct_count) == row_count and non_null_count == row_count:
                candidates.append(
                    {
                        "columns": [column_name],
                        "confidence": "high",
                        "reason": "unique_and_non_null",
                    }
                )
            elif int(distinct_count) >= int(0.995 * row_count) and missingness is not None and float(missingness) <= 0.005:
                candidates.append(
                    {
                        "columns": [column_name],
                        "confidence": "medium",
                        "reason": "near_unique",
                    }
                )

        id_like = [col for col in table_meta.columns if col.lower().endswith("id")]
        relation_sql = access.relation_sql(table_meta.table_name, filtered=True)
        if row_count > self.config.max_rows_exact_profile:
            relation_sql = access.sampled_relation_sql(
                table_meta.table_name,
                filtered=True,
                sample_size=min(self.config.sample_size, row_count),
                columns=id_like[:12],
            )
        pairs_checked = 0
        for idx, col_a in enumerate(id_like[:12]):
            for col_b in id_like[idx + 1 : 12]:
                if pairs_checked >= self.config.max_composite_key_pairs:
                    break
                pairs_checked += 1
                qcol_a = f'"{col_a}"'
                qcol_b = f'"{col_b}"'
                try:
                    row = access.fetchdf(
                        f"WITH valid AS ("
                        f"  SELECT {qcol_a} AS col_a, {qcol_b} AS col_b FROM ({relation_sql}) rel "
                        f"  WHERE {qcol_a} IS NOT NULL AND {qcol_b} IS NOT NULL"
                        f"), counts AS ("
                        f"  SELECT COUNT(*) AS non_null_rows FROM valid"
                        f"), distinct_pairs AS ("
                        f"  SELECT COUNT(*) AS distinct_rows FROM (SELECT DISTINCT col_a, col_b FROM valid)"
                        f") "
                        "SELECT non_null_rows, distinct_rows FROM counts CROSS JOIN distinct_pairs"
                    )
                except Exception:
                    continue
                if row.empty:
                    continue
                non_null_rows = int(row.iloc[0].get("non_null_rows", 0) or 0)
                distinct_rows = int(row.iloc[0].get("distinct_rows", 0) or 0)
                if non_null_rows <= 0:
                    continue
                coverage = float(non_null_rows / row_count)
                if distinct_rows == non_null_rows and coverage >= 0.95:
                    candidates.append(
                        {
                            "columns": [col_a, col_b],
                            "confidence": "medium",
                            "reason": "composite_unique",
                        }
                    )
        return candidates

    def _build_schema_artifact(
        self,
        access: Phase1DataAccess,
        stats_artifact: dict[str, Any],
        cutoff: pd.Timestamp,
        fk_catalog: list[dict[str, Any]],
    ) -> dict[str, Any]:
        header = self._artifact_header(cutoff)
        stats_by_table = {
            str(table.get("table_name", "")): table
            for table in stats_artifact.get("tables", [])
        }
        tables: list[dict[str, Any]] = []
        for table_name in access.table_names:
            table_meta = access.table_meta(table_name)
            stats_table = stats_by_table.get(table_name, {})
            columns = []
            for column_name in table_meta.columns:
                stats_col = (stats_table.get("columns", {}) or {}).get(column_name, {})
                columns.append(
                    {
                        "name": column_name,
                        "dtype": table_meta.dtypes.get(column_name, ""),
                        "nullable": bool((stats_col.get("missingness") or 0.0) > 0.0),
                        "n_unique": stats_col.get("distinct_count"),
                        "n_unique_method": stats_col.get("distinct_count_method"),
                    }
                )
            tables.append(
                {
                    "table_name": table_name,
                    "row_count": int(stats_table.get("row_count", 0)),
                    "rows_before_cutoff_filter": int(stats_table.get("rows_before_cutoff_filter", 0)),
                    "rows_after_cutoff_filter": int(stats_table.get("rows_after_cutoff_filter", 0)),
                    "rows_dropped_invalid_time": int(stats_table.get("rows_dropped_invalid_time", 0)),
                    "provided_primary_key": table_meta.provided_primary_key,
                    "provided_foreign_keys": dict(table_meta.provided_foreign_keys),
                    "time_column": table_meta.time_column,
                    "detected_time_columns": list(table_meta.detected_time_columns),
                    "metadata_source": table_meta.metadata_source,
                    "metadata_warnings": list(table_meta.metadata_warnings),
                    "columns": columns,
                    "inferred_pk_candidates": self._infer_pk_candidates(access, table_meta, stats_table),
                }
            )
        return {
            **header,
            "tables": tables,
            "provided_foreign_key_candidates": [fk for fk in fk_catalog if fk.get("source") == "provided"],
            "inferred_foreign_key_candidates": [fk for fk in fk_catalog if fk.get("source") != "provided"],
            "foreign_key_catalog": fk_catalog,
        }

    def _table_role(
        self,
        table_name: str,
        schema_table: dict[str, Any],
        stats_table: dict[str, Any],
    ) -> str:
        lower = table_name.lower()
        time_column = schema_table.get("time_column")
        provided_fks = schema_table.get("provided_foreign_keys", {}) or {}
        columns = schema_table.get("columns", [])
        row_count = int(stats_table.get("row_count", schema_table.get("row_count", 0)) or 0)
        if time_column is None:
            if len(provided_fks) >= 2 and len(columns) <= len(provided_fks) + 4:
                return "bridge"
            return "static"
        if any(tok in lower for tok in {"event", "log", "history", "fact", "transaction"}) or lower.endswith("s"):
            return "event"
        if len(provided_fks) >= 2 and len(columns) <= len(provided_fks) + 5:
            return "bridge"
        low_card_cols = 0
        for column_name, column_meta in (stats_table.get("columns", {}) or {}).items():
            distinct_count = column_meta.get("distinct_count")
            if distinct_count is not None and row_count > 0 and int(distinct_count) <= min(50, max(5, int(0.02 * row_count))):
                low_card_cols += 1
        if row_count > 0 and low_card_cols >= max(2, len(columns) // 2):
            return "lookup"
        return "entity"

    def _column_role(
        self,
        column_name: str,
        dtype_name: str,
        column_stats: dict[str, Any],
    ) -> str:
        low = column_name.lower()
        distinct_count = column_stats.get("distinct_count")
        non_null_count = int(column_stats.get("non_null_count", 0) or 0)
        cardinality_ratio = (
            float(distinct_count) / max(1, non_null_count)
            if distinct_count is not None and non_null_count > 0
            else 0.0
        )
        if low == "id" or low.endswith("id") or low.endswith("_key"):
            return "id"
        if is_temporal_dtype_name(dtype_name) or any(tok in low for tok in ["date", "time", "timestamp"]):
            return "timestamp"
        if low.startswith(("is_", "has_")) or low.endswith("_flag") or "status" in low or "state" in low:
            return "status"
        if is_boolean_dtype_name(dtype_name):
            return "status"
        if is_text_dtype_name(dtype_name):
            if any(tok in low for tok in ["name", "title", "desc", "description", "text", "body", "comment", "message"]):
                return "text"
            if distinct_count is not None and int(distinct_count) > 50:
                return "text"
            return "category"
        if is_numeric_dtype_name(dtype_name):
            if cardinality_ratio >= 0.98 and any(tok in low for tok in ["id", "key"]):
                return "id"
            return "measure"
        if distinct_count is not None and int(distinct_count) <= 20:
            return "category"
        return "other"

    def _column_prior_info(
        self,
        *,
        column_name: str,
        dtype_name: str,
        column_stats: dict[str, Any],
    ) -> dict[str, Any]:
        non_null = int(column_stats.get("non_null_count", 0) or 0)
        unique = column_stats.get("distinct_count")
        heuristic_role = self._column_role(column_name, dtype_name, column_stats)
        top_values = []
        categorical_meta = column_stats.get("categorical", {})
        if isinstance(categorical_meta, dict):
            top_values = categorical_meta.get("top_values", [])[:5]
        numeric_meta = column_stats.get("numeric", {})
        numeric_range = None
        if isinstance(numeric_meta, dict) and numeric_meta:
            numeric_range = {
                "min": numeric_meta.get("min"),
                "max": numeric_meta.get("max"),
                "median": numeric_meta.get("median"),
            }
        return {
            "name": column_name,
            "dtype": dtype_name,
            "non_null": non_null,
            "missing_ratio": column_stats.get("missingness"),
            "unique": unique,
            "unique_ratio": (float(unique) / max(1, non_null)) if unique is not None and non_null > 0 else 0.0,
            "heuristic_prior_role": heuristic_role,
            "top_values": top_values,
            "numeric_range": numeric_range,
        }

    def _build_semantics_artifact(
        self,
        schema_artifact: dict[str, Any],
        stats_artifact: dict[str, Any],
        cutoff: pd.Timestamp,
    ) -> dict[str, Any]:
        header = self._artifact_header(cutoff)
        stats_by_table = {
            str(table.get("table_name", "")): table
            for table in stats_artifact.get("tables", [])
        }
        tables = []
        leakage_patterns = ["label", "target", "future", "leak", "outcome"]
        llm_used = False
        llm_failures = 0
        llm_diagnostics: list[dict[str, Any]] = []
        llm = self._get_llm() if self.config.use_llm_semantics else None

        for schema_table in schema_artifact.get("tables", []):
            table_name = str(schema_table.get("table_name", ""))
            stats_table = stats_by_table.get(table_name, {})
            table_role = self._table_role(table_name, schema_table, stats_table)
            col_specs = [
                self._column_prior_info(
                    column_name=str(column.get("name", "")),
                    dtype_name=str(column.get("dtype", "")),
                    column_stats=(stats_table.get("columns", {}) or {}).get(str(column.get("name", "")), {}),
                )
                for column in schema_table.get("columns", [])
            ]
            llm_result: dict[str, Any] | None = None
            if llm is not None:
                llm_result, diag = self._annotate_table_semantics_with_llm(
                    llm=llm,
                    table_name=table_name,
                    heuristic_table_role=table_role,
                    provided_time_col=schema_table.get("time_column"),
                    column_specs=col_specs,
                )
                llm_diagnostics.append(diag)
                if llm_result is not None:
                    llm_used = True
                else:
                    llm_failures += 1

            columns = []
            spec_by_name = {spec["name"]: spec for spec in col_specs}
            if llm_result is not None:
                llm_table_role = self._normalize_table_role(llm_result.get("table_role"), table_role)
                table_role_conf = float(llm_result.get("table_role_confidence", 0.0))
                if table_role_conf >= self.config.llm_semantics_confidence_threshold:
                    table_role = llm_table_role
                llm_columns = llm_result.get("columns", {})
                for spec in col_specs:
                    column_name = spec["name"]
                    col_info = llm_columns.get(column_name, {})
                    heuristic_role = spec["heuristic_prior_role"]
                    llm_role = self._normalize_column_role(col_info.get("semantic_role"), heuristic_role)
                    llm_role = self._semantic_rule_override(column_name, llm_role, heuristic_role)
                    llm_conf = float(col_info.get("confidence", 0.0))
                    llm_abstain = bool(col_info.get("abstain", False))
                    if llm_abstain or llm_conf < self.config.llm_semantics_confidence_threshold:
                        role = heuristic_role
                        confidence = "merged_low_conf_fallback"
                    else:
                        role = llm_role
                        confidence = "llm"
                    leakage_risk = bool(
                        col_info.get(
                            "leakage_risk",
                            any(pattern in column_name.lower() for pattern in leakage_patterns),
                        )
                    )
                    columns.append(
                        {
                            "name": column_name,
                            "semantic_role": role,
                            "leakage_risk": leakage_risk,
                            "confidence": confidence,
                            "llm_confidence": llm_conf,
                            "llm_abstain": llm_abstain,
                            "heuristic_prior_role": heuristic_role,
                        }
                    )
            else:
                for spec in col_specs:
                    column_name = spec["name"]
                    role = spec["heuristic_prior_role"]
                    leakage_risk = any(pattern in column_name.lower() for pattern in leakage_patterns)
                    columns.append(
                        {
                            "name": column_name,
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
                    "time_column": schema_table.get("time_column"),
                    "columns": columns,
                }
            )

        validation = self._validate_semantics_tables(tables)
        return {
            **header,
            "annotation_model": {
                "name": f"local-llama-{self.config.llm_model_size}" if llm_used else "heuristic-summary-baseline",
                "llm_used": llm_used,
                "llm_failures": llm_failures,
                "determinism": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "do_sample": False,
                    "max_retries_on_parse_failure": 2,
                },
                "notes": (
                    "LLM semantic annotation enabled with deterministic generation over schema/stats summaries."
                    if llm_used
                    else "Heuristic semantic baseline built from schema/stats summaries."
                ),
            },
            "llm_diagnostics": llm_diagnostics,
            "validation": validation,
            "tables": tables,
        }

    def _validate_semantics_tables(self, tables: list[dict[str, Any]]) -> dict[str, Any]:
        issues: list[dict[str, Any]] = []
        for table in tables:
            table_name = table.get("table_name", "")
            table_role = str(table.get("table_role", ""))
            if table_role not in ALLOWED_TABLE_ROLES:
                issues.append({"type": "invalid_table_role", "table": table_name, "value": table_role})
            for column in table.get("columns", []):
                column_name = column.get("name", "")
                semantic_role = str(column.get("semantic_role", ""))
                if semantic_role not in ALLOWED_COLUMN_ROLES:
                    issues.append(
                        {
                            "type": "invalid_column_role",
                            "table": table_name,
                            "column": column_name,
                            "value": semantic_role,
                        }
                    )
                lower = str(column_name).lower()
                if any(tok in lower for tok in ["date", "time", "timestamp"]) and semantic_role != "timestamp":
                    issues.append(
                        {
                            "type": "timestamp_name_mismatch",
                            "table": table_name,
                            "column": column_name,
                            "value": semantic_role,
                        }
                    )
                if lower.endswith("id") and semantic_role != "id":
                    issues.append(
                        {
                            "type": "id_name_mismatch",
                            "table": table_name,
                            "column": column_name,
                            "value": semantic_role,
                        }
                    )
        return {"issue_count": len(issues), "issues": issues[:500], "passed": len(issues) == 0}

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

    def _semantic_rule_override(self, column_name: str, role: str, heuristic_role: str) -> str:
        lower = column_name.lower()
        if lower.endswith("id") and role != "id":
            return "id"
        if any(tok in lower for tok in ["date", "time", "timestamp"]) and role != "timestamp":
            return "timestamp"
        if role not in ALLOWED_COLUMN_ROLES:
            return heuristic_role
        return role

    def _build_semantics_prompt(
        self,
        *,
        table_name: str,
        heuristic_table_role: str,
        provided_time_col: str | None,
        column_specs: list[dict[str, Any]],
        variant: str,
    ) -> str:
        lead = "Use the summary features as guidance, not absolute truth."
        if variant == "alt":
            lead = "Focus on consistency between names, dtypes, cardinality, and priors. If uncertain, abstain."
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
            f"Heuristic table role: {heuristic_table_role}\n"
            f"Provided time column: {provided_time_col}\n"
            f"Columns with priors:\n{json.dumps(column_specs, ensure_ascii=True)}\n"
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
        heuristic_table_role: str,
        column_specs: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[str]]:
        issues: list[str] = []
        out: dict[str, Any] = {"columns": {}}
        out["table_role"] = self._normalize_table_role(parsed.get("table_role"), heuristic_table_role)
        if out["table_role"] != str(parsed.get("table_role", "")).strip().lower():
            issues.append(f"{table_name}: repaired_table_role")
        try:
            out["table_role_confidence"] = max(0.0, min(1.0, float(parsed.get("table_role_confidence", 0.0))))
        except Exception:
            out["table_role_confidence"] = 0.0
            issues.append(f"{table_name}: invalid_table_role_confidence")

        llm_columns = parsed.get("columns", {})
        if not isinstance(llm_columns, dict):
            llm_columns = {}
            issues.append(f"{table_name}: invalid_columns_object")

        for spec in column_specs:
            column_name = spec["name"]
            heuristic_role = spec["heuristic_prior_role"]
            payload = llm_columns.get(column_name, {})
            if not isinstance(payload, dict):
                payload = {}
                issues.append(f"{table_name}.{column_name}: invalid_column_payload")
            role = self._normalize_column_role(payload.get("semantic_role"), heuristic_role)
            role = self._semantic_rule_override(column_name, role, heuristic_role)
            try:
                confidence = float(payload.get("confidence", 0.0))
            except Exception:
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))
            abstain = bool(payload.get("abstain", False))
            leakage_risk = bool(
                payload.get(
                    "leakage_risk",
                    any(tok in column_name.lower() for tok in ["label", "target", "future", "leak", "outcome"]),
                )
            )
            out["columns"][column_name] = {
                "semantic_role": role,
                "confidence": confidence,
                "abstain": abstain,
                "leakage_risk": leakage_risk,
            }
        return out, issues

    def _annotate_table_semantics_with_llm(
        self,
        *,
        llm: LocalLLM,
        table_name: str,
        heuristic_table_role: str,
        provided_time_col: str | None,
        column_specs: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        diagnostics: dict[str, Any] = {
            "table_name": table_name,
            "llm_solver_ok": False,
            "llm_critic_ok": False,
            "llm_ensemble_used": False,
            "issues": [],
        }

        def _run_prompt(prompt: str) -> dict[str, Any] | None:
            for _ in range(3):
                try:
                    output = llm.generate_batch([prompt], max_new_tokens=self.config.llm_semantics_max_new_tokens)[0]
                    parsed = self._extract_json_object(output)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
            return None

        primary_prompt = self._build_semantics_prompt(
            table_name=table_name,
            heuristic_table_role=heuristic_table_role,
            provided_time_col=provided_time_col,
            column_specs=column_specs,
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
            heuristic_table_role=heuristic_table_role,
            column_specs=column_specs,
        )
        diagnostics["issues"].extend(primary_issues)

        critic_prompt = self._build_semantics_critic_prompt(primary_sanitized, table_name)
        parsed_critic = _run_prompt(critic_prompt)
        final_sanitized = primary_sanitized
        if isinstance(parsed_critic, dict):
            diagnostics["llm_critic_ok"] = True
            critic_sanitized, critic_issues = self._sanitize_llm_semantics_result(
                parsed=parsed_critic,
                table_name=table_name,
                heuristic_table_role=heuristic_table_role,
                column_specs=column_specs,
            )
            diagnostics["issues"].extend(critic_issues)
            final_sanitized = critic_sanitized

        ambiguity_score = 0
        for spec in column_specs:
            heuristic_role = spec["heuristic_prior_role"]
            column_name = spec["name"]
            result = final_sanitized["columns"][column_name]
            if bool(result.get("abstain")) or float(result.get("confidence", 0.0)) < self.config.llm_semantics_confidence_threshold:
                ambiguity_score += 1
            elif str(result.get("semantic_role")) != heuristic_role:
                ambiguity_score += 1
        if self.config.llm_semantics_use_ensemble and ambiguity_score > max(2, int(0.2 * max(1, len(column_specs)))):
            alt_prompt = self._build_semantics_prompt(
                table_name=table_name,
                heuristic_table_role=heuristic_table_role,
                provided_time_col=provided_time_col,
                column_specs=column_specs,
                variant="alt",
            )
            parsed_alt = _run_prompt(alt_prompt)
            if isinstance(parsed_alt, dict):
                diagnostics["llm_ensemble_used"] = True
                alt_sanitized, alt_issues = self._sanitize_llm_semantics_result(
                    parsed=parsed_alt,
                    table_name=table_name,
                    heuristic_table_role=heuristic_table_role,
                    column_specs=column_specs,
                )
                diagnostics["issues"].extend(alt_issues)
                role_votes = [
                    final_sanitized.get("table_role", heuristic_table_role),
                    alt_sanitized.get("table_role", heuristic_table_role),
                    heuristic_table_role,
                ]
                final_sanitized["table_role"] = max(set(role_votes), key=role_votes.count)
                for spec in column_specs:
                    column_name = spec["name"]
                    votes = [
                        final_sanitized["columns"][column_name]["semantic_role"],
                        alt_sanitized["columns"][column_name]["semantic_role"],
                        spec["heuristic_prior_role"],
                    ]
                    voted = max(set(votes), key=votes.count)
                    final_sanitized["columns"][column_name]["semantic_role"] = voted
                    final_sanitized["columns"][column_name]["confidence"] = max(
                        float(final_sanitized["columns"][column_name].get("confidence", 0.0)),
                        float(alt_sanitized["columns"][column_name].get("confidence", 0.0)),
                    )
                    final_sanitized["columns"][column_name]["abstain"] = bool(
                        final_sanitized["columns"][column_name].get("abstain", False)
                    ) and bool(alt_sanitized["columns"][column_name].get("abstain", False))

        return final_sanitized, diagnostics

    def _build_semantic_context_graph_artifact(
        self,
        *,
        cutoff: pd.Timestamp,
        schema_artifact: dict[str, Any],
        stats_artifact: dict[str, Any],
        semantics_artifact: dict[str, Any],
        fk_catalog: list[dict[str, Any]],
    ) -> dict[str, Any]:
        header = self._artifact_header(cutoff)
        graph = build_semantic_context_graph(
            schema_artifact=schema_artifact,
            stats_artifact=stats_artifact,
            semantics_artifact=semantics_artifact,
            fk_edges=fk_catalog,
        )
        return {**header, **graph}

    def _build_path_catalog_artifact(
        self,
        cutoff: pd.Timestamp,
        fk_catalog: list[dict[str, Any]],
    ) -> dict[str, Any]:
        header = self._artifact_header(cutoff)
        return {
            **header,
            "max_path_depth": self.config.max_path_depth,
            "max_paths_cap": self.config.max_paths,
            "truncated": False,
            "path_scoring_report": {
                "llm_used": False,
                "scored_count": 0,
                "top_k": self.config.llm_path_top_k,
                "failures": 0,
            },
            "foreign_key_edges": fk_catalog,
            "paths": [],
            "notes": "Path catalog generation is intentionally deferred in this SQL-first scalable refactor.",
        }

    def _build_safety_artifact(
        self,
        *,
        cutoff: pd.Timestamp,
        access: Phase1DataAccess,
        cutoff_info: dict[str, Any],
        schema_artifact: dict[str, Any],
        stats_artifact: dict[str, Any],
        semantics_artifact: dict[str, Any],
        fk_catalog: list[dict[str, Any]],
    ) -> dict[str, Any]:
        header = self._artifact_header(cutoff)
        invalid_time_tables = []
        static_tables = []
        metadata_warnings = []
        for table in schema_artifact.get("tables", []):
            if int(table.get("rows_dropped_invalid_time", 0)) > 0:
                invalid_time_tables.append(
                    {
                        "table_name": table.get("table_name"),
                        "rows_dropped_invalid_time": table.get("rows_dropped_invalid_time"),
                    }
                )
            if table.get("time_column") is None:
                static_tables.append(str(table.get("table_name", "")))
            if table.get("metadata_warnings"):
                metadata_warnings.append(
                    {
                        "table_name": table.get("table_name"),
                        "warnings": table.get("metadata_warnings"),
                    }
                )
        leakage_prone_columns = []
        for table in semantics_artifact.get("tables", []):
            for column in table.get("columns", []):
                if bool(column.get("leakage_risk", False)):
                    leakage_prone_columns.append(
                        {
                            "table_name": table.get("table_name"),
                            "column_name": column.get("name"),
                            "semantic_role": column.get("semantic_role"),
                        }
                    )
        return {
            **header,
            "cutoff_diagnostics": cutoff_info,
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
                "profiling_budget_modes": {
                    "allowed_values": ["exact", "sampled", "approx", "skipped"],
                    "default_large_table_mode": "sampled",
                },
            },
            "observations": {
                "tables_with_invalid_time_rows": invalid_time_tables,
                "static_tables": static_tables,
                "metadata_warnings": metadata_warnings,
                "leakage_prone_columns": leakage_prone_columns,
                "foreign_key_summary": {
                    "provided_count": sum(1 for fk in fk_catalog if fk.get("source") == "provided"),
                    "inferred_count": sum(1 for fk in fk_catalog if fk.get("source") != "provided"),
                    "low_confidence_count": sum(1 for fk in fk_catalog if fk.get("confidence") == "low"),
                },
            },
            "validation": {
                "cutoff_errors": cutoff_info.get("errors", []),
                "cutoff_warnings": cutoff_info.get("warnings", []),
                "semantics_validation_passed": bool(
                    (semantics_artifact.get("validation", {}) or {}).get("passed", False)
                ),
                "table_count": len(schema_artifact.get("tables", [])),
            },
        }


def _load_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(raw)
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required to load YAML config files.") from exc
    payload = yaml.safe_load(raw) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return payload


def build_arg_parser(defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    defaults = defaults or {}
    parser = argparse.ArgumentParser(description="Phase 1 DB intelligence extraction pipeline")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default=defaults.get("dataset", "rel-f1"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(defaults.get("output_dir", "artifacts/v2/phase1")),
    )
    parser.add_argument("--max-path-depth", type=int, default=int(defaults.get("max_path_depth", 8)))
    parser.add_argument(
        "--sampling-row-threshold",
        type=int,
        default=int(defaults.get("sampling_row_threshold", 200_000)),
    )
    parser.add_argument("--sample-size", type=int, default=int(defaults.get("sample_size", 100_000)))
    parser.add_argument(
        "--correlation-topk-per-table",
        type=int,
        default=int(defaults.get("correlation_topk_per_table", 30)),
    )
    parser.add_argument(
        "--correlation-global-cap",
        type=int,
        default=int(defaults.get("correlation_global_cap", 500)),
    )
    parser.add_argument(
        "--correlation-sample-size",
        type=int,
        default=int(defaults.get("correlation_sample_size", 50_000)),
    )
    parser.add_argument("--stats-workers", type=int, default=int(defaults.get("stats_workers", 1)))
    parser.add_argument(
        "--disable-sampling",
        action="store_true",
        default=not bool(defaults.get("sampling_enabled", True)),
    )
    parser.add_argument("--max-paths", type=int, default=int(defaults.get("max_paths", 50_000)))
    parser.add_argument(
        "--max-frontier-per-start",
        type=int,
        default=int(defaults.get("max_frontier_per_start", 5_000)),
    )
    parser.add_argument(
        "--skip-correlations",
        action="store_true",
        default=bool(defaults.get("skip_correlations", False)),
    )
    parser.add_argument(
        "--disable-stats-checkpoint-resume",
        action="store_true",
        default=not bool(defaults.get("stats_checkpoint_resume", True)),
    )
    parser.add_argument(
        "--use-llm-semantics",
        action="store_true",
        default=bool(defaults.get("use_llm_semantics", False)),
    )
    parser.add_argument("--llm-model-size", choices=sorted(MODEL_PATHS), default=defaults.get("llm_model_size", "8b"))
    parser.add_argument(
        "--use-llm-path-scoring",
        action="store_true",
        default=bool(defaults.get("use_llm_path_scoring", False)),
    )
    parser.add_argument("--llm-path-top-k", type=int, default=int(defaults.get("llm_path_top_k", 200)))
    parser.add_argument(
        "--compute-path-catalog",
        action="store_true",
        default=bool(defaults.get("compute_path_catalog", False)),
    )
    parser.add_argument(
        "--llm-semantics-confidence-threshold",
        type=float,
        default=float(defaults.get("llm_semantics_confidence_threshold", 0.6)),
    )
    parser.add_argument(
        "--disable-llm-semantics-ensemble",
        action="store_true",
        default=not bool(defaults.get("llm_semantics_use_ensemble", True)),
    )
    parser.add_argument("--histogram-bins", type=int, default=int(defaults.get("histogram_bins", 20)))
    parser.add_argument("--categorical-topk", type=int, default=int(defaults.get("categorical_topk", 20)))
    parser.add_argument("--random-seed", type=int, default=int(defaults.get("random_seed", 42)))
    parser.add_argument("--artifact-version", type=str, default=str(defaults.get("artifact_version", "phase1.v2")))
    parser.add_argument(
        "--max-rows-exact-profile",
        type=int,
        default=int(defaults.get("max_rows_exact_profile", defaults.get("sampling_row_threshold", 200_000))),
    )
    parser.add_argument("--max-corr-columns", type=int, default=int(defaults.get("max_corr_columns", 30)))
    parser.add_argument(
        "--max-corr-numeric-columns",
        type=int,
        default=int(defaults.get("max_corr_numeric_columns", 20)),
    )
    parser.add_argument(
        "--max-corr-categorical-columns",
        type=int,
        default=int(defaults.get("max_corr_categorical_columns", 15)),
    )
    parser.add_argument(
        "--max-corr-cross-columns",
        type=int,
        default=int(defaults.get("max_corr_cross_columns", 15)),
    )
    parser.add_argument("--max-fk-pairs-exact", type=int, default=int(defaults.get("max_fk_pairs_exact", 250)))
    parser.add_argument(
        "--fk-overlap-sample-size",
        type=int,
        default=int(defaults.get("fk_overlap_sample_size", 50_000)),
    )
    parser.add_argument("--max-fk-null-ratio", type=float, default=float(defaults.get("max_fk_null_ratio", 0.98)))
    parser.add_argument(
        "--max-cardinality-for-topk",
        type=int,
        default=int(defaults.get("max_cardinality_for_topk", 5_000)),
    )
    parser.add_argument("--max-text-profile-rows", type=int, default=int(defaults.get("max_text_profile_rows", 5_000)))
    parser.add_argument(
        "--cutoff-task-load-retries",
        type=int,
        default=int(defaults.get("cutoff_task_load_retries", 2)),
    )
    parser.add_argument(
        "--strict-cutoff-task-loading",
        action="store_true",
        default=bool(defaults.get("strict_cutoff_task_loading", True)),
    )
    return parser


def _build_config_from_args(args: argparse.Namespace) -> Phase1Config:
    return Phase1Config(
        dataset=args.dataset,
        output_dir=args.output_dir,
        max_path_depth=args.max_path_depth,
        sampling_enabled=not args.disable_sampling,
        sampling_row_threshold=args.sampling_row_threshold,
        sample_size=args.sample_size,
        correlation_topk_per_table=args.correlation_topk_per_table,
        correlation_global_cap=args.correlation_global_cap,
        correlation_sample_size=args.correlation_sample_size,
        histogram_bins=args.histogram_bins,
        categorical_topk=args.categorical_topk,
        random_seed=args.random_seed,
        artifact_version=args.artifact_version,
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
        stats_workers=max(1, int(args.stats_workers)),
        stats_checkpoint_resume=not args.disable_stats_checkpoint_resume,
        max_rows_exact_profile=args.max_rows_exact_profile,
        max_corr_columns=args.max_corr_columns,
        max_corr_numeric_columns=args.max_corr_numeric_columns,
        max_corr_categorical_columns=args.max_corr_categorical_columns,
        max_corr_cross_columns=args.max_corr_cross_columns,
        max_fk_pairs_exact=args.max_fk_pairs_exact,
        fk_overlap_sample_size=args.fk_overlap_sample_size,
        max_fk_null_ratio=args.max_fk_null_ratio,
        max_cardinality_for_topk=args.max_cardinality_for_topk,
        max_text_profile_rows=args.max_text_profile_rows,
        cutoff_task_load_retries=args.cutoff_task_load_retries,
        strict_cutoff_task_loading=bool(args.strict_cutoff_task_loading),
    )


def main() -> None:
    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument("--config", type=Path, default=None)
    known, _ = config_probe.parse_known_args()
    defaults: dict[str, Any] = {}
    if known.config is not None:
        defaults = _load_config_file(known.config)
    parser = build_arg_parser(defaults)
    args = parser.parse_args()
    cfg = _build_config_from_args(args)
    pipeline = Phase1Pipeline(cfg)
    paths = pipeline.run()
    print("Phase 1 completed. Artifacts:")
    for key, path in paths.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
