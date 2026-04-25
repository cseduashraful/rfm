from __future__ import annotations

import argparse
import copy
import math
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Ensure imports from RFM/code and local phase2 modules.
_RFM_ROOT = Path(__file__).resolve().parents[2]
_RFM_CODE_DIR = _RFM_ROOT / "code"
if _RFM_CODE_DIR.exists():
    sys.path.insert(0, str(_RFM_CODE_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase2_io import append_jsonl, load_phase1_bundle, utc_now_iso, write_json
from phase2_models import BudgetPolicy, CompilationContext, TaskDefinition
from phase2_prompts import build_critic_prompt, build_solver_prompt
from phase2_validation import validate_task_spec_consistency, validate_task_spec_schema
from phase2_policy import (
    auto_finalize_selected_paths,
    build_attribute_aggregation_plan,
    build_attribute_importance_summary,
    compute_selection_quality_metrics,
    merge_solver_patch,
    print_top_selected_paths,
    sanitize_selected_paths,
)
from phase2_rescoring import llm_rescore_candidate_paths

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


@dataclass
class Phase2DFSFeaturePolicy:
    allowed_table_column_pairs: set[tuple[str, str]]
    allowed_aggs: set[str]
    max_features: int = 256


def _normalize_identifier(text: Any) -> str:
    return str(text or "").strip().lower()


def _normalize_agg(text: Any) -> str:
    t = _normalize_identifier(text)
    if t in {"avg"}:
        return "mean"
    return t


def _build_phase2_dfs_feature_policy(task_spec: dict[str, Any]) -> Phase2DFSFeaturePolicy:
    allowed_pairs: set[tuple[str, str]] = set()
    allowed_aggs: set[str] = set()

    attr_plan = task_spec.get("attribute_aggregation_plan", [])
    if isinstance(attr_plan, dict):
        attr_plan = attr_plan.get("per_path", [])

    if isinstance(attr_plan, list):
        for entry in attr_plan:
            if not isinstance(entry, dict):
                continue
            attrs = entry.get("priority_attributes", [])
            if not isinstance(attrs, list):
                continue
            for attr in attrs:
                if not isinstance(attr, dict):
                    continue
                table_name = _normalize_identifier(attr.get("table"))
                col_name = _normalize_identifier(attr.get("column"))
                if table_name and col_name:
                    allowed_pairs.add((table_name, col_name))
                aggs = attr.get("suggested_aggregations", [])
                if isinstance(aggs, list):
                    for agg in aggs:
                        agg_n = _normalize_agg(agg)
                        if agg_n:
                            allowed_aggs.add(agg_n)

    if not allowed_aggs:
        allowed_aggs = {"count", "mean", "std", "sum", "min", "max"}

    return Phase2DFSFeaturePolicy(
        allowed_table_column_pairs=allowed_pairs,
        allowed_aggs=allowed_aggs,
        max_features=256,
    )


def _phase2_dfs_feature_policy_score(feature_name: str, policy: Phase2DFSFeaturePolicy) -> int:
    lowered = feature_name.lower()
    score = 0

    agg_match = re.search(r"\.([a-z_]+)\(", lowered)
    agg = _normalize_agg(agg_match.group(1)) if agg_match else ""
    if agg and agg in policy.allowed_aggs:
        score += 2

    allowed_tables = {t for t, _ in policy.allowed_table_column_pairs}
    for table_name, col_name in policy.allowed_table_column_pairs:
        if f"({table_name}.{col_name})" in lowered:
            score += 4
            break
    else:
        for table_name in allowed_tables:
            if f"({table_name})" in lowered or f"({table_name}." in lowered:
                score += 1
                break

    return score


class Phase2AwareFastDFSBuilder:
    def __init__(self, base_builder: Any, policy: Phase2DFSFeaturePolicy):
        self._base = base_builder
        self._policy = policy

    def summarize_rows(self, rows: list[dict[str, Any]]) -> list[list[str]]:
        return self._base.summarize_rows(rows)

    def feature_dicts_for_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, str]]:
        raw = self._base.feature_dicts_for_rows(rows)
        filtered_rows: list[dict[str, str]] = []
        for row_dict in raw:
            if not isinstance(row_dict, dict):
                filtered_rows.append({})
                continue
            ranked_items = sorted(
                row_dict.items(),
                key=lambda kv: (
                    -_phase2_dfs_feature_policy_score(str(kv[0]), self._policy),
                    str(kv[0]),
                ),
            )
            filtered_rows.append(dict(ranked_items[: self._policy.max_features]))
        return filtered_rows


def _numeric_feature_map(feature_dict: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in feature_dict.items():
        try:
            fv = float(v)
        except Exception:
            continue
        if np.isnan(fv) or np.isinf(fv):
            continue
        out[str(k)] = fv
    return out


def _build_phase2_train_vector_db_config(dataset: str, task: str) -> dict[str, Any]:
    return {
        "name": "phase2_train_vector_db",
        "batch_size": 1,
        "history_length": 10,
        "history_source": "dataset",
        "history_sampling_strategy": "most_recent_k",
        "history_parallel_mode": "grouped_vectorized",
        "history_parallel_workers": 4,
        "cache_dataset_history_labels": True,
        "cache_train_history_candidates": True,
        "include_future_rows": True,
        "dataset_download": True,
        "task_download": True,
        "datasets": [{"name": dataset, "tasks": [task]}],
    }


def _phase2_dfs_max_depth_from_task_spec(task_spec: dict[str, Any]) -> int:
    depth_policy = task_spec.get("depth_policy", {})
    if not isinstance(depth_policy, dict):
        return 2

    default_depth = depth_policy.get("default_max_depth", 2)
    max_allowed = depth_policy.get("max_allowed_depth", None)

    try:
        chosen = int(default_depth)
    except Exception:
        chosen = 2
    if chosen < 1:
        chosen = 1

    if max_allowed is not None:
        try:
            max_allowed_i = int(max_allowed)
            if max_allowed_i >= 1:
                chosen = min(chosen, max_allowed_i)
        except Exception:
            pass
    return chosen


def _phase2_dfs_prune_plan_from_task_spec(
    task_spec: dict[str, Any],
) -> tuple[set[str] | None, dict[str, set[str]] | None]:
    attr_plan = task_spec.get("attribute_aggregation_plan", [])
    if not isinstance(attr_plan, list) or not attr_plan:
        return None, None

    task_def = task_spec.get("task_definition", {})
    entity_table = _normalize_identifier(task_def.get("entity_table"))
    entity_col = _normalize_identifier(task_def.get("entity_col"))
    time_col = _normalize_identifier(task_def.get("time_col"))
    output_col = _normalize_identifier(task_def.get("output_col"))

    include_tables: set[str] = set()
    include_cols: dict[str, set[str]] = {}

    def _add_col(table_name: str, col_name: str) -> None:
        if not table_name or not col_name:
            return
        include_tables.add(table_name)
        include_cols.setdefault(table_name, set()).add(col_name)

    for entry in attr_plan:
        if not isinstance(entry, dict):
            continue
        path_tables = entry.get("path_tables", [])
        if isinstance(path_tables, list):
            for t in path_tables:
                tname = _normalize_identifier(t)
                if tname:
                    include_tables.add(tname)
        attrs = entry.get("priority_attributes", [])
        if not isinstance(attrs, list):
            continue
        for attr in attrs:
            if not isinstance(attr, dict):
                continue
            _add_col(
                _normalize_identifier(attr.get("table")),
                _normalize_identifier(attr.get("column")),
            )

    if entity_table:
        include_tables.add(entity_table)
        _add_col(entity_table, entity_col)
        _add_col(entity_table, time_col)
        _add_col(entity_table, output_col)

    if not include_tables:
        return None, None
    return include_tables, include_cols


def _build_train_dfs_vector_db(
    *,
    dataset: str,
    task: str,
    task_obj: Any,
    task_spec: dict[str, Any],
    output_dir: Path,
    batch_size: int = 64,
    max_vocab: int = 4096,
    profile_enabled: bool = False,
    profile_chunks: int = 8,
    profile_early_stop: bool = False,
    prune_schema_from_task_spec: bool = False,
) -> dict[str, Any]:
    from fastdfs_context import build_fastdfs_context_builder  # type: ignore
    from inference_history import build_inference_resource  # type: ignore

    task_def = task_spec.get("task_definition", {})
    entity_col = str(task_def.get("entity_col", ""))
    time_col = str(task_def.get("time_col", ""))
    output_col = str(task_def.get("output_col", ""))

    train_table = task_obj.get_table("train", mask_input_cols=False)
    train_df = train_table.df.copy().reset_index(drop=True)
    if train_df.empty:
        db_dir = output_dir / "dfs_train_vector_db"
        db_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "enabled": True,
            "status": "empty_train_split",
            "rows": 0,
            "vector_dim": 0,
            "batch_size": int(batch_size),
            "max_vocab": int(max_vocab),
            "artifact_dir": str(db_dir.resolve()),
        }
        write_json(db_dir / "manifest.json", summary)
        return summary

    resource = build_inference_resource(_build_phase2_train_vector_db_config(dataset, task))
    policy = _build_phase2_dfs_feature_policy(task_spec)
    dfs_max_depth = _phase2_dfs_max_depth_from_task_spec(task_spec)
    include_tables, include_table_columns = (None, None)
    if prune_schema_from_task_spec:
        include_tables, include_table_columns = _phase2_dfs_prune_plan_from_task_spec(task_spec)
    engine_path = output_dir / f"phase2_fastdfs_train_{dataset}_{task}.db"
    base_dfs_builder = build_fastdfs_context_builder(
        resource,
        max_depth=dfs_max_depth,
        engine_path=str(engine_path),
        include_tables=include_tables,
        include_table_columns=include_table_columns,
    )
    dfs_builder = Phase2AwareFastDFSBuilder(base_dfs_builder, policy)

    rows = train_df.to_dict(orient="records")
    row_feature_maps: list[dict[str, float]] = []
    feature_counts: dict[str, int] = {}
    bs = max(1, int(batch_size))
    profiling_rows: list[dict[str, Any]] = []
    prof_limit = max(0, int(profile_chunks))
    t_build_start = time.perf_counter()
    total_dfs_seconds = 0.0
    total_map_seconds = 0.0
    processed_rows = 0
    processed_chunks = 0
    total_chunks = (len(rows) + bs - 1) // bs
    chunk_starts: Any = range(0, len(rows), bs)
    if tqdm is not None:
        chunk_starts = tqdm(
            chunk_starts,
            total=total_chunks,
            desc="[phase2] DFS train vector DB",
            unit="chunk",
        )
    for chunk_idx, start in enumerate(chunk_starts):
        chunk = rows[start : start + bs]
        t0 = time.perf_counter()
        feature_dicts = dfs_builder.feature_dicts_for_rows([dict(r) for r in chunk])
        t1 = time.perf_counter()
        map_start = time.perf_counter()
        for fdict in feature_dicts:
            fmap = _numeric_feature_map(fdict if isinstance(fdict, dict) else {})
            row_feature_maps.append(fmap)
            for key in fmap:
                feature_counts[key] = feature_counts.get(key, 0) + 1
        t2 = time.perf_counter()
        dfs_s = t1 - t0
        map_s = t2 - map_start
        total_dfs_seconds += dfs_s
        total_map_seconds += map_s
        processed_rows += len(chunk)
        processed_chunks += 1
        if profile_enabled and len(profiling_rows) < prof_limit:
            profiling_rows.append(
                {
                    "chunk_index": int(chunk_idx),
                    "rows_in_chunk": int(len(chunk)),
                    "dfs_seconds": float(dfs_s),
                    "numeric_map_seconds": float(map_s),
                    "total_chunk_seconds": float(t2 - t0),
                    "rows_per_second": float(len(chunk) / max(1e-9, (t2 - t0))),
                }
            )
        if profile_enabled and profile_early_stop and prof_limit > 0 and len(profiling_rows) >= prof_limit:
            print(
                "[phase2][profile] early stop after "
                f"{processed_chunks} chunks ({processed_rows} rows sampled)"
            )
            break

    if profile_enabled and profile_early_stop and processed_rows < len(rows):
        db_dir = output_dir / "dfs_train_vector_db"
        db_dir.mkdir(parents=True, exist_ok=True)
        profile_payload = {
            "dataset": dataset,
            "task": task,
            "rows_profiled": int(sum(int(x["rows_in_chunk"]) for x in profiling_rows)),
            "chunks_profiled": int(len(profiling_rows)),
            "profile_chunk_limit": int(prof_limit),
            "batch_size": int(bs),
            "notes": (
                "Sampled profiling rows from early chunks only. "
                "Early-stop mode skipped full DFS train vector DB materialization."
            ),
            "chunk_profiles": profiling_rows,
        }
        profile_path = db_dir / "profile_sample.json"
        write_json(profile_path, profile_payload)
        summary = {
            "enabled": True,
            "status": "profile_early_stop",
            "rows": int(processed_rows),
            "rows_total_train": int(len(rows)),
            "vector_dim": 0,
            "batch_size": int(bs),
            "max_vocab": int(max_vocab),
            "policy_max_features_per_row": int(policy.max_features),
            "policy_dfs_max_depth": int(dfs_max_depth),
            "schema_pruning_enabled": bool(prune_schema_from_task_spec),
            "schema_pruning_tables_kept": (
                int(len(include_tables)) if include_tables is not None else None
            ),
            "policy_allowed_table_column_pairs": int(len(policy.allowed_table_column_pairs)),
            "policy_allowed_aggs": sorted(policy.allowed_aggs),
            "artifact_dir": str(db_dir.resolve()),
            "timing": {
                "build_total_seconds": float(time.perf_counter() - t_build_start),
                "dfs_feature_generation_seconds": float(total_dfs_seconds),
                "numeric_map_seconds": float(total_map_seconds),
                "processed_rows": int(processed_rows),
                "processed_chunks": int(processed_chunks),
                "avg_rows_per_second": float(
                    processed_rows / max(1e-9, total_dfs_seconds + total_map_seconds)
                ),
            },
            "profiling": {
                "enabled": True,
                "early_stop": True,
                "profile_sample_path": str(profile_path.resolve()),
                "chunks_profiled": int(len(profiling_rows)),
            },
        }
        write_json(db_dir / "manifest.json", summary)
        try:
            engine_path.unlink(missing_ok=True)
        except Exception:
            pass
        return summary

    ranked_features = sorted(
        feature_counts.items(),
        key=lambda kv: (-kv[1], kv[0]),
    )
    if int(max_vocab) > 0:
        ranked_features = ranked_features[: int(max_vocab)]
    feature_vocab = [k for k, _ in ranked_features]
    feature_to_idx = {k: i for i, k in enumerate(feature_vocab)}

    n_rows = len(row_feature_maps)
    vectors = np.zeros((n_rows, len(feature_vocab)), dtype=np.float32)
    for i, fmap in enumerate(row_feature_maps):
        for key, value in fmap.items():
            j = feature_to_idx.get(key)
            if j is not None:
                vectors[i, j] = np.float32(value)
    norms = np.linalg.norm(vectors, axis=1).astype(np.float32)

    db_dir = output_dir / "dfs_train_vector_db"
    db_dir.mkdir(parents=True, exist_ok=True)
    np.save(db_dir / "vectors.npy", vectors)
    np.save(db_dir / "norms.npy", norms)
    write_json(
        db_dir / "feature_vocab.json",
        {"features": feature_vocab},
    )

    meta_df = pd.DataFrame(
        {
            "row_index": np.arange(len(train_df), dtype=np.int64),
            "entity": (
                train_df[entity_col].map(lambda x: "" if pd.isna(x) else str(x))
                if entity_col in train_df.columns
                else pd.Series([""] * len(train_df))
            ),
            "time": (
                pd.to_datetime(train_df[time_col], errors="coerce")
                .map(lambda x: "" if pd.isna(x) else str(pd.Timestamp(x)))
                if time_col in train_df.columns
                else pd.Series([""] * len(train_df))
            ),
            "target": (
                train_df[output_col].map(lambda x: "" if pd.isna(x) else str(x))
                if output_col in train_df.columns
                else pd.Series([""] * len(train_df))
            ),
        }
    )
    meta_df.to_csv(db_dir / "row_meta.csv", index=False)

    summary = {
        "enabled": True,
        "status": "ok",
        "rows": int(vectors.shape[0]),
        "vector_dim": int(vectors.shape[1]),
        "nonzero_vectors": int(np.count_nonzero(norms)),
        "batch_size": int(bs),
        "max_vocab": int(max_vocab),
        "policy_max_features_per_row": int(policy.max_features),
        "policy_dfs_max_depth": int(dfs_max_depth),
        "schema_pruning_enabled": bool(prune_schema_from_task_spec),
        "schema_pruning_tables_kept": int(len(include_tables)) if include_tables is not None else None,
        "policy_allowed_table_column_pairs": int(len(policy.allowed_table_column_pairs)),
        "policy_allowed_aggs": sorted(policy.allowed_aggs),
        "artifact_dir": str(db_dir.resolve()),
        "artifacts": {
            "vectors_npy": str((db_dir / "vectors.npy").resolve()),
            "norms_npy": str((db_dir / "norms.npy").resolve()),
            "feature_vocab_json": str((db_dir / "feature_vocab.json").resolve()),
            "row_meta_csv": str((db_dir / "row_meta.csv").resolve()),
        },
        "timing": {
            "build_total_seconds": float(time.perf_counter() - t_build_start),
            "dfs_feature_generation_seconds": float(total_dfs_seconds),
            "numeric_map_seconds": float(total_map_seconds),
            "processed_rows": int(processed_rows),
            "processed_chunks": int(processed_chunks),
            "avg_rows_per_second": float(
                processed_rows / max(1e-9, total_dfs_seconds + total_map_seconds)
            ),
        },
    }
    if profile_enabled and prof_limit > 0:
        profile_payload = {
            "dataset": dataset,
            "task": task,
            "rows_profiled": int(sum(int(x["rows_in_chunk"]) for x in profiling_rows)),
            "chunks_profiled": int(len(profiling_rows)),
            "profile_chunk_limit": int(prof_limit),
            "batch_size": int(bs),
            "notes": (
                "Sampled profiling rows from early chunks only. "
                "Full-train per-row traces are intentionally disabled."
            ),
            "chunk_profiles": profiling_rows,
        }
        profile_path = db_dir / "profile_sample.json"
        write_json(profile_path, profile_payload)
        summary["profiling"] = {
            "enabled": True,
            "profile_sample_path": str(profile_path.resolve()),
            "chunks_profiled": int(len(profiling_rows)),
        }
    else:
        summary["profiling"] = {
            "enabled": False,
            "chunks_profiled": 0,
        }
    write_json(db_dir / "manifest.json", summary)

    try:
        engine_path.unlink(missing_ok=True)
    except Exception:
        pass

    return summary


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        safe: dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(k, tuple):
                key = ".".join(str(x) for x in k)
            else:
                key = str(k)
            safe[key] = _json_safe(v)
        return safe
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, tuple):
        return [_json_safe(x) for x in obj]
    return obj


def _infer_task_path_priors_with_llm(
    llm: Any,
    prompt_obj: dict[str, Any],
) -> dict[str, Any]:
    td = prompt_obj.get("task_definition", {})
    sample_paths = []
    for p in prompt_obj.get("candidate_paths", [])[:80]:
        sample_paths.append(
            {
                "path_id": p.get("path_id"),
                "depth": p.get("depth"),
                "path_tables": p.get("path_tables", []),
            }
        )
    prompt = (
        "Infer task-specific path priors for relational prediction.\n"
        "Output STRICT JSON only with keys:\n"
        "{"
        "\"anchor_table\": str,"
        "\"preferred_motifs\": [[str,...]],"
        "\"optional_motifs\": [[str,...]],"
        "\"discouraged_tables\": [str],"
        "\"preferred_max_depth\": int,"
        "\"rationale\": str"
        "}\n"
        "Rules:\n"
        "- Use task definition + candidate path patterns.\n"
        "- Preferred motifs must be general and reusable, not path IDs.\n"
        "- Keep preferred_motifs <= 8, optional_motifs <= 8.\n"
        f"Task definition: {td}\n"
        f"Candidate path samples: {sample_paths}\n"
    )
    parsed, _ = llm.generate_json(prompt, max_new_tokens=500, retries=1)
    if not isinstance(parsed, dict):
        # Safe fallback: generic anchor-first priors.
        anchor = str(td.get("entity_table", ""))
        out = str(td.get("output_col", ""))
        motifs = [[anchor, "results"], [anchor, "qualifying"], [anchor, "standings"]]
        if out:
            motifs.append([anchor, out])
        return {
            "anchor_table": anchor,
            "preferred_motifs": motifs,
            "optional_motifs": [[anchor, "results", "races"]],
            "discouraged_tables": [],
            "preferred_max_depth": 3,
            "rationale": "fallback_priors",
        }
    parsed.setdefault("anchor_table", str(td.get("entity_table", "")))
    parsed.setdefault("preferred_motifs", [])
    parsed.setdefault("optional_motifs", [])
    parsed.setdefault("discouraged_tables", [])
    parsed.setdefault("preferred_max_depth", 3)
    parsed.setdefault("rationale", "llm_inferred")
    return parsed


def _apply_task_path_priors(
    prompt_obj: dict[str, Any],
    priors: dict[str, Any],
) -> None:
    preferred = priors.get("preferred_motifs", [])
    optional = priors.get("optional_motifs", [])
    discouraged = {str(x).lower() for x in priors.get("discouraged_tables", [])}
    pref_max_depth = int(priors.get("preferred_max_depth", 3))

    def motif_match_score(path_tables: list[str], motif: list[str], weight: float) -> float:
        p = [str(x).lower() for x in path_tables]
        m = [str(x).lower() for x in motif]
        if not m:
            return 0.0
        if len(m) <= len(p) and p[: len(m)] == m:
            return weight + 0.8
        if all(x in p for x in m):
            return weight
        return 0.0

    for p in prompt_obj.get("candidate_paths", []):
        tables = p.get("path_tables", [])
        depth = p.get("depth")
        bonus = 0.0
        if isinstance(preferred, list):
            for m in preferred:
                if isinstance(m, list):
                    bonus += motif_match_score(tables, m, 3.0)
        if isinstance(optional, list):
            for m in optional:
                if isinstance(m, list):
                    bonus += motif_match_score(tables, m, 1.5)
        if any(str(t).lower() in discouraged for t in tables):
            bonus -= 2.0
        if isinstance(depth, int) and depth > pref_max_depth:
            bonus -= 2.0 * (depth - pref_max_depth)
        if isinstance(depth, int) and depth > pref_max_depth + 1:
            # Hard downgrade paths much deeper than inferred preference.
            p["recommended"] = False
        p["task_prior_bonus"] = round(float(bonus), 3)
        p["final_relevance_score"] = round(float(p.get("relevance_score", 0.0)) + bonus, 3)

    prompt_obj["candidate_paths"].sort(
        key=lambda x: (
            float(x.get("final_relevance_score", x.get("relevance_score", 0.0))),
            bool(x.get("recommended", False)),
            bool(x.get("temporal_valid", True)),
        ),
        reverse=True,
    )
    prompt_obj["recommended_path_ids"] = [
        p.get("path_id")
        for p in prompt_obj.get("candidate_paths", [])
        if p.get("recommended", False) and isinstance(p.get("path_id"), str)
    ]


def _apply_task_sql_feature_hints(
    prompt_obj: dict[str, Any],
    feature_hints: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(feature_hints, dict):
        return {"applied": False, "reason": "no_feature_hints"}

    table_weights_raw = feature_hints.get("tables", {})
    column_weights_raw = feature_hints.get("columns", {})
    table_weights: dict[str, float] = {
        str(k).lower(): float(v) for k, v in (table_weights_raw.items() if isinstance(table_weights_raw, dict) else [])
    }

    column_weights: dict[tuple[str, str], float] = {}
    if isinstance(column_weights_raw, dict):
        for k, v in column_weights_raw.items():
            if isinstance(k, (tuple, list)) and len(k) == 2:
                column_weights[(str(k[0]).lower(), str(k[1]).lower())] = float(v)

    touched_paths = 0
    for p in prompt_obj.get("candidate_paths", []):
        tables = [str(t).lower() for t in p.get("path_tables", [])]
        bonus = sum(table_weights.get(t, 0.0) for t in tables) * 0.35
        if bonus > 0:
            touched_paths += 1
        p["sql_hint_bonus"] = round(float(bonus), 3)
        base = float(p.get("final_relevance_score", p.get("relevance_score", 0.0)))
        p["final_relevance_score"] = round(base + bonus, 3)

        attrs = p.get("top_attribute_candidates", [])
        if isinstance(attrs, list):
            for a in attrs:
                t = str(a.get("table", "")).lower()
                c = str(a.get("column", "")).lower()
                c_bonus = column_weights.get((t, c), 0.0) * 0.45
                if c_bonus:
                    a["sql_hint_bonus"] = round(float(c_bonus), 3)
                    a["importance_score"] = round(float(a.get("importance_score", 0.0)) + c_bonus, 3)
            attrs.sort(key=lambda x: float(x.get("importance_score", 0.0)), reverse=True)
            p["top_attribute_candidates"] = attrs[:10]

    prompt_obj["candidate_paths"].sort(
        key=lambda x: (
            float(x.get("final_relevance_score", x.get("relevance_score", 0.0))),
            bool(x.get("recommended", False)),
            bool(x.get("temporal_valid", True)),
        ),
        reverse=True,
    )
    prompt_obj["recommended_path_ids"] = [
        p.get("path_id")
        for p in prompt_obj.get("candidate_paths", [])
        if p.get("recommended", False) and isinstance(p.get("path_id"), str)
    ]

    return {
        "applied": True,
        "tables_with_hints": len(table_weights),
        "columns_with_hints": len(column_weights),
        "touched_paths": touched_paths,
    }


def _make_table_profiles(
    stats: dict[str, Any],
    semantics: dict[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    sem_by_table: dict[str, dict[str, dict[str, Any]]] = {}
    for t in semantics.get("tables", []):
        tname = str(t.get("table_name", ""))
        col_map: dict[str, dict[str, Any]] = {}
        for c in t.get("columns", []):
            cname = str(c.get("name", ""))
            col_map[cname] = {
                "semantic_role": c.get("semantic_role"),
                "leakage_risk": c.get("leakage_risk", False),
                "confidence": c.get("confidence"),
            }
        sem_by_table[tname] = col_map

    stats_by_table: dict[str, dict[str, dict[str, Any]]] = {}
    for t in stats.get("tables", []):
        tname = str(t.get("table_name", ""))
        stats_by_table[tname] = {}
        cols = t.get("columns", {})
        if isinstance(cols, dict):
            for cname, meta in cols.items():
                stats_by_table[tname][str(cname)] = {
                    "numeric": bool(isinstance(meta, dict) and meta.get("numeric")),
                    "missingness": (meta.get("missingness") if isinstance(meta, dict) else None),
                }

    merged: dict[str, dict[str, dict[str, Any]]] = {}
    all_tables = set(sem_by_table) | set(stats_by_table)
    for tname in all_tables:
        merged[tname] = {}
        all_cols = set(sem_by_table.get(tname, {})) | set(stats_by_table.get(tname, {}))
        for cname in all_cols:
            sem_meta = sem_by_table.get(tname, {}).get(cname, {})
            st_meta = stats_by_table.get(tname, {}).get(cname, {})
            merged[tname][cname] = {
                "semantic_role": sem_meta.get("semantic_role"),
                "leakage_risk": sem_meta.get("leakage_risk", False),
                "numeric": st_meta.get("numeric", False),
                "missingness": st_meta.get("missingness"),
            }
    return merged


def _suggest_aggregations(column_name: str, numeric: bool, semantic_role: str | None) -> list[str]:
    cname = column_name.lower()
    if not numeric:
        if semantic_role == "category":
            return ["count", "nunique", "mode"]
        return ["count"]
    if any(k in cname for k in ["position", "order", "grid", "rank"]):
        return ["mean", "std", "min", "max", "last", "delta"]
    if any(k in cname for k in ["point", "win", "lap", "millisecond", "fastest"]):
        return ["mean", "sum", "std", "max", "last", "delta"]
    return ["mean", "std", "min", "max", "sum"]


def _score_column_relevance(
    *,
    table_name: str,
    column_name: str,
    semantic_role: str | None,
    numeric: bool,
    leakage_risk: bool,
    task_name: str,
    output_col: str,
) -> float:
    tname = table_name.lower()
    cname = column_name.lower()
    task_l = task_name.lower()
    out_l = output_col.lower()
    tokens = set(re.findall(r"[a-z0-9]+", f"{task_l} {out_l}"))
    tokens = {t for t in tokens if len(t) >= 3}
    name_tokens = set(re.findall(r"[a-z0-9]+", f"{tname} {cname}"))
    name_tokens = {t for t in name_tokens if len(t) >= 3}
    score = 0.0
    if semantic_role == "measure":
        score += 2.0
    if numeric:
        score += 1.0
    if leakage_risk:
        score -= 4.0
    if semantic_role == "id" or cname.endswith("id"):
        score -= 3.0
    if semantic_role in {"timestamp", "text"}:
        score -= 1.5
    if out_l and out_l in cname:
        score += 4.0
    overlap = len(tokens & name_tokens)
    score += 1.4 * min(overlap, 4)
    return score


def _infer_output_col(task: Any) -> str:
    if getattr(task, "target_col", None) is not None:
        return str(task.target_col)
    if getattr(task, "dst_entity_col", None) is not None:
        return str(task.dst_entity_col)
    raise ValueError(f"Could not infer output column for task={task}")


def _infer_task_type(task: Any) -> str:
    output_col = _infer_output_col(task)
    train_table = task.get_table("train", mask_input_cols=False)
    if output_col not in train_table.df.columns:
        return "unknown"
    series = train_table.df[output_col]
    return "regression" if str(series.dtype).startswith(("float", "int")) else "classification"


def _build_compilation_context(
    *,
    dataset: str,
    task_name: str,
    phase1_bundle: dict[str, dict[str, Any]],
    task_obj: Any,
    budgets: BudgetPolicy,
) -> CompilationContext:
    safety = phase1_bundle["safety_rules"]
    schema = phase1_bundle["schema"]
    stats = phase1_bundle["stats"]
    semantics = phase1_bundle["semantics"]
    semantic_graph = phase1_bundle.get("semantic_context_graph", {})

    train_table = task_obj.get_table("train", mask_input_cols=False)
    time_col = getattr(task_obj, "time_col", train_table.time_col)
    task_def = TaskDefinition(
        entity_table=str(getattr(task_obj, "entity_table", "")),
        entity_col=str(getattr(task_obj, "entity_col", "")),
        time_col=str(time_col),
        output_col=_infer_output_col(task_obj),
        task_type=_infer_task_type(task_obj),
    )

    task_name_l = task_name.lower()
    entity_table_l = task_def.entity_table.lower()
    output_col_l = task_def.output_col.lower()

    def _tokens(text: str) -> set[str]:
        stop = {
            "the",
            "for",
            "and",
            "with",
            "from",
            "task",
            "predict",
            "prediction",
            "next",
            "value",
            "target",
            "entity",
            "table",
            "column",
            "same",
            "using",
        }
        toks = {
            t
            for t in re.findall(r"[a-z0-9]+", str(text).lower())
            if len(t) >= 3 and t not in stop
        }
        expanded = set(toks)
        for t in list(toks):
            if t.endswith("s") and len(t) > 3:
                expanded.add(t[:-1])
            if t.endswith("es") and len(t) > 4:
                expanded.add(t[:-2])
        return expanded

    task_tokens = _tokens(task_name) | _tokens(task_def.output_col)

    table_profiles = _make_table_profiles(stats, semantics)

    table_predictive_score: dict[str, float] = {}
    for tname, cols in table_profiles.items():
        t_low = str(tname).lower()
        t_tokens = _tokens(t_low)
        name_overlap = len(task_tokens & t_tokens)
        measure_hits = 0
        lexical_hits = 0
        for cname, meta in cols.items():
            c_tokens = _tokens(str(cname))
            if task_tokens & c_tokens:
                lexical_hits += 1
            if str(meta.get("semantic_role", "")) == "measure" and not bool(meta.get("leakage_risk", False)):
                measure_hits += 1
        table_predictive_score[tname] = (
            0.9 * min(name_overlap, 3)
            + 0.6 * min(lexical_hits, 6)
            + 0.15 * min(measure_hits, 10)
        )

    def _path_relevance_score(
        path_tables: list[str],
        path_edges: list[dict[str, Any]],
        depth: Any,
        row_mult: Any,
        recommended: bool,
        temporal_valid: bool,
        first_hop_preferred: set[str],
    ) -> float:
        tables = [str(t).lower() for t in path_tables]
        score = 0.0
        if tables and tables[0] == entity_table_l:
            score += 5.0
        if entity_table_l in tables:
            score += 3.0

        # Generic task-conditioned scoring based on output/task token overlap.
        for t in path_tables:
            score += 0.45 * float(table_predictive_score.get(str(t), 0.0))

        if len(path_tables) >= 2:
            hop1 = str(path_tables[1]).lower()
            if hop1 in first_hop_preferred:
                score += 2.5
            else:
                score -= 1.4

        if recommended:
            score += 1.0
        if temporal_valid:
            score += 1.0
        if isinstance(path_edges, list) and path_edges:
            edge_w = [float(e.get("edge_weight", 0.6)) for e in path_edges]
            score += 2.0 * (sum(edge_w) / max(1, len(edge_w)) - 0.5)
            if any(str(e.get("direction", "")).lower() == "parent_to_child" for e in path_edges):
                score -= 0.2

        if isinstance(depth, int):
            # Strongly prefer compact paths for predictive stability.
            score -= 1.25 * max(depth - 2, 0)
        if isinstance(row_mult, (int, float)):
            score -= 0.15 * math.log1p(max(float(row_mult), 0.0))
        # Penalize paths that touch too many distinct tables.
        if len(tables) >= 5:
            score -= 0.6 * (len(tables) - 4)
        return score
    row_count_map = {
        t.get("table_name", ""): int(t.get("row_count", 0))
        for t in stats.get("tables", [])
    }

    adjacency: dict[str, list[dict[str, Any]]] = {}
    for t in schema.get("tables", []):
        tname = str(t.get("table_name", ""))
        adjacency.setdefault(tname, [])

    graph_edges = semantic_graph.get("edges", []) if isinstance(semantic_graph, dict) else []
    used_semantic_graph = False
    if isinstance(graph_edges, list) and graph_edges:
        for e in graph_edges:
            if not isinstance(e, dict):
                continue
            src = str(e.get("src_table", ""))
            dst = str(e.get("dst_table", ""))
            if not src or not dst:
                continue
            ew = float(e.get("edge_weight", 0.6))
            if ew < 0.50:
                continue
            edge = {
                "src_table": src,
                "dst_table": dst,
                "src_column": e.get("src_column"),
                "dst_column": e.get("dst_column"),
                "source": str(e.get("source", "semantic_context_graph")),
                "recommended_by_default": bool(e.get("recommended_by_default", True)),
                "confidence": str(e.get("confidence", "medium")),
                "direction": str(e.get("direction", "unknown")),
                "edge_weight": ew,
            }
            adjacency.setdefault(src, []).append(edge)
            adjacency.setdefault(dst, adjacency.get(dst, []))
            used_semantic_graph = True

    if not used_semantic_graph:
        for t in schema.get("tables", []):
            child = str(t.get("table_name", ""))
            for child_col, parent in (t.get("provided_foreign_keys", {}) or {}).items():
                parent_table = str(parent)
                if not parent_table:
                    continue
                adjacency.setdefault(child, [])
                adjacency.setdefault(parent_table, [])
                edge_fwd = {
                    "src_table": child,
                    "dst_table": parent_table,
                    "src_column": str(child_col),
                    "dst_column": None,
                    "source": "provided_fk",
                    "recommended_by_default": True,
                    "confidence": "high",
                    "direction": "child_to_parent",
                    "edge_weight": 0.85,
                }
                edge_rev = {
                    "src_table": parent_table,
                    "dst_table": child,
                    "src_column": None,
                    "dst_column": str(child_col),
                    "source": "provided_fk_reverse",
                    "recommended_by_default": True,
                    "confidence": "high",
                    "direction": "parent_to_child",
                    "edge_weight": 0.78,
                }
                adjacency[child].append(edge_fwd)
                adjacency[parent_table].append(edge_rev)

    anchor_table = task_def.entity_table
    first_hop_scores: list[tuple[str, float]] = []
    for e in adjacency.get(anchor_table, []):
        dst = str(e.get("dst_table", ""))
        if not dst:
            continue
        ew = float(e.get("edge_weight", 0.6))
        rec = 1.0 if bool(e.get("recommended_by_default", True)) else 0.0
        score = 0.65 * ew + 0.35 * rec + 0.40 * float(table_predictive_score.get(dst, 0.0))
        first_hop_scores.append((dst.lower(), score))
    first_hop_scores.sort(key=lambda x: x[1], reverse=True)
    first_hop_preferred = {t for t, _ in first_hop_scores[:4]}

    max_depth = 4
    max_paths = 900
    enumerated: list[dict[str, Any]] = []

    def _estimate_row_multiplier(path_tables: list[str]) -> float:
        base = max(float(row_count_map.get(anchor_table, 1)), 1.0)
        mult = 1.0
        for tname in path_tables[1:]:
            rc = max(float(row_count_map.get(tname, 1)), 1.0)
            mult *= max(math.sqrt(rc / base), 1.0)
        return float(mult)

    def _dfs(cur_table: str, tables: list[str], edges: list[dict[str, Any]]) -> None:
        depth = len(edges)
        if depth > 0:
            enumerated.append({"path_tables": tables[:], "path_edges": edges[:]})
            if len(enumerated) >= max_paths:
                return
        if depth >= max_depth:
            return
        for e in adjacency.get(cur_table, []):
            nxt = e["dst_table"]
            if nxt in tables:
                continue
            edges.append(e)
            tables.append(nxt)
            _dfs(nxt, tables, edges)
            tables.pop()
            edges.pop()
            if len(enumerated) >= max_paths:
                return

    _dfs(anchor_table, [anchor_table], [])

    candidate_paths: list[dict[str, Any]] = []
    recommended_ids: list[str] = []
    low_conf_excluded = 0
    for idx, p in enumerate(enumerated, start=1):
        pid = f"p{idx:06d}"
        depth = len(p["path_edges"])
        row_mult = _estimate_row_multiplier(p["path_tables"])
        edge_recs = [bool(e.get("recommended_by_default", True)) for e in p["path_edges"]]
        edge_ws = [float(e.get("edge_weight", 0.6)) for e in p["path_edges"]]
        recommended = bool(all(edge_recs) and (sum(edge_ws) / max(1, len(edge_ws)) >= 0.58))
        temporal_valid = True
        compact = {
            "path_id": pid,
            "depth": depth,
            "estimated_row_multiplier": row_mult,
            "path_tables": p["path_tables"],
            "path_edges": p["path_edges"],
            "recommended": recommended,
            "temporal_valid": temporal_valid,
        }
        compact["relevance_score"] = _path_relevance_score(
            compact["path_tables"],
            compact["path_edges"],
            compact["depth"],
            compact["estimated_row_multiplier"],
            recommended,
            temporal_valid,
            first_hop_preferred,
        )

        attr_candidates: list[dict[str, Any]] = []
        for tname in compact["path_tables"]:
            for cname, meta in table_profiles.get(str(tname), {}).items():
                rel = _score_column_relevance(
                    table_name=str(tname),
                    column_name=str(cname),
                    semantic_role=meta.get("semantic_role"),
                    numeric=bool(meta.get("numeric", False)),
                    leakage_risk=bool(meta.get("leakage_risk", False)),
                    task_name=task_name,
                    output_col=task_def.output_col,
                )
                if rel <= 0:
                    continue
                attr_candidates.append(
                    {
                        "table": str(tname),
                        "column": str(cname),
                        "importance_score": round(float(rel), 3),
                        "semantic_role": meta.get("semantic_role"),
                        "numeric": bool(meta.get("numeric", False)),
                        "suggested_aggregations": _suggest_aggregations(
                            str(cname),
                            bool(meta.get("numeric", False)),
                            meta.get("semantic_role"),
                        ),
                    }
                )
        attr_candidates.sort(
            key=lambda x: (float(x.get("importance_score", 0.0)), x.get("numeric", False)),
            reverse=True,
        )
        compact["top_attribute_candidates"] = attr_candidates[:10]
        candidate_paths.append(compact)
        recommended_ids.append(pid)

    # Rank candidate paths by task-aware relevance (phase-2 responsibility).
    candidate_paths.sort(
        key=lambda x: (
            float(x.get("relevance_score", 0.0)),
            bool(x.get("recommended", False)),
            bool(x.get("temporal_valid", True)),
            -int(x.get("depth", 99) if isinstance(x.get("depth"), int) else 99),
        ),
        reverse=True,
    )
    # Collapse near-duplicate permutations by table-family signature.
    deduped: list[dict[str, Any]] = []
    seen_family: set[tuple[Any, ...]] = set()
    for p in candidate_paths:
        tables = p.get("path_tables", [])
        if not isinstance(tables, list) or not tables:
            continue
        family = (str(tables[0]).lower(), tuple(sorted(str(t).lower() for t in tables[1:])))
        if family in seen_family:
            continue
        seen_family.add(family)
        deduped.append(p)
    candidate_paths = deduped[:220]
    recommended_ids = [p["path_id"] for p in candidate_paths if p.get("recommended")]

    semantics_summary = {
        "annotation_model": semantics.get("annotation_model", {}),
        "table_roles": {
            t.get("table_name", ""): t.get("table_role", "")
            for t in semantics.get("tables", [])
        },
        "semantic_context_graph": {
            "used_for_enumeration": used_semantic_graph,
            "node_count": int(semantic_graph.get("node_count", 0)) if isinstance(semantic_graph, dict) else 0,
            "edge_count": int(semantic_graph.get("edge_count", 0)) if isinstance(semantic_graph, dict) else 0,
        },
    }

    stats_summary = {
        "table_row_counts": {
            t.get("table_name", ""): t.get("row_count", 0)
            for t in stats.get("tables", [])
        },
        "global_top_correlations_count": len(stats.get("global_top_correlations", [])),
        "task_table_priors": {
            "task_tokens": sorted(task_tokens),
            "first_hop_preferred": sorted(first_hop_preferred),
            "top_predictive_tables": [
                t
                for t, _ in sorted(
                    table_predictive_score.items(),
                    key=lambda kv: float(kv[1]),
                    reverse=True,
                )[:8]
            ],
        },
    }

    cutoff = str(safety.get("global_cutoff_time", ""))

    return CompilationContext(
        dataset=dataset,
        task=task_name,
        task_definition=task_def,
        phase1_artifacts_dir="",
        cutoff_time=cutoff,
        candidate_paths=candidate_paths,
        recommended_path_ids=recommended_ids,
        low_confidence_excluded_count=low_conf_excluded,
        semantics_summary=semantics_summary,
        stats_summary=stats_summary,
        safety_rules=safety.get("rules", {}),
    )


def _context_to_prompt_obj(ctx: CompilationContext, budgets: BudgetPolicy) -> dict[str, Any]:
    return {
        "dataset": ctx.dataset,
        "task": ctx.task,
        "task_definition": {
            "entity_table": ctx.task_definition.entity_table,
            "entity_col": ctx.task_definition.entity_col,
            "time_col": ctx.task_definition.time_col,
            "output_col": ctx.task_definition.output_col,
            "task_type": ctx.task_definition.task_type,
            "anchor_entity_table": ctx.task_definition.entity_table,
            "anchor_entity_col": ctx.task_definition.entity_col,
        },
        "global_cutoff_time": ctx.cutoff_time,
        "safety_rules": ctx.safety_rules,
        "candidate_paths": ctx.candidate_paths,
        "recommended_path_ids": ctx.recommended_path_ids,
        "low_confidence_excluded_count": ctx.low_confidence_excluded_count,
        "budget_defaults": {
            "self_history_budget": budgets.self_history_budget,
            "one_hop_budget": budgets.one_hop_budget,
            "multi_hop_budget_total": budgets.multi_hop_budget_total,
            "neighbor_budget": budgets.neighbor_budget,
        },
        "semantics_summary": ctx.semantics_summary,
        "stats_summary": ctx.stats_summary,
        "task_sql_feature_hints": {},
    }


def _default_policy(prompt_obj: dict[str, Any], budgets: BudgetPolicy) -> dict[str, Any]:
    ranked_paths = prompt_obj.get("candidate_paths", [])
    rec_paths = [
        p.get("path_id")
        for p in ranked_paths
        if p.get("recommended", False) and p.get("temporal_valid", True) is True
    ]
    rec_paths = [p for p in rec_paths if isinstance(p, str)]
    return {
        "path_scoring_rules": {
            "selection_filters": {
                "require_temporal_valid": True,
                "prefer_recommended_paths": True,
            },
            "ranking_formula": {
                "semantic_relevance_weight": 0.45,
                "depth_penalty_weight": 0.20,
                "cardinality_risk_penalty_weight": 0.25,
                "leakage_risk_penalty_weight": 0.10,
            },
            "path_exclusions": {
                "exclude_low_confidence_fk_by_default": True,
            },
            "exclude_low_confidence_fk_by_default": True,
            "selected_path_ids": rec_paths[: min(40, len(rec_paths))],
        },
        "feature_rules": {
            "allowed_aggregations": ["count", "mean", "min", "max", "std", "sum"],
            "temporal_features": ["recency_count", "recent_mean", "recent_std"],
            "trend_features": ["delta", "slope_proxy"],
            "blocked_features": ["future_leakage_columns"],
        },
        "depth_policy": {
            "default_max_depth": 2,
            "escalation_conditions": [
                "high_semantic_relevance",
                "manageable_cardinality",
                "no_temporal_risk",
            ],
            "max_allowed_depth": 8,
        },
        "budget_policy": {
            "self_history_budget": budgets.self_history_budget,
            "one_hop_budget": budgets.one_hop_budget,
            "multi_hop_budget_total": budgets.multi_hop_budget_total,
            "neighbor_budget": budgets.neighbor_budget,
        },
        "safety_constraints": {
            "global_temporal_cutoff": prompt_obj.get("global_cutoff_time"),
            "no_future_rows": True,
            "exclude_low_confidence_fk_by_default": True,
            "leakage_prevention_rules": [
                "no_columns_with_target_or_future_patterns",
                "no_information_beyond_cutoff",
            ],
            "forbidden_columns": ["label", "target", "future", "leak", "outcome"],
        },
    }


def _build_critic_context(
    prompt_obj: dict[str, Any],
    candidate_spec: dict[str, Any],
    *,
    round_index: int | None = None,
) -> dict[str, Any]:
    selected_ids = (
        candidate_spec.get("path_scoring_rules", {}).get("selected_path_ids", [])
    )
    if not isinstance(selected_ids, list):
        selected_ids = []
    selected_ids = selected_ids[:40]

    by_id = {p.get("path_id"): p for p in prompt_obj.get("candidate_paths", [])}
    selected_paths = []
    for pid in selected_ids[:20]:
        p = by_id.get(pid)
        if not p:
            continue
        selected_paths.append(
            {
                "path_id": pid,
                "final_relevance_score": p.get(
                    "final_relevance_score", p.get("relevance_score", 0.0)
                ),
                "llm_task_relevance": p.get("llm_task_relevance", 0.0),
                "llm_confidence": p.get("llm_confidence", 0.0),
                "depth": p.get("depth"),
                "estimated_row_multiplier": p.get("estimated_row_multiplier"),
                "path_tables": p.get("path_tables", []),
                "recommended": p.get("recommended", False),
                "temporal_valid": p.get("temporal_valid", True),
            }
        )

    preferred_max_depth = int(
        candidate_spec.get("depth_policy", {}).get(
            "default_max_depth",
            prompt_obj.get("task_path_priors", {}).get("preferred_max_depth", 3),
        )
    )
    selection_metrics = compute_selection_quality_metrics(
        selected_path_ids=selected_ids,
        prompt_obj=prompt_obj,
        preferred_max_depth=preferred_max_depth,
    )

    return {
        "dataset": prompt_obj.get("dataset"),
        "task": prompt_obj.get("task"),
        "round_index": int(round_index) if isinstance(round_index, int) else None,
        "task_definition": prompt_obj.get("task_definition"),
        "global_cutoff_time": prompt_obj.get("global_cutoff_time"),
        "safety_rules": prompt_obj.get("safety_rules"),
        "selected_path_ids": selected_ids,
        "selected_path_count": len(selected_ids),
        "selected_path_summaries": selected_paths,
        "top_unselected_candidates": [
            p
            for p in prompt_obj.get("candidate_paths", [])
            if p.get("path_id") not in set(selected_ids)
        ][:20],
        "selected_set_quality": selection_metrics,
        "task_sql_feature_hints": prompt_obj.get("task_sql_feature_hints", {}),
        "budget_defaults": prompt_obj.get("budget_defaults"),
    }


def _extract_flagged_path_ids(
    critic_obj: dict[str, Any],
    *,
    selected_ids: set[str] | None = None,
) -> set[str]:
    selected_ids = selected_ids or set()
    flagged: set[str] = set()

    majors = critic_obj.get("major_issues", [])
    if isinstance(majors, list):
        for m in majors:
            if isinstance(m, dict):
                affected = m.get("affected_path_ids", [])
                if isinstance(affected, list):
                    for pid in affected:
                        spid = str(pid)
                        if re.fullmatch(r"p\d{6}", spid):
                            flagged.add(spid)
                else:
                    text = " ".join(str(v) for v in m.values())
                    flagged.update(re.findall(r"p\d{6}", text))
            else:
                flagged.update(re.findall(r"p\d{6}", str(m)))

    # Conservative fallback only if nothing structured found.
    if not flagged:
        text = str(critic_obj.get("major_issues", ""))
        flagged.update(re.findall(r"p\d{6}", text))

    # Never drop paths that are not currently selected.
    if selected_ids:
        flagged = {pid for pid in flagged if pid in selected_ids}
    return flagged


def _normalize_major_issues(critic_obj: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    majors = critic_obj.get("major_issues", [])
    if not isinstance(majors, list):
        majors = [majors]
    for i, m in enumerate(majors, start=1):
        if isinstance(m, dict):
            out.append(
                {
                    "issue_id": str(m.get("issue_id", f"M{i}")),
                    "issue": str(m.get("issue", "")).strip() or str(m),
                }
            )
        else:
            out.append({"issue_id": f"M{i}", "issue": str(m)})
    return out


def _filter_round1_invalid_major_issues(
    *,
    round_index: int,
    major_issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if round_index > 1:
        return major_issues
    filtered: list[dict[str, Any]] = []
    for m in major_issues:
        text = str(m.get("issue", "")).lower()
        if "degraded vs previous round" in text or "previous round" in text:
            continue
        filtered.append(m)
    return filtered


def _count_addressed_issues(
    solver_patch: dict[str, Any],
    prior_major_issues: list[dict[str, Any]],
) -> tuple[int, int]:
    addressed = solver_patch.get("addressed_issues", [])
    if not isinstance(addressed, list):
        addressed = []
    resolved_ids: set[str] = set()
    for a in addressed:
        if not isinstance(a, dict):
            continue
        status = str(a.get("status", "")).lower()
        if status != "resolved":
            continue
        iid = str(a.get("issue_id", "")).strip()
        if iid:
            resolved_ids.add(iid)
    total = len(prior_major_issues)
    done = sum(1 for m in prior_major_issues if m.get("issue_id") in resolved_ids)
    return done, total


def _print_temporal_leakage_examples(
    *,
    round_index: int,
    critic_obj: dict[str, Any],
    prompt_obj: dict[str, Any],
    max_examples: int = 3,
) -> None:
    major_issues = critic_obj.get("major_issues", [])
    if not isinstance(major_issues, list):
        major_issues = [str(major_issues)]
    temporal_lines = [m for m in major_issues if "temporal leakage" in str(m).lower()]
    if not temporal_lines:
        return

    by_id = {p.get("path_id"): p for p in prompt_obj.get("candidate_paths", [])}
    all_ids: list[str] = []
    for line in temporal_lines:
        ids = re.findall(r"p\d{6}", str(line))
        all_ids.extend(ids)

    if not all_ids:
        print(f"[phase2][round {round_index}] temporal_leakage_example: no path_id parsed")
        print(f"[phase2][round {round_index}] temporal_leakage_issue_text={temporal_lines[0]}")
        return

    seen: set[str] = set()
    examples: list[str] = []
    for pid in all_ids:
        if pid in seen:
            continue
        seen.add(pid)
        path_meta = by_id.get(pid, {})
        depth = path_meta.get("depth", "?")
        temporal_valid = path_meta.get("temporal_valid", "?")
        recommended = path_meta.get("recommended", "?")
        tables = path_meta.get("path_tables", [])
        tables_text = "->".join(str(t) for t in tables[:4]) if isinstance(tables, list) else "?"
        examples.append(
            f"{pid} depth={depth} temporal_valid={temporal_valid} "
            f"recommended={recommended} tables={tables_text}"
        )
        if len(examples) >= max_examples:
            break

    print(f"[phase2][round {round_index}] temporal_leakage_examples:")
    for ex in examples:
        print(f"[phase2][round {round_index}]   - {ex}")


def _sanitize_selected_paths(
    policy: dict[str, Any],
    prompt_obj: dict[str, Any],
    *,
    max_keep: int = 40,
    drop_ids: set[str] | None = None,
) -> list[str]:
    return sanitize_selected_paths(
        policy=policy,
        prompt_obj=prompt_obj,
        max_keep=max_keep,
        drop_ids=drop_ids,
    )

def _build_attribute_aggregation_plan(
    *,
    selected_path_ids: list[str],
    prompt_obj: dict[str, Any],
    max_paths: int = 20,
    max_attrs_per_path: int = 5,
) -> list[dict[str, Any]]:
    return build_attribute_aggregation_plan(
        selected_path_ids=selected_path_ids,
        prompt_obj=prompt_obj,
        max_paths=max_paths,
        max_attrs_per_path=max_attrs_per_path,
    )


def _build_attribute_importance_summary(
    *,
    selected_path_ids: list[str],
    prompt_obj: dict[str, Any],
    max_attributes: int = 40,
) -> dict[str, Any]:
    return build_attribute_importance_summary(
        selected_path_ids=selected_path_ids,
        prompt_obj=prompt_obj,
        max_attributes=max_attributes,
    )

def _auto_finalize_selected_paths(
    *,
    prompt_obj: dict[str, Any],
    current_selected: list[str],
    task_priors: dict[str, Any],
    max_keep: int = 10,
) -> list[str]:
    return auto_finalize_selected_paths(
        prompt_obj=prompt_obj,
        current_selected=current_selected,
        task_priors=task_priors,
        max_keep=max_keep,
    )

def _print_top_selected_paths(
    *,
    prompt_obj: dict[str, Any],
    selected_path_ids: list[str],
    top_k: int = 10,
) -> None:
    return print_top_selected_paths(
        prompt_obj=prompt_obj,
        selected_path_ids=selected_path_ids,
        top_k=top_k,
    )

def _llm_rescore_candidate_paths(
    *,
    llm: Any,
    prompt_obj: dict[str, Any],
    max_candidates: int = 220,
    batch_size: int = 20,
) -> dict[str, Any]:
    return llm_rescore_candidate_paths(
        llm=llm,
        prompt_obj=prompt_obj,
        max_candidates=max_candidates,
        batch_size=batch_size,
    )

def _merge_solver_patch(base_policy: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    return merge_solver_patch(base_policy, patch)

def compile_task_spec(args: argparse.Namespace) -> dict[str, Any]:
    from relbench.tasks import get_task
    from phase2_llm import LocalJsonLLM
    try:
        from task_history_queries import get_task_history_feature_hints
    except Exception:
        get_task_history_feature_hints = None

    phase1_dir = Path(args.phase1_artifacts_dir)
    # Reduce noisy non-fatal warnings from featuretools/woodwork during repeated DFS calls.
    warnings.filterwarnings(
        "ignore",
        message="Could not infer format, so each element will be parsed individually",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="Logical type Integer for child column .* does not match parent column .*",
        category=UserWarning,
    )
    if not phase1_dir.exists():
        alt = _RFM_ROOT / phase1_dir
        if alt.exists():
            phase1_dir = alt
    if args.task_output_dir is not None:
        output_dir = Path(args.task_output_dir)
    else:
        output_dir = Path(args.output_dir) / args.dataset / args.task
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_phase1_bundle(phase1_dir, args.dataset)
    task_obj = get_task(args.dataset, args.task, download=True)

    budgets = BudgetPolicy(
        self_history_budget=args.self_history_budget,
        one_hop_budget=args.one_hop_budget,
        multi_hop_budget_total=args.multi_hop_budget_total,
        neighbor_budget=args.neighbor_budget,
    )

    ctx = _build_compilation_context(
        dataset=args.dataset,
        task_name=args.task,
        phase1_bundle=bundle,
        task_obj=task_obj,
        budgets=budgets,
    )
    prompt_obj = _context_to_prompt_obj(ctx, budgets)
    scg_meta = (
        prompt_obj.get("semantics_summary", {})
        .get("semantic_context_graph", {})
    )
    print(
        "[phase2] semantic_context_graph:"
        f" used_for_enumeration={scg_meta.get('used_for_enumeration', False)}"
        f" nodes={scg_meta.get('node_count', 0)}"
        f" edges={scg_meta.get('edge_count', 0)}"
    )

    llm = LocalJsonLLM(model_size=args.model_size, print_log=True)
    sql_hint_info = {"applied": False, "reason": "feature_hint_loader_unavailable"}
    if get_task_history_feature_hints is not None:
        hints = get_task_history_feature_hints(args.dataset, args.task)
        prompt_obj["task_sql_feature_hints"] = _json_safe(hints or {})
        sql_hint_info = {
            "applied": True,
            "tables_with_hints": len((hints or {}).get("tables", {}) if isinstance(hints, dict) else {}),
            "columns_with_hints": len((hints or {}).get("columns", {}) if isinstance(hints, dict) else {}),
            "touched_paths": len(prompt_obj.get("candidate_paths", [])),
        }
        print(
            "[phase2] task_sql_feature_hints:"
            f" applied={sql_hint_info.get('applied')}"
            f" touched_paths={sql_hint_info.get('touched_paths', 0)}"
            f" hinted_tables={sql_hint_info.get('tables_with_hints', 0)}"
            f" hinted_columns={sql_hint_info.get('columns_with_hints', 0)}"
        )
    else:
        prompt_obj["task_sql_feature_hints"] = {}

    task_priors = _infer_task_path_priors_with_llm(llm, prompt_obj)
    prompt_obj["task_path_priors"] = task_priors
    print(
        "[phase2] task_path_priors:"
        f" anchor={task_priors.get('anchor_table')}"
        f" preferred_max_depth={task_priors.get('preferred_max_depth')}"
        f" preferred_motifs={len(task_priors.get('preferred_motifs', []))}"
        f" optional_motifs={len(task_priors.get('optional_motifs', []))}"
    )
    llm_rescore_info = _llm_rescore_candidate_paths(
        llm=llm,
        prompt_obj=prompt_obj,
        max_candidates=args.llm_path_score_max_candidates,
        batch_size=args.llm_path_score_batch_size,
    )
    print(
        "[phase2] llm_path_rescoring:"
        f" scored={llm_rescore_info.get('scored', 0)}"
        f" matched={llm_rescore_info.get('matched_scores', 0)}"
        f" returned_items={llm_rescore_info.get('returned_score_items', 0)}"
        f" batches={llm_rescore_info.get('batches', 0)}"
        f" failed_batches={llm_rescore_info.get('failed_batches', 0)}"
    )

    solver_rounds_path = output_dir / "solver_rounds.jsonl"
    critic_rounds_path = output_dir / "critic_rounds.jsonl"
    if solver_rounds_path.exists():
        solver_rounds_path.unlink()
    if critic_rounds_path.exists():
        critic_rounds_path.unlink()

    policy = _default_policy(prompt_obj, budgets)
    critic_feedback = None
    stop_reason = "max_rounds_reached"
    major_issue_count = 0
    round_count = 0
    pending_drop_ids: set[str] = set()
    prev_selected_signature: tuple[str, ...] | None = None
    unchanged_rounds = 0
    prior_major_issues: list[dict[str, Any]] = []

    for r in range(1, args.max_rounds + 1):
        round_count = r
        print(f"[phase2][round {r}] solver starting...")
        solver_prompt = build_solver_prompt(
            context=prompt_obj,
            previous_spec=policy,
            critic_feedback=critic_feedback,
        )
        solver_patch, solver_raw = llm.generate_json(
            solver_prompt,
            max_new_tokens=1024,
            retries=2,
        )
        solver_ok = isinstance(solver_patch, dict)
        if not solver_ok:
            solver_patch = {}
        addressed_done, addressed_total = _count_addressed_issues(
            solver_patch, prior_major_issues
        )
        policy = _merge_solver_patch(policy, solver_patch)
        selected_now = _sanitize_selected_paths(
            policy,
            prompt_obj,
            max_keep=40,
            drop_ids=pending_drop_ids,
        )

        sig = tuple(selected_now)
        if prev_selected_signature is not None and sig == prev_selected_signature:
            unchanged_rounds += 1
        else:
            unchanged_rounds = 0
        prev_selected_signature = sig

        if unchanged_rounds >= 2 and selected_now:
            stricter = selected_now[: min(20, len(selected_now))]
            policy["path_scoring_rules"]["selected_path_ids"] = stricter
            selected_now = stricter
            print(
                f"[phase2][round {r}] convergence_guard activated: "
                f"pruned selected_path_ids to {len(selected_now)}"
            )

        print(
            f"[phase2][round {r}] solver parsed_ok={solver_ok} "
            f"selected_path_ids={len(selected_now)}"
        )
        if addressed_total > 0:
            print(
                f"[phase2][round {r}] solver addressed_issues="
                f"{addressed_done}/{addressed_total}"
            )

        append_jsonl(
            solver_rounds_path,
            {
                "round_index": r,
                "role": "solver",
                "model_name": args.model_size,
                "parsed_ok": solver_ok,
                "prompt_preview": solver_prompt[:1200],
                "raw_output_preview": solver_raw[:1200],
                "payload": solver_patch,
            },
        )

        critic_context = _build_critic_context(prompt_obj, policy, round_index=r)
        critic_prompt = build_critic_prompt(context=critic_context, candidate_spec=policy)
        print(f"[phase2][round {r}] critic starting...")
        critic_obj, critic_raw = llm.generate_json(
            critic_prompt,
            max_new_tokens=768,
            retries=2,
        )
        critic_ok = isinstance(critic_obj, dict)
        if not critic_ok:
            critic_obj = {
                "no_major_issues": False,
                "major_issues": ["critic_parse_failed"],
                "minor_issues": [],
                "fixes": [],
            }

        normalized_majors = _normalize_major_issues(critic_obj)
        normalized_majors = _filter_round1_invalid_major_issues(
            round_index=r,
            major_issues=normalized_majors,
        )
        critic_obj["major_issues"] = normalized_majors
        major_issues = [m.get("issue", "") for m in normalized_majors]
        minor_issues = critic_obj.get("minor_issues", [])
        if not isinstance(minor_issues, list):
            minor_issues = [str(minor_issues)]
        major_issue_count = len(major_issues)
        no_major = bool(critic_obj.get("no_major_issues", False))
        if major_issue_count == 0:
            # Normalize inconsistent critic payloads:
            # if there are no major issues, treat as no_major_issues=True.
            no_major = True
            critic_obj["no_major_issues"] = True
        print(
            f"[phase2][round {r}] critic parsed_ok={critic_ok} "
            f"no_major_issues={no_major} "
            f"major={len(major_issues)} minor={len(minor_issues)}"
        )
        if major_issues:
            print(f"[phase2][round {r}] critic first_major_issue={major_issues[0]}")
            _print_temporal_leakage_examples(
                round_index=r,
                critic_obj=critic_obj,
                prompt_obj=prompt_obj,
                max_examples=3,
            )
            prior_major_issues = normalized_majors
        else:
            prior_major_issues = []
        flagged_ids = _extract_flagged_path_ids(
            critic_obj,
            selected_ids=set(selected_now),
        )
        if flagged_ids:
            pending_drop_ids.update(flagged_ids)
            print(
                f"[phase2][round {r}] critic flagged path_ids={len(flagged_ids)} "
                f"(will drop next round)"
            )

        append_jsonl(
            critic_rounds_path,
            {
                "round_index": r,
                "role": "critic",
                "model_name": args.model_size,
                "parsed_ok": critic_ok,
                "prompt_preview": critic_prompt[:1200],
                "raw_output_preview": critic_raw[:1200],
                "payload": critic_obj,
            },
        )

        critic_feedback = critic_obj
        if no_major or major_issue_count == 0:
            stop_reason = "critic_no_major_issues"
            print(f"[phase2][round {r}] stopping early: {stop_reason}")
            break

    # Final deterministic compact re-selection to prevent noisy end states.
    finalized_selected = _auto_finalize_selected_paths(
        prompt_obj=prompt_obj,
        current_selected=policy.get("path_scoring_rules", {}).get("selected_path_ids", []),
        task_priors=task_priors,
        max_keep=10,
    )
    policy.setdefault("path_scoring_rules", {})
    policy["path_scoring_rules"]["selected_path_ids"] = finalized_selected
    print(f"[phase2] auto_finalize_selected_paths kept={len(finalized_selected)}")
    final_selection_metrics = compute_selection_quality_metrics(
        selected_path_ids=finalized_selected,
        prompt_obj=prompt_obj,
        preferred_max_depth=int(policy.get("depth_policy", {}).get("default_max_depth", 3)),
    )

    task_spec = {
        "artifact_version": "phase2.v1",
        "dataset_name": args.dataset,
        "task_name": args.task,
        "generation_timestamp_utc": utc_now_iso(),
        "source_artifacts": {
            "phase1_artifacts_dir": str((phase1_dir / args.dataset).resolve()),
            "global_cutoff_time": bundle["safety_rules"].get("global_cutoff_time"),
        },
        "task_definition": {
            "entity_table": ctx.task_definition.entity_table,
            "entity_col": ctx.task_definition.entity_col,
            "time_col": ctx.task_definition.time_col,
            "output_col": ctx.task_definition.output_col,
            "task_type": ctx.task_definition.task_type,
        },
        **policy,
        "attribute_aggregation_plan": _build_attribute_aggregation_plan(
            selected_path_ids=policy.get("path_scoring_rules", {}).get("selected_path_ids", []),
            prompt_obj=prompt_obj,
            max_paths=20,
            max_attrs_per_path=5,
        ),
        "solver_critic_diagnostics": {
            "rounds": round_count,
            "major_issues_count": major_issue_count,
            "stop_reason": stop_reason,
            "model_name": args.model_size,
            "selection_quality": final_selection_metrics,
        },
        "validation": {},
    }

    candidate_path_ids = {p.get("path_id") for p in prompt_obj.get("candidate_paths", [])}
    candidate_path_ids.discard(None)
    schema_issues = validate_task_spec_schema(task_spec)
    consistency_issues = validate_task_spec_consistency(
        task_spec,
        candidate_path_ids=candidate_path_ids,
        phase1_cutoff=str(bundle["safety_rules"].get("global_cutoff_time")),
    )
    all_issues = schema_issues + consistency_issues
    task_spec["validation"] = {
        "schema_valid": len(schema_issues) == 0,
        "consistency_checks": consistency_issues,
        "warnings": all_issues,
    }
    if args.build_dfs_train_vector_db:
        print("[phase2] building train DFS vector DB artifact...")
        db_summary = _build_train_dfs_vector_db(
            dataset=args.dataset,
            task=args.task,
            task_obj=task_obj,
            task_spec=task_spec,
            output_dir=output_dir,
            batch_size=max(1, int(args.dfs_train_vector_db_batch_size)),
            max_vocab=max(0, int(args.dfs_train_vector_db_max_vocab)),
            profile_enabled=bool(args.profile_dfs_train_vector_db),
            profile_chunks=max(0, int(args.profile_dfs_train_vector_db_chunks)),
            profile_early_stop=bool(args.profile_dfs_train_vector_db_early_stop),
            prune_schema_from_task_spec=bool(args.dfs_train_vector_db_prune_schema),
        )
        task_spec["dfs_train_vector_db"] = db_summary
        print(
            "[phase2] train DFS vector DB:"
            f" status={db_summary.get('status')}"
            f" rows={db_summary.get('rows', 0)}"
            f" dim={db_summary.get('vector_dim', 0)}"
        )
    else:
        task_spec["dfs_train_vector_db"] = {
            "enabled": False,
            "status": "disabled_by_flag",
        }

    write_json(output_dir / "task_spec.json", task_spec)
    write_json(
        output_dir / "attribute_importance.json",
        _build_attribute_importance_summary(
            selected_path_ids=policy.get("path_scoring_rules", {}).get("selected_path_ids", []),
            prompt_obj=prompt_obj,
            max_attributes=50,
        ),
    )
    write_json(
        output_dir / "final_validation_report.json",
        {
            "schema_issues": schema_issues,
            "consistency_issues": consistency_issues,
            "all_issues": all_issues,
            "passed": len(all_issues) == 0,
        },
    )
    _print_top_selected_paths(
        prompt_obj=prompt_obj,
        selected_path_ids=policy.get("path_scoring_rules", {}).get("selected_path_ids", []),
        top_k=10,
    )
    return task_spec


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 2 task compilation pipeline")
    parser.add_argument("--dataset", type=str, default="rel-f1")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--phase1-artifacts-dir", type=Path, default=Path("RFM/v2/phase1/artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("RFM/v2/phase2/artifacts"))
    parser.add_argument(
        "--task-output-dir",
        type=Path,
        default=None,
        help=(
            "If provided, write this task's outputs directly to this exact directory "
            "(no automatic /<dataset>/<task> suffix)."
        ),
    )
    parser.add_argument("--model-size", type=str, default="8b")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--self-history-budget", type=int, default=10)
    parser.add_argument("--one-hop-budget", type=int, default=10)
    parser.add_argument("--multi-hop-budget-total", type=int, default=50)
    parser.add_argument("--neighbor-budget", type=int, default=10)
    parser.add_argument("--llm-path-score-max-candidates", type=int, default=220)
    parser.add_argument("--llm-path-score-batch-size", type=int, default=20)
    parser.add_argument(
        "--build-dfs-train-vector-db",
        dest="build_dfs_train_vector_db",
        action="store_true",
        help="Build train-split DFS feature vector DB artifact for fast retrieval in inference.",
    )
    parser.add_argument(
        "--no-build-dfs-train-vector-db",
        dest="build_dfs_train_vector_db",
        action="store_false",
        help="Disable building train-split DFS feature vector DB artifact.",
    )
    parser.set_defaults(build_dfs_train_vector_db=True)
    parser.add_argument("--dfs-train-vector-db-batch-size", type=int, default=64)
    parser.add_argument(
        "--dfs-train-vector-db-max-vocab",
        type=int,
        default=4096,
        help="Maximum DFS feature vocabulary size for vector DB (0 means all).",
    )
    parser.add_argument(
        "--profile-dfs-train-vector-db",
        action="store_true",
        help=(
            "Write sampled per-chunk timing profile for train DFS vector DB build "
            "(compact JSON, not full-train traces)."
        ),
    )
    parser.add_argument(
        "--profile-dfs-train-vector-db-chunks",
        type=int,
        default=8,
        help="Number of initial chunks to sample in DFS vector DB profiling.",
    )
    parser.add_argument(
        "--profile-dfs-train-vector-db-early-stop",
        action="store_true",
        help=(
            "With --profile-dfs-train-vector-db enabled, stop DFS train vector DB build "
            "after sampled profile chunks instead of processing full train split."
        ),
    )
    parser.add_argument(
        "--dfs-train-vector-db-prune-schema",
        action="store_true",
        help=(
            "Conservatively prune DFS schema to task-spec path/attribute tables+columns "
            "before building train vector DB."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.max_rounds < 1:
        raise ValueError("--max-rounds must be >= 1")
    if args.dfs_train_vector_db_batch_size < 1:
        raise ValueError("--dfs-train-vector-db-batch-size must be >= 1")
    if args.dfs_train_vector_db_max_vocab < 0:
        raise ValueError("--dfs-train-vector-db-max-vocab must be >= 0")
    if args.profile_dfs_train_vector_db_chunks < 0:
        raise ValueError("--profile-dfs-train-vector-db-chunks must be >= 0")
    if args.profile_dfs_train_vector_db_early_stop and not args.profile_dfs_train_vector_db:
        raise ValueError(
            "--profile-dfs-train-vector-db-early-stop requires --profile-dfs-train-vector-db"
        )
    if args.dry_run:
        print("[phase2] dry run requested; exiting before compile.")
        return
    spec = compile_task_spec(args)
    print("Phase 2 completed.")
    print(f"- dataset={spec['dataset_name']}")
    print(f"- task={spec['task_name']}")
    print(f"- rounds={spec['solver_critic_diagnostics']['rounds']}")
    print(f"- stop_reason={spec['solver_critic_diagnostics']['stop_reason']}")


if __name__ == "__main__":
    main()
