from __future__ import annotations

import argparse
import pdb
import json
import os
import queue
import re
import sys
import tempfile
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Reuse existing RFM/code modules.
_RFM_ROOT = Path(__file__).resolve().parents[2]
_RFM_CODE_DIR = _RFM_ROOT / "code"
if str(_RFM_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_RFM_CODE_DIR))

from fastdfs_context import build_fastdfs_context_builder  # type: ignore
from grag import RelBenchGraphRAGStore, RowFeatureExtractor, build_zero_shot_prompt  # type: ignore
from inference_history import build_inference_resource, build_raw_dataset_timestamps  # type: ignore
from relbench.tasks import get_task  # type: ignore
from task_history_queries import TASK_HISTORY_FEATURE_HINTS_REGISTRY  # type: ignore
from zero_shot_llm import LocalLLM, MODEL_PATHS, extract_number  # type: ignore


print_prompt = False

@dataclass
class Phase2FeaturePolicy:
    allowed_table_column_pairs: set[tuple[str, str]]
    allowed_aggs: set[str]
    max_features: int = 256


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_identifier(text: Any) -> str:
    return str(text or "").strip().lower()


def _normalize_agg(text: Any) -> str:
    t = _normalize_identifier(text)
    if t in {"avg"}:
        return "mean"
    return t


def _phase2_task_dir(phase2_artifacts_dir: Path, dataset: str, task: str) -> Path:
    return phase2_artifacts_dir / dataset / task


def _build_phase2_feature_policy(task_spec: dict[str, Any]) -> Phase2FeaturePolicy:
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

    return Phase2FeaturePolicy(
        allowed_table_column_pairs=allowed_pairs,
        allowed_aggs=allowed_aggs,
        max_features=256,
    )


def _inject_phase2_hints_into_registry(
    *,
    dataset: str,
    task: str,
    task_spec: dict[str, Any],
    attr_importance: dict[str, Any] | None,
) -> None:
    table_weights: dict[str, int] = {}
    column_weights: dict[tuple[str, str], int] = {}

    selected_paths = set(task_spec.get("path_scoring_rules", {}).get("selected_path_ids", []))
    attr_plan = task_spec.get("attribute_aggregation_plan", [])
    if isinstance(attr_plan, list):
        for path_entry in attr_plan:
            if not isinstance(path_entry, dict):
                continue
            pid = path_entry.get("path_id")
            if selected_paths and pid not in selected_paths:
                continue
            tables = path_entry.get("path_tables", [])
            if isinstance(tables, list):
                for t in tables:
                    tk = _normalize_identifier(t)
                    if tk:
                        table_weights[tk] = table_weights.get(tk, 0) + 1
            attrs = path_entry.get("priority_attributes", [])
            if isinstance(attrs, list):
                for a in attrs:
                    if not isinstance(a, dict):
                        continue
                    tk = _normalize_identifier(a.get("table"))
                    ck = _normalize_identifier(a.get("column"))
                    if tk:
                        table_weights[tk] = table_weights.get(tk, 0) + 2
                    if tk and ck:
                        column_weights[(tk, ck)] = column_weights.get((tk, ck), 0) + 3

    if isinstance(attr_importance, dict):
        for row in attr_importance.get("top_attributes", [])[:120]:
            if not isinstance(row, dict):
                continue
            tk = _normalize_identifier(row.get("table"))
            ck = _normalize_identifier(row.get("column"))
            score = float(row.get("importance_score_sum", 0.0) or 0.0)
            bump = max(1, int(round(score)))
            if tk:
                table_weights[tk] = table_weights.get(tk, 0) + bump
            if tk and ck:
                column_weights[(tk, ck)] = column_weights.get((tk, ck), 0) + 2 * bump

    TASK_HISTORY_FEATURE_HINTS_REGISTRY[(dataset, task)] = {
        "tables": dict(sorted(table_weights.items(), key=lambda kv: (-kv[1], kv[0]))),
        "columns": dict(sorted(column_weights.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def _feature_policy_score(feature_name: str, policy: Phase2FeaturePolicy) -> int:
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
    def __init__(self, base_builder: Any, policy: Phase2FeaturePolicy):
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
                    -_feature_policy_score(str(kv[0]), self._policy),
                    str(kv[0]),
                ),
            )
            filtered_rows.append(dict(ranked_items[: self._policy.max_features]))
        return filtered_rows


@dataclass
class EvalItem:
    row_index: int
    entity_value: Any
    query_time: pd.Timestamp
    query_row: dict[str, Any]
    history_rows: list[dict[str, Any]]


@dataclass
class StaticSimilarityIndex:
    entity_keys: list[str]
    key_to_idx: dict[str, int]
    num_matrix: np.ndarray
    num_norms: np.ndarray
    cat_matrix: np.ndarray
    num_cols: list[str]
    cat_cols: list[str]


@dataclass
class TrainDFSVectorDBIndex:
    vectors: np.ndarray
    norms: np.ndarray
    feature_vocab: list[str]
    feature_to_idx: dict[str, int]
    row_meta: pd.DataFrame


def _build_single_task_inference_config(
    *,
    dataset: str,
    task: str,
    history_length: int,
    history_sampling_strategy: str,
) -> dict[str, Any]:
    return {
        "name": "phase3_dfs_only",
        "batch_size": 1,
        "history_length": int(history_length),
        "history_source": "dataset",
        "history_sampling_strategy": str(history_sampling_strategy),
        "history_parallel_mode": "grouped_vectorized",
        "history_parallel_workers": 4,
        "cache_dataset_history_labels": True,
        "cache_train_history_candidates": True,
        "include_future_rows": True,
        "dataset_download": True,
        "task_download": True,
        "datasets": [{"name": dataset, "tasks": [task]}],
    }


def _resolve_self_history_min_gap(task_object: Any) -> pd.Timedelta:
    raw_gap = getattr(task_object, "timedelta", None)
    if raw_gap is None:
        return pd.Timedelta(0)
    try:
        parsed = pd.to_timedelta(raw_gap)
    except Exception:
        return pd.Timedelta(0)
    if pd.isna(parsed) or parsed <= pd.Timedelta(0):
        return pd.Timedelta(0)
    return parsed


def _build_example_pool(
    resource: Any,
    *,
    entity_col: str,
    time_col: str,
    output_col: str,
    max_query_time: pd.Timestamp,
) -> pd.DataFrame:
    required = {entity_col, time_col, output_col}
    timestamps = build_raw_dataset_timestamps(resource.db)
    if timestamps.empty:
        return pd.DataFrame(columns=[entity_col, time_col, output_col])

    timestamps = timestamps[pd.to_datetime(timestamps, errors="coerce") < max_query_time]
    if timestamps.empty:
        return pd.DataFrame(columns=[entity_col, time_col, output_col])

    history_table = resource.task_object.make_table(resource.db, pd.Series(timestamps))
    pooled = history_table.df.copy()
    if not required.issubset(set(pooled.columns)):
        return pd.DataFrame(columns=[entity_col, time_col, output_col])

    pooled[time_col] = pd.to_datetime(pooled[time_col], errors="coerce")
    pooled = pooled.dropna(subset=[time_col]).reset_index(drop=True)
    pooled = pooled.sort_values(by=[time_col], kind="stable").reset_index(drop=True)
    pooled = pooled.drop_duplicates(subset=[entity_col, time_col], keep="first").reset_index(drop=True)
    return pooled


def _sample_history_before_query(
    *,
    pool_df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    output_col: str,
    entity_value: Any,
    query_time: pd.Timestamp,
    history_length: int,
    min_history_gap: pd.Timedelta,
) -> list[dict[str, Any]]:
    if pool_df.empty:
        return []
    if min_history_gap > pd.Timedelta(0):
        safe_cutoff = query_time - min_history_gap
        subset = pool_df[
            (pool_df[entity_col] == entity_value)
            & (pool_df[time_col] <= safe_cutoff)
        ]
    else:
        subset = pool_df[
            (pool_df[entity_col] == entity_value) & (pool_df[time_col] < query_time)
        ]
    if subset.empty:
        return []
    subset = subset.sort_values(by=[time_col], kind="stable")
    subset = subset.tail(max(1, int(history_length)))
    records = subset.to_dict(orient="records")
    # Mark as self rows so DFS table renderer places them in self section.
    for rec in records:
        rec["__example_scope"] = "self"
    return records


def _entity_key(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return ""
        vf = float(value)
        if vf.is_integer():
            return str(int(vf))
        return str(vf)
    return str(value).strip()


def _build_static_similarity_index(resource: Any) -> StaticSimilarityIndex | None:
    table = resource.db.table_dict.get(resource.entity_table)
    if table is None:
        return None
    df = table.df.copy()
    if df.empty:
        return None

    entity_col = resource.entity_col if resource.entity_col in df.columns else table.pkey_col
    if entity_col is None or entity_col not in df.columns:
        return None

    # Keep one static snapshot per entity.
    df = df.drop_duplicates(subset=[entity_col], keep="last").reset_index(drop=True)
    df["__entity_key"] = df[entity_col].map(_entity_key)
    df = df[df["__entity_key"] != ""]
    if df.empty:
        return None

    excluded = {entity_col}
    if table.time_col and table.time_col in df.columns:
        excluded.add(table.time_col)
    if resource.time_col in df.columns:
        excluded.add(resource.time_col)
    candidate_cols = [c for c in df.columns if c not in excluded and c != "__entity_key"]

    num_cols: list[str] = []
    cat_cols: list[str] = []
    for col in candidate_cols:
        s = df[col]
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            num_cols.append(col)
        else:
            nunique = int(s.nunique(dropna=True))
            if 1 <= nunique <= 200:
                cat_cols.append(col)

    num_matrix = np.zeros((len(df), 0), dtype=np.float32)
    num_norms = np.zeros((len(df),), dtype=np.float32)
    if num_cols:
        num_df = df[num_cols].apply(pd.to_numeric, errors="coerce")
        med = num_df.median(axis=0)
        std = num_df.std(axis=0).replace(0, np.nan).fillna(1.0)
        normed = (num_df.fillna(med) - med) / std
        num_matrix = normed.to_numpy(dtype=np.float32, copy=True)
        num_norms = np.linalg.norm(num_matrix, axis=1).astype(np.float32)

    cat_matrix = np.empty((len(df), 0), dtype=object)
    if cat_cols:
        cat_df = (
            df[cat_cols]
            .fillna("__MISSING__")
            .astype(str)
            .apply(lambda col: col.str.strip().str.lower())
        )
        cat_matrix = cat_df.to_numpy(dtype=object, copy=True)

    entity_keys = df["__entity_key"].astype(str).tolist()
    key_to_idx = {k: i for i, k in enumerate(entity_keys)}
    return StaticSimilarityIndex(
        entity_keys=entity_keys,
        key_to_idx=key_to_idx,
        num_matrix=num_matrix,
        num_norms=num_norms,
        cat_matrix=cat_matrix,
        num_cols=num_cols,
        cat_cols=cat_cols,
    )


def _top_static_similar_entities(
    index: StaticSimilarityIndex,
    *,
    entity_value: Any,
    k: int,
    exclude_keys: set[str],
) -> list[str]:
    query_key = _entity_key(entity_value)
    q_idx = index.key_to_idx.get(query_key)
    if q_idx is None or k <= 0:
        return []

    n = len(index.entity_keys)
    sim = np.zeros((n,), dtype=np.float32)
    has_num = index.num_matrix.shape[1] > 0
    has_cat = index.cat_matrix.shape[1] > 0

    if has_num:
        qv = index.num_matrix[q_idx]
        qn = float(np.linalg.norm(qv))
        if qn > 0:
            denom = (index.num_norms * qn) + 1e-8
            cos = np.dot(index.num_matrix, qv) / denom
            sim_num = np.clip((cos + 1.0) * 0.5, 0.0, 1.0)
        else:
            sim_num = np.zeros((n,), dtype=np.float32)
    else:
        sim_num = np.zeros((n,), dtype=np.float32)

    if has_cat:
        q_cat = index.cat_matrix[q_idx]
        valid = q_cat != "__missing__"
        if np.any(valid):
            eq = (index.cat_matrix[:, valid] == q_cat[valid])
            sim_cat = np.mean(eq, axis=1).astype(np.float32)
        else:
            sim_cat = np.zeros((n,), dtype=np.float32)
    else:
        sim_cat = np.zeros((n,), dtype=np.float32)

    if has_num and has_cat:
        sim = (0.7 * sim_num) + (0.3 * sim_cat)
    elif has_num:
        sim = sim_num
    elif has_cat:
        sim = sim_cat

    for key in exclude_keys | {query_key}:
        idx = index.key_to_idx.get(key)
        if idx is not None:
            sim[idx] = -1.0

    ranked_idx = np.argsort(-sim)
    out: list[str] = []
    for idx in ranked_idx:
        if len(out) >= k:
            break
        if sim[idx] < 0:
            continue
        out.append(index.entity_keys[int(idx)])
    return out


def _select_other_entity_examples(
    *,
    graph_store: RelBenchGraphRAGStore,
    resource: Any,
    example_pool_df: pd.DataFrame,
    entity_value: Any,
    cutoff_time: pd.Timestamp,
    max_rows: int,
    neighbor_entity_count: int,
    neighbor_history_count: int,
    neighbor_search_hops: int,
    top_k: int,
    static_similarity_index: StaticSimilarityIndex | None,
    use_static_similar_others: bool,
) -> list[dict[str, Any]]:
    if (
        max_rows <= 0
        or neighbor_entity_count <= 0
        or neighbor_history_count <= 0
        or example_pool_df.empty
    ):
        return []

    query_entity_key = _entity_key(entity_value)
    pool = example_pool_df[example_pool_df[resource.time_col] < cutoff_time].copy()
    if pool.empty:
        return []
    pool["__entity_key"] = pool[resource.entity_col].map(_entity_key)
    pool = pool[pool["__entity_key"] != query_entity_key]
    if pool.empty:
        return []

    neighbor_rank_by_entity: dict[str, tuple[int, int]] = {}
    start_node_id = graph_store.node_id_map.get((resource.entity_table, entity_value))
    if start_node_id is not None:
        graph_neighbors = graph_store.get_multihop_neighbors_before(
            start_node=int(start_node_id),
            cutoff_time=cutoff_time,
            num_hops=max(1, int(neighbor_search_hops)),
            top_k=max(int(top_k), int(neighbor_entity_count) * 24),
        )
        for nbr in graph_neighbors:
            dst_id = int(nbr["dst"])
            dst_table, dst_pk = graph_store.rev_node_id[dst_id]
            if dst_table != resource.entity_table or dst_pk == entity_value:
                continue
            ek = _entity_key(dst_pk)
            hop = int(nbr.get("hop", 10**9))
            ts_val = int(nbr.get("ts", -1))
            prev = neighbor_rank_by_entity.get(ek)
            if prev is None or hop < prev[0] or (hop == prev[0] and ts_val > prev[1]):
                neighbor_rank_by_entity[ek] = (hop, ts_val)

    selected_entity_keys: list[str] = []
    if neighbor_rank_by_entity:
        ranked_neighbor_entities = sorted(
            neighbor_rank_by_entity.items(),
            key=lambda item: (item[1][0], -item[1][1], item[0]),
        )
        selected_entity_keys = [k for k, _ in ranked_neighbor_entities[:neighbor_entity_count]]

    # If graph neighbors are sparse, augment with static-similar anchors.
    if (
        use_static_similar_others
        and static_similarity_index is not None
        and len(selected_entity_keys) < neighbor_entity_count
    ):
        pool_entity_keys = set(pool["__entity_key"].astype(str).tolist())
        remaining = max(0, neighbor_entity_count - len(selected_entity_keys))
        extra_keys = _top_static_similar_entities(
            static_similarity_index,
            entity_value=entity_value,
            k=max(remaining * 3, remaining),
            exclude_keys=set(selected_entity_keys),
        )
        for key in extra_keys:
            if len(selected_entity_keys) >= neighbor_entity_count:
                break
            if key not in pool_entity_keys or key in selected_entity_keys:
                continue
            selected_entity_keys.append(key)

    if not selected_entity_keys:
        # Task-agnostic cold-start fallback:
        # If graph neighbors are unavailable, use recent same-type peer entities
        # before cutoff so the prompt still has analogical evidence.
        latest_per_entity = (
            pool.sort_values(by=[resource.time_col], ascending=False, kind="stable")
            .drop_duplicates(subset=["__entity_key"], keep="first")
        )
        if not latest_per_entity.empty:
            selected_entity_keys = (
                latest_per_entity["__entity_key"]
                .head(neighbor_entity_count)
                .astype(str)
                .tolist()
            )
    if not selected_entity_keys:
        return []

    records: list[dict[str, Any]] = []
    for ek in selected_entity_keys:
        entity_rows = pool[pool["__entity_key"] == ek]
        if entity_rows.empty:
            continue
        recent_rows = (
            entity_rows.sort_values(by=[resource.time_col], ascending=False, kind="stable")
            .head(neighbor_history_count)
            .sort_values(by=[resource.time_col], kind="stable")
        )
        for rec in recent_rows.to_dict(orient="records"):
            rec.pop("__entity_key", None)
            rec["__example_scope"] = "other"
            records.append(rec)

    records.sort(key=lambda rec: pd.Timestamp(rec[resource.time_col]))
    if len(records) > max_rows:
        records = records[-max_rows:]
    return records


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


def _query_feature_cache_key(entity_value: Any, cutoff_time: Any) -> tuple[str, int]:
    return (_entity_key(entity_value), int(pd.Timestamp(cutoff_time).value))


def _load_train_dfs_vector_db_index(
    *,
    phase2_task_dir: Path,
    resource: Any,
) -> TrainDFSVectorDBIndex | None:
    db_dir = phase2_task_dir / "dfs_train_vector_db"
    if not db_dir.exists():
        return None

    vectors_path = db_dir / "vectors.npy"
    norms_path = db_dir / "norms.npy"
    vocab_path = db_dir / "feature_vocab.json"
    row_meta_path = db_dir / "row_meta.csv"
    if not (vectors_path.exists() and norms_path.exists() and vocab_path.exists() and row_meta_path.exists()):
        return None

    vectors = np.load(vectors_path)
    norms = np.load(norms_path)
    vocab_json = _read_json(vocab_path)
    feature_vocab = vocab_json.get("features", [])
    if not isinstance(feature_vocab, list):
        return None
    feature_vocab = [str(x) for x in feature_vocab]
    row_meta = pd.read_csv(row_meta_path)

    if vectors.ndim != 2:
        return None
    if norms.ndim != 1:
        return None
    if vectors.shape[0] != norms.shape[0]:
        return None
    if vectors.shape[1] != len(feature_vocab):
        return None
    if len(row_meta) != vectors.shape[0]:
        return None

    # Normalize/validate metadata columns to match runtime retrieval fields.
    required_meta = {"entity", "time", "target"}
    if not required_meta.issubset(set(row_meta.columns)):
        return None
    row_meta = row_meta.copy()
    parsed_time = pd.to_datetime(row_meta["time"], errors="coerce")
    valid_time_mask = parsed_time.notna().to_numpy()
    if not np.all(valid_time_mask):
        vectors = vectors[valid_time_mask]
        norms = norms[valid_time_mask]
        row_meta = row_meta.loc[valid_time_mask].reset_index(drop=True)
        parsed_time = pd.to_datetime(row_meta["time"], errors="coerce")
    row_meta["time"] = parsed_time

    row_meta = row_meta.rename(
        columns={
            "entity": resource.entity_col,
            "time": resource.time_col,
            "target": resource.output_col,
        }
    )
    row_meta["__entity_key"] = row_meta[resource.entity_col].map(_entity_key)

    return TrainDFSVectorDBIndex(
        vectors=vectors.astype(np.float32, copy=False),
        norms=norms.astype(np.float32, copy=False),
        feature_vocab=feature_vocab,
        feature_to_idx={k: i for i, k in enumerate(feature_vocab)},
        row_meta=row_meta,
    )


def _vectorize_query_features(
    query_features: dict[str, float],
    *,
    feature_to_idx: dict[str, int],
    dim: int,
) -> np.ndarray:
    out = np.zeros((dim,), dtype=np.float32)
    for key, value in query_features.items():
        idx = feature_to_idx.get(key)
        if idx is None:
            continue
        out[idx] = np.float32(value)
    return out


def _select_other_entity_examples_by_train_vector_db(
    *,
    resource: Any,
    index: TrainDFSVectorDBIndex,
    query_features: dict[str, float],
    entity_value: Any,
    cutoff_time: pd.Timestamp,
    max_rows: int,
    neighbor_entity_count: int,
    neighbor_history_count: int,
) -> list[dict[str, Any]]:
    if (
        max_rows <= 0
        or neighbor_entity_count <= 0
        or neighbor_history_count <= 0
    ):
        return []

    query_entity_key = _entity_key(entity_value)
    meta = index.row_meta
    meta = meta[meta["__entity_key"] != ""]
    if meta.empty:
        return []
    meta = meta[meta[resource.time_col] < cutoff_time]
    if meta.empty:
        return []

    selected_entity_keys: list[str] = []

    # Primary path: DFS-vector similarity retrieval.
    if query_features and index.vectors.size > 0:
        query_vec = _vectorize_query_features(
            query_features,
            feature_to_idx=index.feature_to_idx,
            dim=index.vectors.shape[1],
        )
        query_norm = float(np.linalg.norm(query_vec))
        if query_norm > 0:
            sim = np.dot(index.vectors, query_vec) / ((index.norms * query_norm) + 1e-8)
            valid_mask = index.norms > 0
            if np.any(valid_mask):
                sim = np.where(valid_mask, sim, -1.0)
                sim_idx = meta.index.to_numpy(dtype=np.int64)
                sim_sub = sim[sim_idx]
                if sim_sub.size > 0:
                    ranked_local = np.argsort(-sim_sub)
                    seen: set[str] = set()
                    for local_i in ranked_local:
                        row_i = int(sim_idx[int(local_i)])
                        ek = str(meta.at[row_i, "__entity_key"])
                        if not ek or ek == query_entity_key or ek in seen:
                            continue
                        seen.add(ek)
                        selected_entity_keys.append(ek)
                        if len(selected_entity_keys) >= neighbor_entity_count:
                            break

    # Task-agnostic fallback:
    # When query DFS features are empty/zero (or similarity retrieval yields no peers),
    # select recent peer entities before cutoff that have non-empty targets.
    if not selected_entity_keys:
        fallback_meta = meta[meta["__entity_key"] != query_entity_key].copy()
        if not fallback_meta.empty:
            output_series = fallback_meta[resource.output_col]
            nonempty_mask = output_series.notna()
            if pd.api.types.is_string_dtype(output_series):
                nonempty_mask = nonempty_mask & (output_series.astype(str).str.strip() != "")
            fallback_meta = fallback_meta[nonempty_mask]
            if not fallback_meta.empty:
                latest_per_entity = (
                    fallback_meta.sort_values(
                        by=[resource.time_col],
                        ascending=False,
                        kind="stable",
                    )
                    .drop_duplicates(subset=["__entity_key"], keep="first")
                )
                selected_entity_keys = (
                    latest_per_entity["__entity_key"]
                    .head(neighbor_entity_count)
                    .astype(str)
                    .tolist()
                )

    if not selected_entity_keys:
        return []

    records: list[dict[str, Any]] = []
    for ek in selected_entity_keys:
        entity_rows = meta[meta["__entity_key"] == ek]
        if entity_rows.empty:
            continue
        recent_rows = (
            entity_rows.sort_values(by=[resource.time_col], ascending=False, kind="stable")
            .head(neighbor_history_count)
            .sort_values(by=[resource.time_col], kind="stable")
        )
        for rec in recent_rows.to_dict(orient="records"):
            out = {
                resource.entity_col: rec.get(resource.entity_col),
                resource.time_col: rec.get(resource.time_col),
                resource.output_col: rec.get(resource.output_col),
                "__example_scope": "other",
            }
            records.append(out)

    records.sort(key=lambda rec: pd.Timestamp(rec[resource.time_col]))
    if len(records) > max_rows:
        records = records[-max_rows:]
    return records


def _prepare_prompt_pack_for_eval_items(
    *,
    batch_items: list[EvalItem],
    dfs_builder: Any,
    graph_store: RelBenchGraphRAGStore,
    extractor: RowFeatureExtractor,
    resource: Any,
    args: argparse.Namespace,
) -> tuple[list[str], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    per_item_entry_rows: list[list[dict[str, Any]]] = []
    for item in batch_items:
        entry_rows = [dict(x) for x in item.history_rows] + [dict(item.query_row)]
        per_item_entry_rows.append(entry_rows)
        all_rows.extend(entry_rows)

    all_feature_dicts = dfs_builder.feature_dicts_for_rows(all_rows)
    cursor = 0
    prompts: list[str] = []
    outputs_meta: list[dict[str, Any]] = []
    for item, entry_rows in zip(batch_items, per_item_entry_rows):
        n = len(entry_rows)
        precomputed_feature_dicts = all_feature_dicts[cursor : cursor + n]
        cursor += n
        precomputed_summaries = [[] for _ in entry_rows]

        prompt = build_zero_shot_prompt(
            store=graph_store,
            extractor=extractor,
            resource=resource,
            query_row=item.query_row,
            history_rows=item.history_rows,
            top_k=max(1, int(args.top_k)),
            num_hops=max(1, int(args.num_hops)),
            include_hop_aggregation=False,
            include_semantic_retrieval=False,
            use_dfs=True,
            dfs_context_builder=dfs_builder,
            context_workers=1,
            precomputed_entry_dfs_summaries=precomputed_summaries,
            precomputed_entry_dfs_feature_dicts=precomputed_feature_dicts,
            recent_context_k=1,
            include_neighbors=True,
            include_dfs_summary=True,
            include_dfs_table=True,
            other_neighbor_entity_count=max(0, int(args.other_neighbor_entity_count)),
            other_neighbor_history_count=max(0, int(args.other_neighbor_history_count)),
        )
        prompts.append(_make_strict_numeric_prompt(prompt))
        outputs_meta.append(
            {
                "row_index": item.row_index,
                "entity": item.entity_value,
                "timestamp": str(item.query_time),
                "target": item.query_row.get(resource.output_col),
                "history_len": len(item.history_rows),
            }
        )
    return prompts, outputs_meta


def _run_llm_for_prompt_pack(
    *,
    llm: LocalLLM,
    prompts: list[str],
    outputs_meta: list[dict[str, Any]],
    llm_batch_size: int,
    max_new_tokens: int,
    print_predictions: bool,
    breakpoint_on_nonpositive: bool,
    breakpoint_large_gap_threshold: float,
    predictions: list[float],
    targets: list[float],
    records: list[dict[str, Any]],
) -> None:
    for i in range(0, len(prompts), llm_batch_size):
        prompt_batch = prompts[i : i + llm_batch_size]
        meta_batch = outputs_meta[i : i + llm_batch_size]
        raw_outputs = llm.generate_numeric_batch(
            prompt_batch,
            max_new_tokens=max(4, min(12, int(max_new_tokens))),
        )
        for meta, raw_text, used_prompt in zip(meta_batch, raw_outputs, prompt_batch):
            pred = float(extract_number(raw_text))
            try:
                target = float(meta["target"])
            except Exception:
                target = float("nan")
            if not np.isnan(target):
                predictions.append(pred)
                targets.append(target)
            records.append(
                {
                    **meta,
                    "prediction": pred,
                    "raw_generation": raw_text,
                    "abs_error": (abs(pred - target) if not np.isnan(target) else None),
                }
            )

            
            if print_predictions:
                if print_prompt:
                    print(used_prompt)
                print(f"item_index={meta['row_index']}")
                print(f"  entity={meta['entity']}")
                print(f"  timestamp={meta['timestamp']}")
                print(f"  prediction={pred} | ground_truth={meta['target']}")

                if print_prompt and pred - meta['target'] > 200:
                    breakpoint()

                # if pred == 0.0:
                #     raw_preview = (raw_text or "").strip().replace("\n", "\\n")
                #     if len(raw_preview) > 240:
                #         raw_preview = raw_preview[:237] + "..."
                #     print(f"  raw_generation='{raw_preview}'")
            if _is_debug_breakpoint_prediction(
                pred,
                target,
                large_gap_threshold=breakpoint_large_gap_threshold,
            ) and breakpoint_on_nonpositive:
                print(
                    f"[phase3][debug] breakpoint prediction detected "
                    f"(item_index={meta['row_index']}, prediction={pred})."
                )
                print("[phase3][debug] full_prompt_start")
                print(used_prompt)
                print("[phase3][debug] full_prompt_end")
                print(
                    "[phase3][debug] entering pdb "
                    f"(prediction <= 0 or abs gap >= {breakpoint_large_gap_threshold})"
                )
                pdb.set_trace()


def _build_query_feature_cache_for_rows(
    *,
    rows: list[dict[str, Any]],
    resource: Any,
    dfs_builder: Any,
) -> dict[tuple[str, int], dict[str, float]]:
    if not rows:
        return {}
    feature_dicts = dfs_builder.feature_dicts_for_rows([dict(r) for r in rows])
    cache: dict[tuple[str, int], dict[str, float]] = {}
    for row, fdict in zip(rows, feature_dicts):
        cache_key = _query_feature_cache_key(
            row.get(resource.entity_col),
            row.get(resource.time_col),
        )
        cache[cache_key] = _numeric_feature_map(fdict)
    return cache


def _dfs_feature_similarity(
    query_features: dict[str, float],
    candidate_features: dict[str, float],
) -> tuple[float, int]:
    if not query_features or not candidate_features:
        return -1.0, 0
    shared = sorted(set(query_features.keys()) & set(candidate_features.keys()))
    if not shared:
        return -1.0, 0

    rel_errors = []
    for k in shared:
        qv = query_features[k]
        cv = candidate_features[k]
        denom = max(abs(qv), 1e-6)
        rel_errors.append(abs(cv - qv) / denom)
    mean_rel_err = float(np.mean(rel_errors)) if rel_errors else 1e9

    # Smooth bounded similarity from relative error.
    value_sim = float(np.exp(-mean_rel_err))
    overlap = len(shared) / max(1, len(query_features))
    score = (0.75 * value_sim) + (0.25 * overlap)
    return score, len(shared)


def _select_other_entity_examples_by_dfs_similarity(
    *,
    resource: Any,
    example_pool_df: pd.DataFrame,
    query_row: dict[str, Any],
    entity_value: Any,
    cutoff_time: pd.Timestamp,
    max_rows: int,
    neighbor_entity_count: int,
    neighbor_history_count: int,
    dfs_builder: Any,
    cutoff_cache: dict[int, dict[str, Any]] | None = None,
    cutoff_cache_size: int = 8,
    query_features_override: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    if (
        max_rows <= 0
        or neighbor_entity_count <= 0
        or neighbor_history_count <= 0
        or example_pool_df.empty
    ):
        return []

    query_entity_key = _entity_key(entity_value)
    cutoff_ns = int(pd.Timestamp(cutoff_time).value)
    cache_entry: dict[str, Any] | None = None
    if cutoff_cache is not None:
        cache_entry = cutoff_cache.get(cutoff_ns)

    if cache_entry is None:
        pool_all = example_pool_df[example_pool_df[resource.time_col] < cutoff_time].copy()
        if pool_all.empty:
            return []
        pool_all["__entity_key"] = pool_all[resource.entity_col].map(_entity_key)
        if pool_all.empty:
            return []

        # Representative candidate row: latest pre-cutoff row per entity.
        reps = (
            pool_all.sort_values(by=[resource.time_col], ascending=False, kind="stable")
            .drop_duplicates(subset=["__entity_key"], keep="first")
            .reset_index(drop=True)
        )
        if reps.empty:
            return []
        rep_rows = reps.to_dict(orient="records")
        rep_feature_dicts = dfs_builder.feature_dicts_for_rows([dict(r) for r in rep_rows])
        rep_numeric_features = [_numeric_feature_map(fdict) for fdict in rep_feature_dicts]
        cache_entry = {
            "pool_all": pool_all,
            "rep_rows": rep_rows,
            "rep_numeric_features": rep_numeric_features,
        }
        if cutoff_cache is not None and cutoff_cache_size > 0:
            cutoff_cache[cutoff_ns] = cache_entry
            while len(cutoff_cache) > int(cutoff_cache_size):
                oldest_key = next(iter(cutoff_cache))
                cutoff_cache.pop(oldest_key, None)

    pool_all = cache_entry["pool_all"]
    rep_rows = cache_entry["rep_rows"]
    rep_numeric_features = cache_entry["rep_numeric_features"]

    query_features = query_features_override
    if query_features is None:
        query_feature_dicts = dfs_builder.feature_dicts_for_rows([dict(query_row)])
        if not query_feature_dicts:
            return []
        query_features = _numeric_feature_map(query_feature_dicts[0])
    if not query_features:
        return []

    ranked: list[tuple[float, int, pd.Timestamp, str]] = []
    for rep_row, cand_features in zip(rep_rows, rep_numeric_features):
        ek = str(rep_row.get("__entity_key", ""))
        if not ek or ek == query_entity_key:
            continue
        score, shared_count = _dfs_feature_similarity(query_features, cand_features)
        if score < 0:
            continue
        ts = pd.Timestamp(rep_row[resource.time_col])
        ranked.append((score, shared_count, ts, ek))

    if not ranked:
        return []

    ranked.sort(key=lambda x: (-x[0], -x[1], -x[2].value, x[3]))
    selected_entity_keys = [ek for _, _, _, ek in ranked[:neighbor_entity_count]]
    if not selected_entity_keys:
        return []

    records: list[dict[str, Any]] = []
    for ek in selected_entity_keys:
        entity_rows = pool_all[pool_all["__entity_key"] == ek]
        if entity_rows.empty:
            continue
        recent_rows = (
            entity_rows.sort_values(by=[resource.time_col], ascending=False, kind="stable")
            .head(neighbor_history_count)
            .sort_values(by=[resource.time_col], kind="stable")
        )
        for rec in recent_rows.to_dict(orient="records"):
            rec.pop("__entity_key", None)
            rec["__example_scope"] = "other"
            records.append(rec)

    records.sort(key=lambda rec: pd.Timestamp(rec[resource.time_col]))
    if len(records) > max_rows:
        records = records[-max_rows:]
    return records


def _mae(preds: list[float], targets: list[float]) -> float:
    if not preds:
        return 0.0
    arr_p = np.asarray(preds, dtype=float)
    arr_t = np.asarray(targets, dtype=float)
    return float(np.mean(np.abs(arr_p - arr_t)))


def _error_stats(preds: list[float], targets: list[float]) -> dict[str, float]:
    if not preds:
        return {
            "abs_error_variance": 0.0,
            "abs_error_std": 0.0,
        }
    arr_p = np.asarray(preds, dtype=float)
    arr_t = np.asarray(targets, dtype=float)
    err = arr_p - arr_t
    abs_err = np.abs(err)
    return {
        "abs_error_variance": float(np.var(abs_err)),
        "abs_error_std": float(np.std(abs_err)),
    }


_NUM_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
_DEBUG_LARGE_GAP_THRESHOLD = 100.0


def _make_strict_numeric_prompt(base_prompt: str) -> str:
    return (
        base_prompt
        + "\n\nSTRICT OUTPUT FORMAT:\n"
        + "- Return exactly one numeric value only.\n"
        + "- No code, no explanation, no markdown, no extra text.\n"
        + "- Keep the value within plausible self historical range when possible.\n"
        + "Answer:"
    )


def _is_debug_breakpoint_prediction(
    pred: float,
    target: float,
    *,
    large_gap_threshold: float,
) -> bool:
    if np.isnan(target):
        return False
    return abs(pred - target) >= float(large_gap_threshold)


def _configure_tmp_dirs(default_tmp_dir: Path) -> None:
    tmp_path = default_tmp_dir.expanduser()
    try:
        tmp_path.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    if not os.access(tmp_path, os.W_OK):
        return

    # Set only when caller hasn't explicitly configured temp dirs.
    for env_name in ("TMPDIR", "TMP", "TEMP", "DUCKDB_TMPDIR"):
        if not os.environ.get(env_name):
            os.environ[env_name] = str(tmp_path)
    # Force Python tempfile module to honor this directory in-process.
    tempfile.tempdir = str(tmp_path)


def run(args: argparse.Namespace) -> None:
    _configure_tmp_dirs(args.tmp_dir)
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

    phase2_task_dir = _phase2_task_dir(args.phase2_artifacts_dir, args.dataset, args.task)
    task_spec = _read_json(phase2_task_dir / "task_spec.json")
    attr_importance_path = phase2_task_dir / "attribute_importance.json"
    attr_importance = _read_json(attr_importance_path) if attr_importance_path.exists() else None

    _inject_phase2_hints_into_registry(
        dataset=args.dataset,
        task=args.task,
        task_spec=task_spec,
        attr_importance=attr_importance,
    )
    policy = _build_phase2_feature_policy(task_spec)

    config = _build_single_task_inference_config(
        dataset=args.dataset,
        task=args.task,
        history_length=args.history_length,
        history_sampling_strategy=args.history_sampling_strategy,
    )
    resource = build_inference_resource(config)

    graph_store = RelBenchGraphRAGStore()
    graph_store.build_base(resource.db, task=resource.task_object)
    extractor = RowFeatureExtractor(resource.db, graph_store)

    fastdfs_engine_path = args.tmp_dir / f"fastdfs_{args.dataset}_{args.task}_{os.getpid()}.db"
    base_dfs_builder = build_fastdfs_context_builder(
        resource,
        engine_path=str(fastdfs_engine_path),
    )
    dfs_builder = Phase2AwareFastDFSBuilder(base_dfs_builder, policy)

    task_obj = get_task(args.dataset, args.task, download=True)
    test_table = task_obj.get_table("test", mask_input_cols=False)
    test_rows = list(test_table.df.itertuples(index=False))
    if args.max_items > 0:
        test_rows = test_rows[: args.max_items]

    max_query_time = max(
        (pd.Timestamp(row._asdict()[resource.time_col]) for row in test_rows),
        default=pd.Timestamp.min,
    )
    pool_df = _build_example_pool(
        resource,
        entity_col=resource.entity_col,
        time_col=resource.time_col,
        output_col=resource.output_col,
        max_query_time=max_query_time,
    )
    static_similarity_index = (
        _build_static_similarity_index(resource) if args.use_static_similar_others else None
    )
    dfs_similarity_cutoff_cache: dict[int, dict[str, Any]] = {}
    train_vector_db_index = (
        _load_train_dfs_vector_db_index(phase2_task_dir=phase2_task_dir, resource=resource)
        if args.other_example_search_mode == "train_vector_db"
        else None
    )
    if args.other_example_search_mode == "train_vector_db" and train_vector_db_index is None:
        raise FileNotFoundError(
            "train_vector_db mode requested, but Phase 2 vector DB artifacts were not found "
            f"under {phase2_task_dir / 'dfs_train_vector_db'}"
        )
    if args.other_example_search_mode == "train_vector_db":
        assert train_vector_db_index is not None

    self_history_min_gap = _resolve_self_history_min_gap(resource.task_object)
    if self_history_min_gap > pd.Timedelta(0):
        print(
            "[phase3] leakage-safe self-history enabled: "
            f"row_time + {self_history_min_gap} <= query_time"
        )

    llm = LocalLLM(MODEL_PATHS[args.model_size], print_log=True)
    inference_start_time = time.perf_counter()

    total_items = len(test_rows)
    predictions: list[float] = []
    targets: list[float] = []
    records: list[dict[str, Any]] = []
    profiling_batches: list[dict[str, Any]] = []
    prof_retrieval_total = 0.0
    prof_dfs_prompt_total = 0.0
    prof_llm_total = 0.0
    profile_enabled = bool(args.profile_phase3)
    profile_max_batches = max(0, int(args.profile_phase3_max_preprocess_batches))
    processed_preprocess_batches = 0
    profile_early_stop = False
    preprocess_batch_size = max(1, int(args.preprocess_batch_size))
    dfs_batch_size = max(1, int(args.dfs_batch_size))
    llm_batch_size = max(1, int(args.llm_batch_size))
    for pre_start in range(0, total_items, preprocess_batch_size):
        pre_batch_start = time.perf_counter()
        pre_end = min(total_items, pre_start + preprocess_batch_size)
        pre_rows = test_rows[pre_start:pre_end]
        retrieval_seconds = 0.0
        dfs_prompt_seconds = 0.0
        llm_seconds = 0.0
        eval_items_count_for_batch = 0
        dfs_similarity_query_cache: dict[tuple[str, int], dict[str, float]] = {}
        if args.other_example_search_mode in {"dfs_similarity", "train_vector_db"}:
            pre_rows_dicts = [row._asdict() for row in pre_rows]
            dfs_similarity_query_cache = _build_query_feature_cache_for_rows(
                rows=pre_rows_dicts,
                resource=resource,
                dfs_builder=dfs_builder,
            )
        if args.enable_cpu_gpu_pipeline:
            prompt_queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, int(args.pipeline_prompt_queue_size)))
            producer_state: dict[str, Any] = {
                "retrieval_seconds": 0.0,
                "dfs_prompt_seconds": 0.0,
                "eval_items": 0,
                "error": None,
            }

            def _producer() -> None:
                eval_buffer: list[EvalItem] = []
                try:
                    for offset, row in enumerate(pre_rows):
                        row_start = time.perf_counter()
                        row_index = pre_start + offset
                        row_dict = row._asdict()
                        q_time = pd.Timestamp(row_dict[resource.time_col])
                        self_history_rows = _sample_history_before_query(
                            pool_df=pool_df,
                            entity_col=resource.entity_col,
                            time_col=resource.time_col,
                            output_col=resource.output_col,
                            entity_value=row_dict[resource.entity_col],
                            query_time=q_time,
                            history_length=args.history_length,
                            min_history_gap=self_history_min_gap,
                        )
                        if args.other_example_search_mode == "dfs_similarity":
                            other_history_rows = _select_other_entity_examples_by_dfs_similarity(
                                resource=resource,
                                example_pool_df=pool_df,
                                query_row=row_dict,
                                entity_value=row_dict[resource.entity_col],
                                cutoff_time=q_time,
                                max_rows=max(0, args.other_example_max_count),
                                neighbor_entity_count=max(0, args.other_neighbor_entity_count),
                                neighbor_history_count=max(0, args.other_neighbor_history_count),
                                dfs_builder=dfs_builder,
                                cutoff_cache=dfs_similarity_cutoff_cache,
                                cutoff_cache_size=max(0, int(args.dfs_similarity_cutoff_cache_size)),
                                query_features_override=dfs_similarity_query_cache.get(
                                    _query_feature_cache_key(
                                        row_dict.get(resource.entity_col),
                                        q_time,
                                    )
                                ),
                            )
                        elif args.other_example_search_mode == "train_vector_db":
                            query_features = dfs_similarity_query_cache.get(
                                _query_feature_cache_key(
                                    row_dict.get(resource.entity_col),
                                    q_time,
                                ),
                                {},
                            )
                            other_history_rows = _select_other_entity_examples_by_train_vector_db(
                                resource=resource,
                                index=train_vector_db_index,
                                query_features=query_features,
                                entity_value=row_dict[resource.entity_col],
                                cutoff_time=q_time,
                                max_rows=max(0, args.other_example_max_count),
                                neighbor_entity_count=max(0, args.other_neighbor_entity_count),
                                neighbor_history_count=max(0, args.other_neighbor_history_count),
                            )
                        else:
                            other_history_rows = _select_other_entity_examples(
                                graph_store=graph_store,
                                resource=resource,
                                example_pool_df=pool_df,
                                entity_value=row_dict[resource.entity_col],
                                cutoff_time=q_time,
                                max_rows=max(0, args.other_example_max_count),
                                neighbor_entity_count=max(0, args.other_neighbor_entity_count),
                                neighbor_history_count=max(0, args.other_neighbor_history_count),
                                neighbor_search_hops=max(1, args.other_neighbor_search_hops),
                                top_k=max(1, args.top_k),
                                static_similarity_index=static_similarity_index,
                                use_static_similar_others=bool(args.use_static_similar_others),
                            )
                        history_rows = self_history_rows + other_history_rows
                        history_rows.sort(key=lambda r: pd.Timestamp(r[resource.time_col]))
                        eval_buffer.append(
                            EvalItem(
                                row_index=row_index,
                                entity_value=row_dict[resource.entity_col],
                                query_time=q_time,
                                query_row=row_dict,
                                history_rows=history_rows,
                            )
                        )
                        producer_state["retrieval_seconds"] += time.perf_counter() - row_start
                        producer_state["eval_items"] += 1

                        if len(eval_buffer) >= dfs_batch_size:
                            prep_start = time.perf_counter()
                            prompts, outputs_meta = _prepare_prompt_pack_for_eval_items(
                                batch_items=eval_buffer,
                                dfs_builder=dfs_builder,
                                graph_store=graph_store,
                                extractor=extractor,
                                resource=resource,
                                args=args,
                            )
                            producer_state["dfs_prompt_seconds"] += time.perf_counter() - prep_start
                            prompt_queue.put((prompts, outputs_meta))
                            eval_buffer = []

                    if eval_buffer:
                        prep_start = time.perf_counter()
                        prompts, outputs_meta = _prepare_prompt_pack_for_eval_items(
                            batch_items=eval_buffer,
                            dfs_builder=dfs_builder,
                            graph_store=graph_store,
                            extractor=extractor,
                            resource=resource,
                            args=args,
                        )
                        producer_state["dfs_prompt_seconds"] += time.perf_counter() - prep_start
                        prompt_queue.put((prompts, outputs_meta))
                except Exception as exc:
                    producer_state["error"] = exc
                finally:
                    prompt_queue.put(None)

            producer_thread = threading.Thread(target=_producer, daemon=True)
            producer_thread.start()

            llm_start = time.perf_counter()
            while True:
                payload = prompt_queue.get()
                if payload is None:
                    break
                prompts, outputs_meta = payload
                _run_llm_for_prompt_pack(
                    llm=llm,
                    prompts=prompts,
                    outputs_meta=outputs_meta,
                    llm_batch_size=llm_batch_size,
                    max_new_tokens=int(args.max_new_tokens),
                    print_predictions=bool(args.print_predictions),
                    breakpoint_on_nonpositive=bool(args.breakpoint_on_nonpositive),
                    breakpoint_large_gap_threshold=float(args.breakpoint_large_gap_threshold),
                    predictions=predictions,
                    targets=targets,
                    records=records,
                )
            llm_seconds = time.perf_counter() - llm_start
            producer_thread.join()
            if producer_state.get("error") is not None:
                raise RuntimeError("Pipeline producer failed") from producer_state["error"]
            retrieval_seconds = float(producer_state["retrieval_seconds"])
            dfs_prompt_seconds = float(producer_state["dfs_prompt_seconds"])
            eval_items_count_for_batch = int(producer_state["eval_items"])
        else:
            eval_items: list[EvalItem] = []
            retrieval_start = time.perf_counter()
            for offset, row in enumerate(pre_rows):
                row_index = pre_start + offset
                row_dict = row._asdict()
                q_time = pd.Timestamp(row_dict[resource.time_col])
                self_history_rows = _sample_history_before_query(
                    pool_df=pool_df,
                    entity_col=resource.entity_col,
                    time_col=resource.time_col,
                    output_col=resource.output_col,
                    entity_value=row_dict[resource.entity_col],
                    query_time=q_time,
                    history_length=args.history_length,
                    min_history_gap=self_history_min_gap,
                )
                if args.other_example_search_mode == "dfs_similarity":
                    other_history_rows = _select_other_entity_examples_by_dfs_similarity(
                        resource=resource,
                        example_pool_df=pool_df,
                        query_row=row_dict,
                        entity_value=row_dict[resource.entity_col],
                        cutoff_time=q_time,
                        max_rows=max(0, args.other_example_max_count),
                        neighbor_entity_count=max(0, args.other_neighbor_entity_count),
                        neighbor_history_count=max(0, args.other_neighbor_history_count),
                        dfs_builder=dfs_builder,
                        cutoff_cache=dfs_similarity_cutoff_cache,
                        cutoff_cache_size=max(0, int(args.dfs_similarity_cutoff_cache_size)),
                        query_features_override=dfs_similarity_query_cache.get(
                            _query_feature_cache_key(
                                row_dict.get(resource.entity_col),
                                q_time,
                            )
                        ),
                    )
                elif args.other_example_search_mode == "train_vector_db":
                    query_features = dfs_similarity_query_cache.get(
                        _query_feature_cache_key(
                            row_dict.get(resource.entity_col),
                            q_time,
                        ),
                        {},
                    )
                    other_history_rows = _select_other_entity_examples_by_train_vector_db(
                        resource=resource,
                        index=train_vector_db_index,
                        query_features=query_features,
                        entity_value=row_dict[resource.entity_col],
                        cutoff_time=q_time,
                        max_rows=max(0, args.other_example_max_count),
                        neighbor_entity_count=max(0, args.other_neighbor_entity_count),
                        neighbor_history_count=max(0, args.other_neighbor_history_count),
                    )
                else:
                    other_history_rows = _select_other_entity_examples(
                        graph_store=graph_store,
                        resource=resource,
                        example_pool_df=pool_df,
                        entity_value=row_dict[resource.entity_col],
                        cutoff_time=q_time,
                        max_rows=max(0, args.other_example_max_count),
                        neighbor_entity_count=max(0, args.other_neighbor_entity_count),
                        neighbor_history_count=max(0, args.other_neighbor_history_count),
                        neighbor_search_hops=max(1, args.other_neighbor_search_hops),
                        top_k=max(1, args.top_k),
                        static_similarity_index=static_similarity_index,
                        use_static_similar_others=bool(args.use_static_similar_others),
                    )
                history_rows = self_history_rows + other_history_rows
                history_rows.sort(key=lambda r: pd.Timestamp(r[resource.time_col]))
                eval_items.append(
                    EvalItem(
                        row_index=row_index,
                        entity_value=row_dict[resource.entity_col],
                        query_time=q_time,
                        query_row=row_dict,
                        history_rows=history_rows,
                    )
                )
            retrieval_seconds = time.perf_counter() - retrieval_start
            eval_items_count_for_batch = len(eval_items)

            prompt_packs: list[tuple[list[str], list[dict[str, Any]]]] = []
            dfs_prompt_start = time.perf_counter()
            for start in range(0, len(eval_items), dfs_batch_size):
                batch_items = eval_items[start : start + dfs_batch_size]
                if not batch_items:
                    continue
                prompt_packs.append(
                    _prepare_prompt_pack_for_eval_items(
                        batch_items=batch_items,
                        dfs_builder=dfs_builder,
                        graph_store=graph_store,
                        extractor=extractor,
                        resource=resource,
                        args=args,
                    )
                )
            dfs_prompt_seconds = time.perf_counter() - dfs_prompt_start

            llm_start = time.perf_counter()
            for prompts, outputs_meta in prompt_packs:
                _run_llm_for_prompt_pack(
                    llm=llm,
                    prompts=prompts,
                    outputs_meta=outputs_meta,
                    llm_batch_size=llm_batch_size,
                    max_new_tokens=int(args.max_new_tokens),
                    print_predictions=bool(args.print_predictions),
                    breakpoint_on_nonpositive=bool(args.breakpoint_on_nonpositive),
                    breakpoint_large_gap_threshold=float(args.breakpoint_large_gap_threshold),
                    predictions=predictions,
                    targets=targets,
                    records=records,
                )
            llm_seconds = time.perf_counter() - llm_start

        prof_retrieval_total += retrieval_seconds
        prof_dfs_prompt_total += dfs_prompt_seconds
        prof_llm_total += llm_seconds
        processed_preprocess_batches += 1
        batch_total_seconds = time.perf_counter() - pre_batch_start
        if profile_enabled:
            profiling_batches.append(
                {
                    "batch_index": int(processed_preprocess_batches - 1),
                    "rows_start": int(pre_start),
                    "rows_end_exclusive": int(pre_end),
                    "rows_in_batch": int(len(pre_rows)),
                    "eval_items": int(eval_items_count_for_batch),
                    "retrieval_seconds": float(retrieval_seconds),
                    "dfs_prompt_seconds": float(dfs_prompt_seconds),
                    "llm_seconds": float(llm_seconds),
                    "batch_total_seconds": float(batch_total_seconds),
                }
            )

        print(
            f"[phase3] processed rows {pre_end}/{total_items} "
            f"(preprocess_batch_size={preprocess_batch_size}, "
            f"dfs_batch_size={dfs_batch_size}, llm_batch_size={llm_batch_size})"
        )
        if profile_enabled and profile_max_batches > 0 and processed_preprocess_batches >= profile_max_batches:
            profile_early_stop = True
            print(
                f"[phase3][profile] early stop after {processed_preprocess_batches} "
                f"preprocess batches."
            )
            break

    mae = _mae(predictions, targets)
    err_stats = _error_stats(predictions, targets)
    inference_seconds = time.perf_counter() - inference_start_time

    out_dir = args.output_dir / args.dataset / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    run_summary = {
        "dataset": args.dataset,
        "task": args.task,
        "split": "test",
        "mode": "dfs_table_only",
        "phase2_task_spec": str((phase2_task_dir / "task_spec.json").resolve()),
        "items": total_items,
        "scored_items": len(predictions),
        "mae": mae,
        "abs_error_variance": err_stats["abs_error_variance"],
        "abs_error_std": err_stats["abs_error_std"],
        "history_length": int(args.history_length),
        "self_history_min_gap": str(self_history_min_gap),
        "self_history_leakage_safe": bool(self_history_min_gap > pd.Timedelta(0)),
        "model_size": args.model_size,
        "other_example_search_mode": args.other_example_search_mode,
        "enable_cpu_gpu_pipeline": bool(args.enable_cpu_gpu_pipeline),
        "inference_seconds": float(inference_seconds),
        "profile_phase3": bool(profile_enabled),
        "profile_phase3_max_preprocess_batches": int(profile_max_batches),
        "processed_preprocess_batches": int(processed_preprocess_batches),
        "profile_early_stop": bool(profile_early_stop),
        "timing_breakdown": {
            "retrieval_seconds": float(prof_retrieval_total),
            "dfs_prompt_seconds": float(prof_dfs_prompt_total),
            "llm_seconds": float(prof_llm_total),
        },
    }
    if profile_enabled:
        run_summary["profiling_batches"] = profiling_batches

    with (out_dir / "phase3_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=True)
    with (out_dir / "phase3_predictions.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=True)

    print("Phase 3 completed.")
    print(f"- dataset={args.dataset}")
    print(f"- task={args.task}")
    print("- split=test")
    print(f"- items={total_items}")
    print(f"- scored_items={len(predictions)}")
    print(f"- mae={mae:.6f} | inference_seconds={inference_seconds:.2f}")
    print(f"- abs_error_variance={err_stats['abs_error_variance']:.6f}")
    print(f"- abs_error_std={err_stats['abs_error_std']:.6f}")
    print(f"- output_dir={out_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 3 DFS-table-only inference pipeline")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--phase2-artifacts-dir", type=Path, default=Path("RFM/v2/phase2/artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("RFM/v2/phase3/artifacts"))
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=Path("/work/pi_mserafini_umass_edu/ashraful/tmp"),
        help="Default temp directory for fastdfs/duckdb scratch files.",
    )
    parser.add_argument("--model-size", type=str, default="8b", choices=sorted(MODEL_PATHS.keys()))
    parser.add_argument("--llm-batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--history-length", type=int, default=10)
    parser.add_argument("--history-sampling-strategy", type=str, default="most_recent_k")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--num-hops", type=int, default=2)
    parser.add_argument("--preprocess-batch-size", type=int, default=256)
    parser.add_argument("--dfs-batch-size", type=int, default=64)
    parser.add_argument(
        "--enable-cpu-gpu-pipeline",
        action="store_true",
        help=(
            "Overlap CPU retrieval/DFS prompt preparation with GPU LLM generation "
            "using a producer-consumer prompt queue."
        ),
    )
    parser.add_argument(
        "--pipeline-prompt-queue-size",
        type=int,
        default=2,
        help="Prompt-pack queue depth used when --enable-cpu-gpu-pipeline is enabled.",
    )
    parser.add_argument("--other-example-max-count", type=int, default=9)
    parser.add_argument("--other-neighbor-entity-count", type=int, default=10)
    parser.add_argument("--other-neighbor-history-count", type=int, default=3)
    parser.add_argument("--other-neighbor-search-hops", type=int, default=2)
    parser.add_argument(
        "--dfs-similarity-cutoff-cache-size",
        type=int,
        default=8,
        help=(
            "Number of cutoff timestamps to cache for dfs_similarity candidate "
            "features. Set 0 to disable cache."
        ),
    )
    parser.add_argument(
        "--other-example-search-mode",
        type=str,
        default="dfs_similarity",
        choices=["graph_static", "dfs_similarity", "train_vector_db"],
        help=(
            "How to select 'other examples': "
            "'graph_static' (existing graph/static strategy) or "
            "'dfs_similarity' (query DFS feature similarity) or "
            "'train_vector_db' (nearest neighbors over Phase 2 train DFS vector DB)."
        ),
    )
    parser.add_argument(
        "--use-static-similar-others",
        dest="use_static_similar_others",
        action="store_true",
        help="Use static-feature similarity over anchor ids to augment other examples.",
    )
    parser.add_argument(
        "--no-use-static-similar-others",
        dest="use_static_similar_others",
        action="store_false",
        help="Disable static-feature-similar anchor retrieval for other examples.",
    )
    parser.set_defaults(use_static_similar_others=True)
    parser.add_argument("--max-items", type=int, default=0, help="0 means all test rows")
    parser.add_argument(
        "--print-predictions",
        action="store_true",
        help="Print per-item prediction and ground truth during inference.",
    )
    parser.add_argument(
        "--breakpoint-on-nonpositive",
        action="store_true",
        help=(
            "If set, print full prompt and enter pdb when "
            "abs(pred-ground_truth) exceeds --breakpoint-large-gap-threshold."
        ),
    )
    parser.add_argument(
        "--breakpoint-large-gap-threshold",
        type=float,
        default=_DEBUG_LARGE_GAP_THRESHOLD,
        help="Absolute-error threshold used by --breakpoint-on-nonpositive (default: 10).",
    )
    parser.add_argument(
        "--profile-phase3",
        action="store_true",
        help="Enable phase3 timing profiling and include per-batch timing in summary JSON.",
    )
    parser.add_argument(
        "--profile-phase3-max-preprocess-batches",
        type=int,
        default=1,
        help=(
            "When --profile-phase3 is enabled, stop after this many preprocess batches. "
            "Set 0 to disable early stop."
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.history_length <= 0:
        raise ValueError("--history-length must be > 0")
    if args.preprocess_batch_size <= 0:
        raise ValueError("--preprocess-batch-size must be > 0")
    if args.dfs_batch_size <= 0:
        raise ValueError("--dfs-batch-size must be > 0")
    if args.llm_batch_size <= 0:
        raise ValueError("--llm-batch-size must be > 0")
    if args.pipeline_prompt_queue_size <= 0:
        raise ValueError("--pipeline-prompt-queue-size must be > 0")
    if args.profile_phase3_max_preprocess_batches < 0:
        raise ValueError("--profile-phase3-max-preprocess-batches must be >= 0")
    run(args)


if __name__ == "__main__":
    main()
