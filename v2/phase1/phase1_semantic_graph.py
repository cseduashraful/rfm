from __future__ import annotations

from collections import Counter
from typing import Any, Callable

import pandas as pd


TableRoleFn = Callable[[str, Any], str]
InferFkFn = Callable[[Any, dict[str, pd.DataFrame]], list[dict[str, Any]]]


def role_affinity(src_role: str, dst_role: str) -> float:
    pair = (src_role, dst_role)
    if pair in {("entity", "event"), ("event", "entity")}:
        return 1.0
    if pair in {("event", "event"), ("entity", "entity")}:
        return 0.85
    if "lookup" in pair:
        return 0.65
    if "bridge" in pair:
        return 0.75
    if "static" in pair:
        return 0.55
    return 0.7


def confidence_to_score(confidence: str) -> float:
    c = str(confidence).lower()
    if c == "high":
        return 1.0
    if c == "medium":
        return 0.75
    if c == "low":
        return 0.45
    return 0.6


def compute_fk_overlap_ratio(
    *,
    filtered: dict[str, pd.DataFrame],
    child: str,
    child_col: str,
    parent: str,
    parent_col: str,
) -> float:
    child_df = filtered.get(child, pd.DataFrame())
    parent_df = filtered.get(parent, pd.DataFrame())
    if child_df.empty or parent_df.empty:
        return 0.0
    if child_col not in child_df.columns or parent_col not in parent_df.columns:
        return 0.0
    cvals = set(child_df[child_col].dropna().unique().tolist())
    if not cvals:
        return 0.0
    pvals = set(parent_df[parent_col].dropna().unique().tolist())
    overlap = len(cvals & pvals)
    return float(overlap / max(1, len(cvals)))


def build_semantic_context_graph(
    *,
    db: Any,
    filtered: dict[str, pd.DataFrame],
    schema_artifact: dict[str, Any],
    stats_artifact: dict[str, Any],
    semantics_artifact: dict[str, Any],
    infer_fk_candidates_fn: InferFkFn,
    table_role_fn: TableRoleFn,
) -> dict[str, Any]:
    role_by_table: dict[str, str] = {}
    cols_by_table: dict[str, dict[str, dict[str, Any]]] = {}
    for t in semantics_artifact.get("tables", []):
        tname = str(t.get("table_name", ""))
        role_by_table[tname] = str(t.get("table_role", "entity"))
        cmeta: dict[str, dict[str, Any]] = {}
        for c in t.get("columns", []):
            cname = str(c.get("name", ""))
            cmeta[cname] = {
                "semantic_role": str(c.get("semantic_role", "other")),
                "leakage_risk": bool(c.get("leakage_risk", False)),
                "confidence": str(c.get("confidence", "heuristic")),
            }
        cols_by_table[tname] = cmeta

    stats_cols: dict[str, dict[str, dict[str, Any]]] = {}
    for t in stats_artifact.get("tables", []):
        tname = str(t.get("table_name", ""))
        cmeta: dict[str, dict[str, Any]] = {}
        for cname, meta in (t.get("columns", {}) or {}).items():
            if not isinstance(meta, dict):
                continue
            num_meta = meta.get("numeric", {})
            cmeta[str(cname)] = {
                "missingness": meta.get("missingness"),
                "numeric": isinstance(num_meta, dict) and len(num_meta) > 0,
            }
        stats_cols[tname] = cmeta

    for t in schema_artifact.get("tables", []):
        tname = str(t.get("table_name", ""))
        if tname not in role_by_table and tname in db.table_dict:
            role_by_table[tname] = table_role_fn(tname, db.table_dict[tname])

    table_nodes: list[dict[str, Any]] = []
    for t in schema_artifact.get("tables", []):
        tname = str(t.get("table_name", ""))
        row_count = int(t.get("row_count", 0))
        cols = t.get("columns", [])
        role = role_by_table.get(tname, table_role_fn(tname, db.table_dict[tname]))

        semantic_profile = Counter()
        trusted_cols: list[str] = []
        for c in cols:
            cname = str(c.get("name", ""))
            role_meta = cols_by_table.get(tname, {}).get(cname, {})
            srole = str(role_meta.get("semantic_role", "other"))
            semantic_profile[srole] += 1
            leak = bool(role_meta.get("leakage_risk", False))
            sm = stats_cols.get(tname, {}).get(cname, {})
            missing = sm.get("missingness")
            numeric = bool(sm.get("numeric", False))
            if (not leak) and (missing is None or float(missing) <= 0.25) and numeric:
                trusted_cols.append(cname)

        table_nodes.append(
            {
                "table_name": tname,
                "table_role": role,
                "row_count": row_count,
                "time_column": t.get("time_column"),
                "provided_primary_key": t.get("provided_primary_key"),
                "semantic_profile": dict(semantic_profile),
                "trusted_measure_columns": trusted_cols[:20],
            }
        )

    schema_fk_edges: list[dict[str, Any]] = []
    for t in schema_artifact.get("tables", []):
        child = str(t.get("table_name", ""))
        for ccol, parent in (t.get("provided_foreign_keys", {}) or {}).items():
            parent_table = str(parent)
            if not parent_table:
                continue
            schema_fk_edges.append(
                {
                    "child_table": child,
                    "child_column": str(ccol),
                    "parent_table": parent_table,
                    "parent_column": db.table_dict[parent_table].pkey_col,
                    "source": "provided",
                    "confidence": "high",
                    "recommended_by_default": True,
                }
            )

    inferred = infer_fk_candidates_fn(db, filtered)
    existing = {
        (e["child_table"], e["child_column"], e["parent_table"], e["parent_column"])
        for e in schema_fk_edges
    }
    for e in inferred:
        key = (e["child_table"], e["child_column"], e["parent_table"], e["parent_column"])
        if key not in existing:
            schema_fk_edges.append(e)

    edges: list[dict[str, Any]] = []
    for e in schema_fk_edges:
        child = str(e["child_table"])
        parent = str(e["parent_table"])
        ccol = str(e["child_column"])
        pcol = str(e.get("parent_column") or "")
        conf = str(e.get("confidence", "medium"))
        overlap_ratio = float(e.get("overlap_ratio", 0.0))
        if overlap_ratio <= 0.0:
            overlap_ratio = compute_fk_overlap_ratio(
                filtered=filtered,
                child=child,
                child_col=ccol,
                parent=parent,
                parent_col=pcol,
            )

        src_role = role_by_table.get(child, "entity")
        dst_role = role_by_table.get(parent, "entity")
        role_aff = role_affinity(src_role, dst_role)
        conf_score = confidence_to_score(conf)

        child_unique_ratio = 0.0
        child_df = filtered.get(child, pd.DataFrame())
        if (not child_df.empty) and ccol in child_df.columns:
            nn = int(child_df[ccol].notna().sum())
            if nn > 0:
                child_unique_ratio = float(child_df[ccol].nunique(dropna=True) / nn)

        cardinality_risk = max(0.0, 1.0 - min(child_unique_ratio, 1.0))
        temporal_alignment = 1.0
        ctime = db.table_dict[child].time_col
        ptime = db.table_dict[parent].time_col
        if (ctime is None) ^ (ptime is None):
            temporal_alignment = 0.8

        base_weight = (
            0.40 * conf_score
            + 0.25 * float(overlap_ratio)
            + 0.20 * role_aff
            + 0.15 * temporal_alignment
            - 0.20 * cardinality_risk
        )
        base_weight = float(max(0.0, min(1.0, base_weight)))
        recommended = bool(base_weight >= 0.55 and conf != "low")

        fwd = {
            "edge_id": f"{child}.{ccol}->{parent}.{pcol}",
            "src_table": child,
            "dst_table": parent,
            "src_column": ccol,
            "dst_column": pcol,
            "direction": "child_to_parent",
            "source": str(e.get("source", "provided")),
            "confidence": conf,
            "overlap_ratio": float(overlap_ratio),
            "semantic_role_affinity": role_aff,
            "temporal_alignment": temporal_alignment,
            "cardinality_risk": cardinality_risk,
            "edge_weight": base_weight,
            "recommended_by_default": recommended,
        }
        rev = {
            **fwd,
            "edge_id": f"{parent}.{pcol}->{child}.{ccol}",
            "src_table": parent,
            "dst_table": child,
            "src_column": pcol,
            "dst_column": ccol,
            "direction": "parent_to_child",
            "edge_weight": round(base_weight * 0.92, 6),
            "recommended_by_default": bool(recommended and base_weight >= 0.65),
        }
        edges.append(fwd)
        edges.append(rev)

    degree_map: dict[str, dict[str, int]] = {}
    for t in role_by_table:
        degree_map[t] = {"in_degree": 0, "out_degree": 0}
    for e in edges:
        s = str(e["src_table"])
        d = str(e["dst_table"])
        degree_map.setdefault(s, {"in_degree": 0, "out_degree": 0})
        degree_map.setdefault(d, {"in_degree": 0, "out_degree": 0})
        degree_map[s]["out_degree"] += 1
        degree_map[d]["in_degree"] += 1

    for n in table_nodes:
        tname = str(n["table_name"])
        n["in_degree"] = int(degree_map.get(tname, {}).get("in_degree", 0))
        n["out_degree"] = int(degree_map.get(tname, {}).get("out_degree", 0))
        n["hub_score"] = float(n["in_degree"] + n["out_degree"])

    edges.sort(
        key=lambda x: (
            float(x.get("edge_weight", 0.0)),
            bool(x.get("recommended_by_default", False)),
        ),
        reverse=True,
    )
    table_nodes.sort(
        key=lambda x: (float(x.get("hub_score", 0.0)), x.get("row_count", 0)),
        reverse=True,
    )

    return {
        "graph_version": "semantic_context_graph.v1",
        "node_count": len(table_nodes),
        "edge_count": len(edges),
        "table_nodes": table_nodes,
        "edges": edges,
    }
