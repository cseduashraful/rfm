from __future__ import annotations

from collections import Counter
from typing import Any


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
    low = str(confidence).lower()
    if low == "high":
        return 1.0
    if low == "medium":
        return 0.75
    if low == "low":
        return 0.45
    return 0.60


def build_semantic_context_graph(
    *,
    schema_artifact: dict[str, Any],
    stats_artifact: dict[str, Any],
    semantics_artifact: dict[str, Any],
    fk_edges: list[dict[str, Any]],
) -> dict[str, Any]:
    role_by_table: dict[str, str] = {}
    cols_by_table: dict[str, dict[str, dict[str, Any]]] = {}
    for table in semantics_artifact.get("tables", []):
        table_name = str(table.get("table_name", ""))
        role_by_table[table_name] = str(table.get("table_role", "entity"))
        column_meta: dict[str, dict[str, Any]] = {}
        for column in table.get("columns", []):
            column_name = str(column.get("name", ""))
            column_meta[column_name] = {
                "semantic_role": str(column.get("semantic_role", "other")),
                "leakage_risk": bool(column.get("leakage_risk", False)),
                "confidence": str(column.get("confidence", "heuristic")),
            }
        cols_by_table[table_name] = column_meta

    stats_cols: dict[str, dict[str, dict[str, Any]]] = {}
    row_counts: dict[str, int] = {}
    for table in stats_artifact.get("tables", []):
        table_name = str(table.get("table_name", ""))
        row_counts[table_name] = int(table.get("row_count", 0))
        table_columns = table.get("columns", {}) or {}
        stats_cols[table_name] = {
            str(column_name): {
                "missingness": meta.get("missingness"),
                "numeric": bool(meta.get("numeric")),
                "distinct_count": meta.get("distinct_count"),
            }
            for column_name, meta in table_columns.items()
            if isinstance(meta, dict)
        }

    table_nodes: list[dict[str, Any]] = []
    for table in schema_artifact.get("tables", []):
        table_name = str(table.get("table_name", ""))
        semantic_profile = Counter()
        trusted_measure_columns: list[str] = []
        for column in table.get("columns", []):
            column_name = str(column.get("name", ""))
            role_meta = cols_by_table.get(table_name, {}).get(column_name, {})
            semantic_role = str(role_meta.get("semantic_role", "other"))
            leakage_risk = bool(role_meta.get("leakage_risk", False))
            stats_meta = stats_cols.get(table_name, {}).get(column_name, {})
            missingness = stats_meta.get("missingness")
            numeric = bool(stats_meta.get("numeric", False))
            semantic_profile[semantic_role] += 1
            if (
                semantic_role == "measure"
                and numeric
                and (missingness is None or float(missingness) <= 0.25)
                and not leakage_risk
            ):
                trusted_measure_columns.append(column_name)
        table_nodes.append(
            {
                "table_name": table_name,
                "table_role": role_by_table.get(table_name, "entity"),
                "row_count": int(row_counts.get(table_name, table.get("row_count", 0))),
                "time_column": table.get("time_column"),
                "provided_primary_key": table.get("provided_primary_key"),
                "semantic_profile": dict(semantic_profile),
                "trusted_measure_columns": trusted_measure_columns[:20],
            }
        )

    edges: list[dict[str, Any]] = []
    for fk in fk_edges:
        child = str(fk.get("child_table", ""))
        parent = str(fk.get("parent_table", ""))
        child_col = str(fk.get("child_column", ""))
        parent_col = str(fk.get("parent_column", ""))
        if not child or not parent or not child_col or not parent_col:
            continue
        conf = str(fk.get("confidence", "medium"))
        overlap_ratio = float(fk.get("overlap_ratio", 0.0))
        child_unique_ratio = float(fk.get("child_unique_ratio", 0.0))
        null_ratio = float(fk.get("null_ratio", 0.0))
        src_role = role_by_table.get(child, "entity")
        dst_role = role_by_table.get(parent, "entity")
        role_aff = role_affinity(src_role, dst_role)
        temporal_alignment = _temporal_alignment(schema_artifact, child, parent)
        cardinality_risk = max(0.0, 1.0 - min(child_unique_ratio, 1.0))
        conf_score = confidence_to_score(conf)
        base_weight = (
            0.42 * conf_score
            + 0.23 * overlap_ratio
            + 0.20 * role_aff
            + 0.15 * temporal_alignment
            - 0.10 * cardinality_risk
            - 0.05 * null_ratio
        )
        base_weight = float(max(0.0, min(1.0, base_weight)))
        recommended = bool(base_weight >= 0.55 and conf != "low")
        edge_base = {
            "src_column": child_col,
            "dst_column": parent_col,
            "source": str(fk.get("source", "inferred")),
            "confidence": conf,
            "overlap_ratio": overlap_ratio,
            "semantic_role_affinity": role_aff,
            "temporal_alignment": temporal_alignment,
            "cardinality_risk": cardinality_risk,
            "null_ratio": null_ratio,
            "edge_weight": base_weight,
            "recommended_by_default": recommended,
        }
        edges.append(
            {
                "edge_id": f"{child}.{child_col}->{parent}.{parent_col}",
                "src_table": child,
                "dst_table": parent,
                "direction": "child_to_parent",
                **edge_base,
            }
        )
        edges.append(
            {
                "edge_id": f"{parent}.{parent_col}->{child}.{child_col}",
                "src_table": parent,
                "dst_table": child,
                "src_column": parent_col,
                "dst_column": child_col,
                "direction": "parent_to_child",
                **{
                    **edge_base,
                    "edge_weight": round(base_weight * 0.92, 6),
                    "recommended_by_default": bool(recommended and base_weight >= 0.65),
                },
            }
        )

    degree_map: dict[str, dict[str, int]] = {
        node["table_name"]: {"in_degree": 0, "out_degree": 0}
        for node in table_nodes
    }
    for edge in edges:
        src = str(edge["src_table"])
        dst = str(edge["dst_table"])
        degree_map.setdefault(src, {"in_degree": 0, "out_degree": 0})
        degree_map.setdefault(dst, {"in_degree": 0, "out_degree": 0})
        degree_map[src]["out_degree"] += 1
        degree_map[dst]["in_degree"] += 1

    for node in table_nodes:
        table_name = str(node["table_name"])
        node["in_degree"] = int(degree_map.get(table_name, {}).get("in_degree", 0))
        node["out_degree"] = int(degree_map.get(table_name, {}).get("out_degree", 0))
        node["hub_score"] = float(node["in_degree"] + node["out_degree"])

    edges.sort(
        key=lambda row: (
            float(row.get("edge_weight", 0.0)),
            bool(row.get("recommended_by_default", False)),
        ),
        reverse=True,
    )
    table_nodes.sort(
        key=lambda row: (float(row.get("hub_score", 0.0)), int(row.get("row_count", 0))),
        reverse=True,
    )
    return {
        "graph_version": "semantic_context_graph.v2",
        "node_count": len(table_nodes),
        "edge_count": len(edges),
        "table_nodes": table_nodes,
        "edges": edges,
    }


def _temporal_alignment(
    schema_artifact: dict[str, Any],
    child_table: str,
    parent_table: str,
) -> float:
    time_by_table = {
        str(table.get("table_name", "")): table.get("time_column")
        for table in schema_artifact.get("tables", [])
    }
    child_time = time_by_table.get(child_table)
    parent_time = time_by_table.get(parent_table)
    if child_time and parent_time:
        return 1.0
    if child_time or parent_time:
        return 0.8
    return 0.9
