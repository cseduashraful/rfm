from __future__ import annotations

from collections import Counter
from typing import Any


def role_affinity(src_role: str, dst_role: str) -> float:
    pair = (src_role, dst_role)
    if pair in {("entity", "event"), ("event", "entity")}:
        return 1.0
    if pair in {("event", "event"), ("entity", "entity")}:
        return 0.88
    if "lookup" in pair:
        return 0.70
    if "bridge" in pair:
        return 0.78
    if "static" in pair:
        return 0.58
    return 0.72


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
    semantics_by_table: dict[str, dict[str, Any]] = {}
    columns_by_table: dict[str, dict[str, dict[str, Any]]] = {}
    for table in semantics_artifact.get("tables", []):
        table_name = str(table.get("table_name", ""))
        semantics_by_table[table_name] = table
        column_meta: dict[str, dict[str, Any]] = {}
        for column in table.get("columns", []):
            column_name = str(column.get("name", ""))
            if not column_name:
                continue
            column_meta[column_name] = {
                "semantic_role": str(column.get("semantic_role", "other")),
                "leakage_risk": bool(column.get("leakage_risk", False)),
                "confidence": str(column.get("confidence", "heuristic")),
                "predictive_value": str(column.get("predictive_value", "low")),
                "aggregation_hints": list(column.get("aggregation_hints", []) or []),
            }
        columns_by_table[table_name] = column_meta

    stats_cols: dict[str, dict[str, dict[str, Any]]] = {}
    row_counts: dict[str, int] = {}
    for table in stats_artifact.get("tables", []):
        table_name = str(table.get("table_name", ""))
        row_counts[table_name] = int(table.get("row_count", 0))
        table_columns = table.get("columns", {}) or {}
        stats_cols[table_name] = {}
        if isinstance(table_columns, dict):
            items = table_columns.items()
        else:
            items = [(c.get("name"), c) for c in table_columns if isinstance(c, dict)]
        for column_name, meta in items:
            if not column_name or not isinstance(meta, dict):
                continue
            stats_cols[table_name][str(column_name)] = {
                "missingness": meta.get("missingness"),
                "numeric": bool(meta.get("numeric")) or isinstance(meta.get("numeric"), dict),
                "distinct_count": meta.get("distinct_count"),
            }

    incoming_counts: dict[str, int] = {}
    for fk in fk_edges:
        parent = str(fk.get("parent_table", ""))
        if parent:
            incoming_counts[parent] = incoming_counts.get(parent, 0) + 1

    table_nodes: list[dict[str, Any]] = []
    for table in schema_artifact.get("tables", []):
        table_name = str(table.get("table_name", ""))
        sem_table = semantics_by_table.get(table_name, {})
        sem_columns = columns_by_table.get(table_name, {})
        semantic_profile = Counter()
        trusted_measure_columns: list[str] = []
        predictive_columns: list[str] = []
        for column_name, role_meta in sem_columns.items():
            semantic_role = str(role_meta.get("semantic_role", "other"))
            leakage_risk = bool(role_meta.get("leakage_risk", False))
            predictive_value = str(role_meta.get("predictive_value", "low"))
            stats_meta = stats_cols.get(table_name, {}).get(column_name, {})
            missingness = stats_meta.get("missingness")
            numeric = bool(stats_meta.get("numeric", False))
            semantic_profile[semantic_role] += 1
            if (
                semantic_role == "measure"
                and numeric
                and not leakage_risk
                and (missingness is None or float(missingness) <= 0.25)
            ):
                trusted_measure_columns.append(column_name)
            if predictive_value in {"high", "medium"} and not leakage_risk:
                predictive_columns.append(column_name)

        table_nodes.append(
            {
                "table_name": table_name,
                "table_role": str(sem_table.get("table_role", "entity")),
                "row_count": int(row_counts.get(table_name, table.get("row_count", 0))),
                "time_column": table.get("time_column"),
                "filter_time_column": sem_table.get("filter_time_column", table.get("time_column")),
                "feature_time_columns": list(sem_table.get("feature_time_columns", []) or []),
                "temporal_kind": str(sem_table.get("temporal_kind", "none")),
                "usable_for_recency": bool(sem_table.get("usable_for_recency", False)),
                "row_grain": sem_table.get("row_grain"),
                "table_summary": sem_table.get("table_summary"),
                "primary_subject_entities": list(sem_table.get("primary_subject_entities", []) or []),
                "signal_families": list(sem_table.get("signal_families", []) or []),
                "anchor_strength": float(sem_table.get("anchor_strength", 0.5) or 0.5),
                "history_value": float(sem_table.get("history_value", 0.0) or 0.0),
                "provided_primary_key": table.get("provided_primary_key"),
                "important_columns": list(sem_table.get("important_columns", []) or [])[:12],
                "trusted_measure_columns": trusted_measure_columns[:20],
                "predictive_columns": predictive_columns[:20],
                "semantic_profile": dict(semantic_profile),
                "incoming_fk_count": int(incoming_counts.get(table_name, 0)),
            }
        )

    node_by_table = {str(node["table_name"]): node for node in table_nodes}
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
        child_node = node_by_table.get(child, {})
        parent_node = node_by_table.get(parent, {})
        src_role = str(child_node.get("table_role", "entity"))
        dst_role = str(parent_node.get("table_role", "entity"))
        role_aff = role_affinity(src_role, dst_role)
        temporal_alignment = _temporal_alignment(child_node, parent_node)
        fanout_risk = max(0.0, 1.0 - min(child_unique_ratio, 1.0))
        conf_score = confidence_to_score(conf)
        signal_overlap = _signal_overlap(child_node, parent_node)
        base_weight = (
            0.36 * conf_score
            + 0.22 * overlap_ratio
            + 0.16 * role_aff
            + 0.12 * temporal_alignment
            + 0.08 * signal_overlap
            - 0.10 * fanout_risk
            - 0.04 * null_ratio
        )
        base_weight = float(max(0.0, min(1.0, base_weight)))
        recommended = bool(base_weight >= 0.56 and conf != "low")
        edge_common = {
            "src_column": child_col,
            "dst_column": parent_col,
            "source": str(fk.get("source", "inferred")),
            "confidence": conf,
            "overlap_ratio": overlap_ratio,
            "semantic_role_affinity": role_aff,
            "temporal_alignment": temporal_alignment,
            "fanout_risk": fanout_risk,
            "null_ratio": null_ratio,
            "edge_weight": base_weight,
            "recommended_by_default": recommended,
            "relation_type": "many_to_one" if child_unique_ratio < 0.98 else "one_to_one",
            "join_cardinality": "many_to_one" if child_unique_ratio < 0.98 else "one_to_one",
            "temporal_join_semantics": _temporal_join_semantics(child_node, parent_node, "child_to_parent"),
            "traversal_semantics": _traversal_semantics(child_node, parent_node, "child_to_parent"),
            "preferred_aggregations": _edge_preferred_aggregations(child_node, parent_node, "child_to_parent"),
            "shared_subject_entities": _shared_subjects(child_node, parent_node),
            "signal_overlap": signal_overlap,
        }
        edges.append(
            {
                "edge_id": f"{child}.{child_col}->{parent}.{parent_col}",
                "src_table": child,
                "dst_table": parent,
                "direction": "child_to_parent",
                **edge_common,
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
                    **edge_common,
                    "edge_weight": round(base_weight * 0.92, 6),
                    "recommended_by_default": bool(recommended and base_weight >= 0.64),
                    "relation_type": "one_to_many" if child_unique_ratio < 0.98 else "one_to_one",
                    "join_cardinality": "one_to_many" if child_unique_ratio < 0.98 else "one_to_one",
                    "temporal_join_semantics": _temporal_join_semantics(parent_node, child_node, "parent_to_child"),
                    "traversal_semantics": _traversal_semantics(parent_node, child_node, "parent_to_child"),
                    "preferred_aggregations": _edge_preferred_aggregations(parent_node, child_node, "parent_to_child"),
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
        node["hub_score"] = round(
            float(node["in_degree"] + node["out_degree"] + node.get("anchor_strength", 0.0)),
            3,
        )

    motifs = _build_semantic_motifs(node_by_table, edges)
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
    motifs.sort(
        key=lambda row: (
            float(row.get("motif_weight", 0.0)),
            bool(row.get("recommended_by_default", False)),
            -int(row.get("depth", 0)),
        ),
        reverse=True,
    )
    return {
        "graph_version": "semantic_context_graph.v3",
        "node_count": len(table_nodes),
        "edge_count": len(edges),
        "motif_count": len(motifs),
        "table_nodes": table_nodes,
        "edges": edges,
        "motifs": motifs,
    }


def _signal_overlap(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_families = set(str(v) for v in left.get("signal_families", []) or [])
    right_families = set(str(v) for v in right.get("signal_families", []) or [])
    if not left_families or not right_families:
        return 0.55
    overlap = len(left_families & right_families)
    union = max(1, len(left_families | right_families))
    return float(0.55 + 0.45 * (overlap / union))


def _shared_subjects(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    left_subjects = {str(v) for v in left.get("primary_subject_entities", []) or []}
    right_subjects = {str(v) for v in right.get("primary_subject_entities", []) or []}
    return sorted(left_subjects & right_subjects)[:6]


def _temporal_alignment(child_node: dict[str, Any], parent_node: dict[str, Any]) -> float:
    child_kind = str(child_node.get("temporal_kind", "none"))
    parent_kind = str(parent_node.get("temporal_kind", "none"))
    if child_kind == "none" and parent_kind == "none":
        return 0.88
    if child_kind == parent_kind:
        return 1.0
    if "event_time" in {child_kind, parent_kind} and "attribute_time" in {child_kind, parent_kind}:
        return 0.82
    if "snapshot_time" in {child_kind, parent_kind}:
        return 0.90
    if "mixed_time" in {child_kind, parent_kind}:
        return 0.86
    return 0.78


def _temporal_join_semantics(src_node: dict[str, Any], dst_node: dict[str, Any], direction: str) -> str:
    src_kind = str(src_node.get("temporal_kind", "none"))
    dst_kind = str(dst_node.get("temporal_kind", "none"))
    if src_kind == "event_time" and dst_kind in {"attribute_time", "none"}:
        return "attach_event_to_context"
    if src_kind == "event_time" and dst_kind in {"event_time", "mixed_time"}:
        return "align_related_histories"
    if src_kind == "snapshot_time" or dst_kind == "snapshot_time":
        return "align_snapshot_context"
    if direction == "parent_to_child":
        return "expand_context_into_history"
    return "lookup_context"


def _traversal_semantics(src_node: dict[str, Any], dst_node: dict[str, Any], direction: str) -> str:
    if direction == "child_to_parent":
        if str(src_node.get("table_role")) == "event":
            return "roll_up_event_context"
        return "resolve_reference_context"
    if str(dst_node.get("table_role")) == "event":
        return "expand_history"
    return "fan_out_related_records"


def _edge_preferred_aggregations(src_node: dict[str, Any], dst_node: dict[str, Any], direction: str) -> list[str]:
    out: list[str] = []
    if bool(dst_node.get("usable_for_recency", False)) or bool(src_node.get("usable_for_recency", False)):
        out.extend(["count", "last", "recent_count", "delta"])
    signal_families = set(str(v) for v in (src_node.get("signal_families", []) or []))
    signal_families |= set(str(v) for v in (dst_node.get("signal_families", []) or []))
    if "performance" in signal_families:
        out.extend(["mean", "sum", "max", "last"])
    if "status" in signal_families:
        out.extend(["count", "mode", "last"])
    if direction == "parent_to_child":
        out.append("count")
    deduped: list[str] = []
    for item in out:
        if item not in deduped:
            deduped.append(item)
    return deduped[:8]


def _build_semantic_motifs(
    node_by_table: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    adjacency: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        if not bool(edge.get("recommended_by_default", False)):
            continue
        if float(edge.get("edge_weight", 0.0)) < 0.58:
            continue
        adjacency.setdefault(str(edge.get("src_table", "")), []).append(edge)

    anchors = [
        node
        for node in node_by_table.values()
        if float(node.get("anchor_strength", 0.0)) >= 0.65 or str(node.get("table_role", "")) == "entity"
    ]
    anchors.sort(
        key=lambda node: (float(node.get("anchor_strength", 0.0)), float(node.get("hub_score", 0.0))),
        reverse=True,
    )

    motifs: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for anchor in anchors[:12]:
        anchor_table = str(anchor.get("table_name", ""))
        if not anchor_table:
            continue

        def _dfs(cur_table: str, path_tables: list[str], path_edges: list[dict[str, Any]]) -> None:
            if path_edges:
                key = (anchor_table, tuple(path_tables))
                if key not in seen:
                    seen.add(key)
                    motifs.append(_motif_from_path(anchor_table, path_tables, path_edges, node_by_table))
            if len(path_edges) >= 2:
                return
            next_edges = sorted(
                adjacency.get(cur_table, []),
                key=lambda edge: float(edge.get("edge_weight", 0.0)),
                reverse=True,
            )[:4]
            for edge in next_edges:
                nxt = str(edge.get("dst_table", ""))
                if not nxt or nxt in path_tables:
                    continue
                path_edges.append(edge)
                path_tables.append(nxt)
                _dfs(nxt, path_tables, path_edges)
                path_tables.pop()
                path_edges.pop()

        _dfs(anchor_table, [anchor_table], [])

    return motifs[:120]


def _motif_from_path(
    anchor_table: str,
    path_tables: list[str],
    path_edges: list[dict[str, Any]],
    node_by_table: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    path_nodes = [node_by_table.get(table_name, {}) for table_name in path_tables]
    families: set[str] = set()
    core_attributes: list[str] = []
    preferred_aggs: list[str] = []
    for node in path_nodes:
        families |= {str(v) for v in node.get("signal_families", []) or []}
        for column_name in node.get("important_columns", []) or []:
            if column_name not in core_attributes:
                core_attributes.append(str(column_name))
        for column_name in node.get("trusted_measure_columns", []) or []:
            if column_name not in core_attributes:
                core_attributes.append(str(column_name))
    for edge in path_edges:
        for agg in edge.get("preferred_aggregations", []) or []:
            agg_s = str(agg)
            if agg_s not in preferred_aggs:
                preferred_aggs.append(agg_s)

    motif_type = _motif_type(families, path_tables, path_nodes)
    temporal_support = _motif_temporal_support(path_nodes)
    edge_weight_mean = sum(float(edge.get("edge_weight", 0.0)) for edge in path_edges) / max(1, len(path_edges))
    history_bonus = max(float(node.get("history_value", 0.0)) for node in path_nodes)
    recommended = all(bool(edge.get("recommended_by_default", False)) for edge in path_edges)
    return {
        "motif_id": "->".join(path_tables),
        "anchor_table": anchor_table,
        "depth": len(path_edges),
        "tables": list(path_tables),
        "edge_ids": [str(edge.get("edge_id", "")) for edge in path_edges],
        "motif_type": motif_type,
        "signal_families": sorted(families)[:8],
        "core_attributes": core_attributes[:12],
        "preferred_aggregations": preferred_aggs[:10],
        "temporal_support": temporal_support,
        "motif_weight": round(min(1.0, edge_weight_mean * 0.75 + history_bonus * 0.25), 3),
        "recommended_by_default": recommended,
        "llm_rationale": _motif_rationale(anchor_table, motif_type, path_tables, families, temporal_support),
    }


def _motif_type(
    families: set[str],
    path_tables: list[str],
    path_nodes: list[dict[str, Any]],
) -> str:
    if "performance" in families:
        return "performance_history"
    if "status" in families:
        return "status_tracking"
    if "schedule" in families:
        return "event_context"
    if "profile" in families and len(path_tables) <= 2:
        return "entity_profile"
    if "geography" in families:
        return "context_lookup"
    if any(bool(node.get("usable_for_recency", False)) for node in path_nodes):
        return "historical_context"
    return "relational_context"


def _motif_temporal_support(path_nodes: list[dict[str, Any]]) -> str:
    if any(bool(node.get("usable_for_recency", False)) for node in path_nodes):
        return "strong"
    if any(str(node.get("temporal_kind", "none")) != "none" for node in path_nodes):
        return "medium"
    return "weak"


def _motif_rationale(
    anchor_table: str,
    motif_type: str,
    path_tables: list[str],
    families: set[str],
    temporal_support: str,
) -> str:
    trail = " -> ".join(path_tables)
    family_text = ", ".join(sorted(families)[:3]) if families else "general relational"
    return (
        f"{trail} forms a {motif_type} motif from anchor {anchor_table}. "
        f"It exposes {family_text} signals with {temporal_support} temporal support."
    )
