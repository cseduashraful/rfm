from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any


def sanitize_selected_paths(
    policy: dict[str, Any],
    prompt_obj: dict[str, Any],
    *,
    max_keep: int = 40,
    drop_ids: set[str] | None = None,
) -> list[str]:
    drop_ids = drop_ids or set()
    selected = policy.get("path_scoring_rules", {}).get("selected_path_ids", [])
    if not isinstance(selected, list):
        selected = []

    by_id = {p.get("path_id"): p for p in prompt_obj.get("candidate_paths", [])}
    safe_ids: list[str] = []
    seen: set[str] = set()
    for pid in selected:
        if not isinstance(pid, str) or pid in seen or pid in drop_ids:
            continue
        p = by_id.get(pid)
        if not p:
            continue
        if p.get("temporal_valid", True) is not True:
            continue
        if p.get("recommended", False) is not True:
            continue
        safe_ids.append(pid)
        seen.add(pid)
        if len(safe_ids) >= max_keep:
            break

    if not safe_ids:
        for p in prompt_obj.get("candidate_paths", []):
            pid = p.get("path_id")
            if (
                isinstance(pid, str)
                and pid not in seen
                and pid not in drop_ids
                and p.get("temporal_valid", True) is True
                and p.get("recommended", False) is True
            ):
                safe_ids.append(pid)
                seen.add(pid)
                if len(safe_ids) >= max_keep:
                    break

    policy.setdefault("path_scoring_rules", {})
    policy["path_scoring_rules"]["selected_path_ids"] = safe_ids
    return safe_ids


def build_attribute_aggregation_plan(
    *,
    selected_path_ids: list[str],
    prompt_obj: dict[str, Any],
    max_paths: int = 20,
    max_attrs_per_path: int = 5,
) -> list[dict[str, Any]]:
    by_id = {p.get("path_id"): p for p in prompt_obj.get("candidate_paths", [])}
    plan: list[dict[str, Any]] = []
    for pid in selected_path_ids[:max_paths]:
        p = by_id.get(pid)
        if not p:
            continue
        attrs = p.get("top_attribute_candidates", [])
        if not isinstance(attrs, list):
            attrs = []
        plan.append(
            {
                "path_id": pid,
                "path_tables": p.get("path_tables", []),
                "priority_attributes": attrs[:max_attrs_per_path],
            }
        )
    return plan


def _extract_hint_tables(prompt_obj: dict[str, Any]) -> set[str]:
    hints = prompt_obj.get("task_sql_feature_hints", {})
    if not isinstance(hints, dict):
        return set()
    tables = hints.get("tables", {})
    if not isinstance(tables, dict):
        return set()
    out: set[str] = set()
    for t, w in tables.items():
        try:
            wt = float(w)
        except Exception:
            wt = 0.0
        if wt > 0:
            out.add(str(t).lower())
    return out


def _extract_target_tables(prompt_obj: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    task_def = prompt_obj.get("task_definition", {})
    if isinstance(task_def, dict):
        anchor = str(task_def.get("entity_table", "")).lower()
        if anchor:
            out.add(anchor)
        output_col = str(task_def.get("output_col", "")).lower()
    else:
        output_col = ""

    for p in prompt_obj.get("candidate_paths", []):
        attrs = p.get("top_attribute_candidates", [])
        if not isinstance(attrs, list):
            continue
        for a in attrs:
            if not isinstance(a, dict):
                continue
            col = str(a.get("column", "")).lower()
            table = str(a.get("table", "")).lower()
            if output_col and col == output_col and table:
                out.add(table)
    return out


def compute_selection_quality_metrics(
    *,
    selected_path_ids: list[str],
    prompt_obj: dict[str, Any],
    preferred_max_depth: int = 3,
) -> dict[str, Any]:
    by_id = {p.get("path_id"): p for p in prompt_obj.get("candidate_paths", [])}
    selected_paths = [by_id[pid] for pid in selected_path_ids if pid in by_id]
    if not selected_paths:
        return {
            "selected_count": 0,
            "avg_relevance": 0.0,
            "one_hop_ratio": 0.0,
            "depth_overflow_ratio": 0.0,
            "hint_table_coverage_ratio": 0.0,
            "target_table_coverage_ratio": 0.0,
            "secondary_hop_diversity": 0,
            "selection_quality_score": 0.0,
            "hint_tables_covered": [],
            "target_tables_covered": [],
        }

    hint_tables = _extract_hint_tables(prompt_obj)
    target_tables = _extract_target_tables(prompt_obj)

    rels = [float(p.get("final_relevance_score", p.get("relevance_score", 0.0))) for p in selected_paths]
    depths = [int(p.get("depth", 99)) if isinstance(p.get("depth"), int) else 99 for p in selected_paths]
    one_hop = sum(1 for d in depths if d <= 1)
    overflow = sum(1 for d in depths if d > preferred_max_depth)

    covered_hint: set[str] = set()
    covered_target: set[str] = set()
    secondary_tables: set[str] = set()

    for p in selected_paths:
        tables = [str(t).lower() for t in p.get("path_tables", [])]
        for t in tables:
            if t in hint_tables:
                covered_hint.add(t)
            if t in target_tables:
                covered_target.add(t)
        if len(tables) >= 2:
            secondary_tables.add(tables[1])

    hint_cov_ratio = (len(covered_hint) / len(hint_tables)) if hint_tables else 0.0
    target_cov_ratio = (len(covered_target) / len(target_tables)) if target_tables else 0.0
    one_hop_ratio = one_hop / max(1, len(selected_paths))
    overflow_ratio = overflow / max(1, len(selected_paths))
    avg_rel = sum(rels) / max(1, len(rels))
    # set-level objective for phase-2 selection quality
    score = (
        1.00 * avg_rel
        + 0.90 * target_cov_ratio
        + 0.60 * hint_cov_ratio
        + 0.35 * one_hop_ratio
        + 0.08 * len(secondary_tables)
        - 0.60 * overflow_ratio
    )

    return {
        "selected_count": len(selected_paths),
        "avg_relevance": round(avg_rel, 4),
        "one_hop_ratio": round(one_hop_ratio, 4),
        "depth_overflow_ratio": round(overflow_ratio, 4),
        "hint_table_coverage_ratio": round(hint_cov_ratio, 4),
        "target_table_coverage_ratio": round(target_cov_ratio, 4),
        "secondary_hop_diversity": len(secondary_tables),
        "selection_quality_score": round(score, 4),
        "hint_tables_covered": sorted(covered_hint),
        "target_tables_covered": sorted(covered_target),
    }


def auto_finalize_selected_paths(
    *,
    prompt_obj: dict[str, Any],
    current_selected: list[str],
    task_priors: dict[str, Any],
    max_keep: int = 10,
) -> list[str]:
    ranked = sorted(
        prompt_obj.get("candidate_paths", []),
        key=lambda p: float(p.get("final_relevance_score", p.get("relevance_score", 0.0))),
        reverse=True,
    )

    pref_max_depth = int(task_priors.get("preferred_max_depth", 3))
    anchor = str(task_priors.get("anchor_table", prompt_obj.get("task_definition", {}).get("entity_table", ""))).lower()

    def _eligible(p: dict[str, Any]) -> bool:
        pid = p.get("path_id")
        if not isinstance(pid, str):
            return False
        tables = [str(x).lower() for x in p.get("path_tables", [])]
        depth = p.get("depth")
        if tables and tables[0] != anchor:
            return False
        if isinstance(depth, int) and depth > pref_max_depth + 1:
            return False
        if p.get("recommended", False) is not True:
            return False
        if p.get("temporal_valid", True) is not True:
            return False
        return True

    pool = [p for p in ranked if _eligible(p)][: max(30, max_keep * 8)]
    if not pool:
        return [pid for pid in current_selected if isinstance(pid, str)][:max_keep]

    core_ids: list[str] = []
    for pid in current_selected:
        if not isinstance(pid, str):
            continue
        if any(pid == p.get("path_id") for p in pool):
            core_ids.append(pid)
        if len(core_ids) >= min(3, max_keep):
            break

    selected: list[str] = core_ids[:]
    selected_set = set(selected)
    while len(selected) < max_keep:
        best_pid = None
        best_score = None
        for p in pool:
            pid = p.get("path_id")
            if not isinstance(pid, str) or pid in selected_set:
                continue
            trial = selected + [pid]
            m = compute_selection_quality_metrics(
                selected_path_ids=trial,
                prompt_obj=prompt_obj,
                preferred_max_depth=pref_max_depth,
            )
            objective = float(m.get("selection_quality_score", 0.0))
            if best_score is None or objective > best_score:
                best_score = objective
                best_pid = pid
        if best_pid is None:
            break
        selected.append(best_pid)
        selected_set.add(best_pid)

    if len(selected) < 3 and current_selected:
        selected = [pid for pid in current_selected if isinstance(pid, str)][:max_keep]
    return selected[:max_keep]


def build_attribute_importance_summary(
    *,
    selected_path_ids: list[str],
    prompt_obj: dict[str, Any],
    max_attributes: int = 40,
) -> dict[str, Any]:
    by_id = {p.get("path_id"): p for p in prompt_obj.get("candidate_paths", [])}
    agg: dict[tuple[str, str], dict[str, Any]] = {}

    for pid in selected_path_ids:
        p = by_id.get(pid)
        if not p:
            continue
        attrs = p.get("top_attribute_candidates", [])
        if not isinstance(attrs, list):
            continue
        for a in attrs:
            if not isinstance(a, dict):
                continue
            t = str(a.get("table", "")).strip()
            c = str(a.get("column", "")).strip()
            if not t or not c:
                continue
            k = (t, c)
            if k not in agg:
                agg[k] = {
                    "table": t,
                    "column": c,
                    "importance_score_sum": 0.0,
                    "path_count": 0,
                    "semantic_roles": defaultdict(int),
                    "suggested_aggregations": defaultdict(int),
                }
            row = agg[k]
            try:
                row["importance_score_sum"] += float(a.get("importance_score", 0.0) or 0.0)
            except Exception:
                pass
            row["path_count"] += 1
            role = str(a.get("semantic_role", ""))
            if role:
                row["semantic_roles"][role] += 1
            aggs = a.get("suggested_aggregations", [])
            if isinstance(aggs, list):
                for g in aggs:
                    sg = str(g).strip()
                    if sg:
                        row["suggested_aggregations"][sg] += 1

    rows: list[dict[str, Any]] = []
    for _, v in agg.items():
        role_counts = sorted(v["semantic_roles"].items(), key=lambda kv: (-kv[1], kv[0]))
        agg_counts = sorted(v["suggested_aggregations"].items(), key=lambda kv: (-kv[1], kv[0]))
        rows.append(
            {
                "table": v["table"],
                "column": v["column"],
                "importance_score_sum": round(float(v["importance_score_sum"]), 4),
                "path_count": int(v["path_count"]),
                "top_semantic_roles": [x[0] for x in role_counts[:3]],
                "top_suggested_aggregations": [x[0] for x in agg_counts[:6]],
            }
        )

    rows.sort(
        key=lambda x: (
            float(x.get("importance_score_sum", 0.0)),
            int(x.get("path_count", 0)),
        ),
        reverse=True,
    )

    by_table: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_table[str(r["table"])].append(r)

    return {
        "selected_path_count": len([pid for pid in selected_path_ids if isinstance(pid, str)]),
        "top_attributes": rows[:max_attributes],
        "by_table": {k: v[:12] for k, v in by_table.items()},
    }


def print_top_selected_paths(
    *,
    prompt_obj: dict[str, Any],
    selected_path_ids: list[str],
    top_k: int = 10,
) -> None:
    by_id = {p.get("path_id"): p for p in prompt_obj.get("candidate_paths", [])}
    rows: list[dict[str, Any]] = []
    for pid in selected_path_ids:
        p = by_id.get(pid)
        if not p:
            continue
        rows.append(
            {
                "path_id": pid,
                "relevance_score": float(
                    p.get("final_relevance_score", p.get("relevance_score", 0.0))
                ),
                "depth": p.get("depth"),
                "path_tables": p.get("path_tables", []),
            }
        )
    rows.sort(key=lambda x: x["relevance_score"], reverse=True)
    rows = rows[: max(0, top_k)]
    print("[phase2] top task-specific selected meta-paths by relevance_score:")
    for r in rows:
        tables = "->".join(str(t) for t in r.get("path_tables", []))
        print(
            f"[phase2]   {r['path_id']} "
            f"score={r['relevance_score']:.3f} depth={r.get('depth')} "
            f"tables={tables}"
        )


def merge_solver_patch(base_policy: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base_policy)
    for key in [
        "path_scoring_rules",
        "feature_rules",
        "depth_policy",
        "budget_policy",
        "safety_constraints",
    ]:
        if isinstance(patch.get(key), dict):
            merged[key].update(patch[key])
    merged["path_scoring_rules"]["exclude_low_confidence_fk_by_default"] = True
    merged["path_scoring_rules"].setdefault("path_exclusions", {})
    merged["path_scoring_rules"]["path_exclusions"]["exclude_low_confidence_fk_by_default"] = True
    merged["safety_constraints"]["exclude_low_confidence_fk_by_default"] = True
    return merged
