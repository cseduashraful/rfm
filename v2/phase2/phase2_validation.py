from __future__ import annotations

from typing import Any

from phase2_models import REQUIRED_TASK_SPEC_KEYS


def validate_task_spec_schema(spec: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    missing = sorted(REQUIRED_TASK_SPEC_KEYS - set(spec.keys()))
    if missing:
        issues.append(f"missing_top_level_keys={missing}")

    for key in [
        "task_definition",
        "path_scoring_rules",
        "feature_rules",
        "depth_policy",
        "budget_policy",
        "safety_constraints",
        "solver_critic_diagnostics",
        "validation",
    ]:
        if key in spec and not isinstance(spec[key], dict):
            issues.append(f"{key}_must_be_object")

    return issues


def validate_task_spec_consistency(
    spec: dict[str, Any],
    *,
    candidate_path_ids: set[str],
    phase1_cutoff: str,
) -> list[str]:
    issues: list[str] = []

    selected = spec.get("path_scoring_rules", {}).get("selected_path_ids", [])
    if not isinstance(selected, list):
        issues.append("selected_path_ids_must_be_list")
        selected = []
    unknown = [pid for pid in selected if pid not in candidate_path_ids]
    if unknown:
        issues.append(f"unknown_selected_path_ids={unknown[:20]}")

    budget = spec.get("budget_policy", {})
    for k in ["self_history_budget", "one_hop_budget", "multi_hop_budget_total", "neighbor_budget"]:
        v = budget.get(k)
        if not isinstance(v, int) or v < 0:
            issues.append(f"invalid_budget_{k}={v}")

    depth = spec.get("depth_policy", {}).get("default_max_depth")
    if not isinstance(depth, int) or depth <= 0:
        issues.append(f"invalid_default_max_depth={depth}")

    cutoff = spec.get("safety_constraints", {}).get("global_temporal_cutoff")
    if cutoff is not None and str(cutoff) != str(phase1_cutoff):
        issues.append(f"cutoff_mismatch_phase1={phase1_cutoff}_spec={cutoff}")

    excluded_low = spec.get("path_scoring_rules", {}).get("exclude_low_confidence_fk_by_default")
    if excluded_low is not True:
        issues.append("exclude_low_confidence_fk_by_default_should_be_true")

    return issues
