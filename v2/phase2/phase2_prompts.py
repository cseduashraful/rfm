from __future__ import annotations

import json
from typing import Any


def _json_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"))


def build_solver_prompt(
    *,
    context: dict[str, Any],
    previous_spec: dict[str, Any] | None,
    critic_feedback: dict[str, Any] | None,
) -> str:
    return (
        "You are the SOLVER in a task compiler pipeline.\n"
        "Produce STRICT JSON only. No markdown, no prose.\n"
        "Goal: compile a task_spec for relational retrieval and feature policy.\n"
        "Use task-aware anchor-first reasoning.\n"
        "Must obey:\n"
        "1) Start from the anchor entity in task_definition (entity_table/entity_col).\n"
        "2) Select meta-paths that are most relevant to the prediction target.\n"
        "3) Prefer paths that directly connect anchor context to predictive signals.\n"
        "4) Exclude low-confidence FK links by default.\n"
        "5) Keep policy dataset-aware and task-agnostic.\n"
        "6) Use provided budget defaults unless strong reason to adjust.\n\n"
        "If critic suggests better alternatives, replace weaker selected paths.\n"
        "You MUST address prior critic major issues explicitly.\n"
        "Optimize selected_path_ids for set-level quality: coverage + relevance + diversity + budget compliance.\n"
        "For each prior major issue, include an item in `addressed_issues` with status resolved/deferred and a concrete action.\n"
        "Output JSON schema (required keys):\n"
        "{\n"
        "  \"path_scoring_rules\": {...},\n"
        "  \"feature_rules\": {...},\n"
        "  \"depth_policy\": {...},\n"
        "  \"budget_policy\": {...},\n"
        "  \"safety_constraints\": {...},\n"
        "  \"addressed_issues\": [{\"issue_id\":\"...\",\"status\":\"resolved|deferred\",\"action\":\"...\"}]\n"
        "}\n\n"
        f"Compilation context:\n{_json_compact(context)}\n\n"
        f"Previous spec (or null):\n{_json_compact(previous_spec)}\n\n"
        f"Critic feedback (or null):\n{_json_compact(critic_feedback)}\n"
    )


def build_critic_prompt(
    *,
    context: dict[str, Any],
    candidate_spec: dict[str, Any],
) -> str:
    return (
        "You are the CRITIC in a task compiler pipeline.\n"
        "Return STRICT JSON only. No markdown, no prose.\n"
        "Evaluate candidate_spec and report major/minor issues and concrete fixes.\n"
        "Your role is RELEVANCE critic only (not leakage critic).\n"
        "Assume temporal safety is handled by the retrieval layer using strict as-of filtering.\n"
        "Do NOT flag temporal leakage, cutoff issues, or future-row leakage.\n"
        "Focus on selected_path_ids and policy-level consistency; do not enumerate every candidate path.\n"
        "Keep output concise: at most 5 major issues, 5 minor issues, and 5 fixes.\n"
        "Group repeated path-level problems into one summary issue.\n"
        "When possible, each major issue should map to concrete path IDs or explicit policy fields.\n"
        "Checks (objective, measurable):\n"
        "- selected_set_quality.selection_quality_score should not degrade vs previous round (apply ONLY when context.round_index >= 2)\n"
        "- selected_set_quality.target_table_coverage_ratio should be high when target-related tables exist\n"
        "- selected_set_quality.hint_table_coverage_ratio should be high when SQL hints exist\n"
        "- selected_set_quality.one_hop_ratio should be non-trivial (not all deep paths)\n"
        "- selected_set_quality.depth_overflow_ratio should be low\n"
        "- semantic coherence between selected paths and task definition\n"
        "- whether unselected top candidates are clearly better than selected ones\n"
        "- redundancy/diversity and budget realism\n"
        "If metrics are acceptable, set no_major_issues=true.\n\n"
        "Output schema:\n"
        "{\n"
        "  \"no_major_issues\": true|false,\n"
        "  \"major_issues\": [{\"issue_id\":\"M1\",\"issue\":\"...\",\"affected_path_ids\":[\"p000001\"],\"target_fields\":[\"path_scoring_rules.selected_path_ids\"]}],\n"
        "  \"minor_issues\": [\"...\"],\n"
        "  \"fixes\": [{\"issue_id\":\"M1\",\"fix\":\"...\",\"priority\":\"high|medium|low\"}]\n"
        "}\n\n"
        f"Compilation context:\n{_json_compact(context)}\n\n"
        f"Candidate spec:\n{_json_compact(candidate_spec)}\n"
    )
