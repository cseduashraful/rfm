from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


REQUIRED_TASK_SPEC_KEYS = {
    "artifact_version",
    "dataset_name",
    "task_name",
    "generation_timestamp_utc",
    "source_artifacts",
    "task_definition",
    "path_scoring_rules",
    "feature_rules",
    "depth_policy",
    "budget_policy",
    "safety_constraints",
    "solver_critic_diagnostics",
    "validation",
}


@dataclass
class TaskDefinition:
    entity_table: str
    entity_col: str
    time_col: str
    output_col: str
    task_type: str


@dataclass
class BudgetPolicy:
    self_history_budget: int = 10
    one_hop_budget: int = 10
    multi_hop_budget_total: int = 50
    neighbor_budget: int = 10


@dataclass
class RoundRecord:
    round_index: int
    role: str
    model_name: str
    prompt_preview: str
    parsed_ok: bool
    payload: dict[str, Any]


@dataclass
class CompilationContext:
    dataset: str
    task: str
    task_definition: TaskDefinition
    phase1_artifacts_dir: str
    cutoff_time: str
    candidate_paths: list[dict[str, Any]] = field(default_factory=list)
    recommended_path_ids: list[str] = field(default_factory=list)
    low_confidence_excluded_count: int = 0
    semantics_summary: dict[str, Any] = field(default_factory=dict)
    stats_summary: dict[str, Any] = field(default_factory=dict)
    safety_rules: dict[str, Any] = field(default_factory=dict)
