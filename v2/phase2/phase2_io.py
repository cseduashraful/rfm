from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_phase1_bundle(phase1_root: Path, dataset: str) -> dict[str, dict[str, Any]]:
    base = phase1_root / dataset
    required = {
        "schema": base / "schema.json",
        "stats": base / "stats.json",
        "semantics": base / "semantics.json",
        "safety_rules": base / "safety_rules.json",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing Phase 1 artifacts for dataset={dataset}: {missing}. Expected under {base}"
        )
    bundle = {name: read_json(path) for name, path in required.items()}
    # Backward-compatible optional artifact.
    optional_path_catalog = base / "path_catalog.json"
    bundle["path_catalog"] = read_json(optional_path_catalog) if optional_path_catalog.exists() else {"paths": []}
    optional_semantic_graph = base / "semantic_context_graph.json"
    bundle["semantic_context_graph"] = (
        read_json(optional_semantic_graph) if optional_semantic_graph.exists() else {"table_nodes": [], "edges": []}
    )
    return bundle
