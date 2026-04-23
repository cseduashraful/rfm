from __future__ import annotations

import math
import re
from typing import Any


def llm_rescore_candidate_paths(
    *,
    llm: Any,
    prompt_obj: dict[str, Any],
    max_candidates: int = 220,
    batch_size: int = 20,
) -> dict[str, Any]:
    candidates = prompt_obj.get("candidate_paths", [])
    if not isinstance(candidates, list) or not candidates:
        return {"scored": 0, "failed_batches": 0, "batches": 0}

    ranked = sorted(
        candidates,
        key=lambda p: float(p.get("relevance_score", 0.0)),
        reverse=True,
    )[: max(1, max_candidates)]
    by_id = {str(p.get("path_id")): p for p in ranked if isinstance(p.get("path_id"), str)}
    hints = prompt_obj.get("task_sql_feature_hints", {})
    hint_tables: set[str] = set()
    if isinstance(hints, dict):
        ht = hints.get("tables", {})
        if isinstance(ht, dict):
            hint_tables = {str(k).lower() for k in ht.keys()}
    sem_summary = prompt_obj.get("semantics_summary", {})
    stats_summary = prompt_obj.get("stats_summary", {})

    def _normalize_pid(raw: Any) -> str | None:
        s = str(raw).strip()
        if not s:
            return None
        m = re.search(r"p\s*0*(\d{1,6})", s, flags=re.IGNORECASE)
        if m:
            return f"p{int(m.group(1)):06d}"
        m2 = re.search(r"\b(\d{1,6})\b", s)
        if m2:
            return f"p{int(m2.group(1)):06d}"
        return None

    failed_batches = 0
    batches = 0
    total_matched = 0
    total_items = 0

    def _apply_order_scores(batch_paths: list[dict[str, Any]], values: list[Any]) -> int:
        assigned = 0
        for idx2, p in enumerate(batch_paths):
            if idx2 >= len(values):
                break
            raw = values[idx2]
            rel = 0.0
            conf = 0.0
            rationale = ""
            if isinstance(raw, dict):
                rel_raw = raw.get("task_relevance", raw.get("score", 0.0))
                try:
                    rel = float(rel_raw)
                except Exception:
                    rel = 0.0
                try:
                    conf = float(raw.get("confidence", 0.0))
                except Exception:
                    conf = 0.0
                rationale = str(raw.get("rationale_short", ""))
            else:
                try:
                    rel = float(raw)
                except Exception:
                    mrel = re.search(r"(-?\d+(?:\.\d+)?)", str(raw))
                    rel = float(mrel.group(1)) if mrel else 0.0
            rel = max(0.0, min(10.0, rel))
            conf = max(0.0, min(1.0, conf))
            p["llm_task_relevance"] = rel
            p["llm_confidence"] = conf
            p["llm_rationale_short"] = rationale[:240]
            assigned += 1
        return assigned

    for i in range(0, len(ranked), max(1, batch_size)):
        batch = ranked[i : i + max(1, batch_size)]
        batches += 1
        batch_payload: list[dict[str, Any]] = []
        for p in batch:
            edges = p.get("path_edges", [])
            edge_weight_mean = 0.0
            if isinstance(edges, list) and edges:
                edge_weight_mean = sum(float(e.get("edge_weight", 0.6)) for e in edges) / max(1, len(edges))
            top_attrs = []
            attrs = p.get("top_attribute_candidates", [])
            if isinstance(attrs, list):
                for a in attrs[:3]:
                    if not isinstance(a, dict):
                        continue
                    top_attrs.append(
                        {
                            "table": a.get("table"),
                            "column": a.get("column"),
                            "role": a.get("semantic_role"),
                            "importance": a.get("importance_score"),
                        }
                    )
            batch_payload.append(
                {
                    "path_id": p.get("path_id"),
                    "depth": p.get("depth"),
                    "estimated_row_multiplier": p.get("estimated_row_multiplier"),
                    "path_tables": p.get("path_tables", []),
                    "top_attributes": top_attrs,
                    "edge_weight_mean": round(float(edge_weight_mean), 4),
                    "recommended": p.get("recommended", False),
                }
            )

        prompt = (
            "You are scoring task-specific relevance of relational meta-paths.\n"
            "Return STRICT JSON only with keys:\n"
            "{"
            "\"scores\":[{\"path_id\":str,\"task_relevance\":float,\"confidence\":float,\"rationale_short\":str}]"
            "}\n"
            "Rules:\n"
            "- Score task_relevance in [0,10].\n"
            "- Use task definition as primary signal; path depth/cardinality as secondary.\n"
            "- Prefer direct, informative, low-noise paths.\n"
            "- Do not assume any specific dataset semantics beyond provided context.\n"
            "- Include every input path_id exactly once in scores.\n"
            "- Copy each path_id EXACTLY as given (e.g., p000123).\n\n"
            f"Task definition: {prompt_obj.get('task_definition', {})}\n"
            f"Global cutoff: {prompt_obj.get('global_cutoff_time')}\n"
            f"SQL feature hints (high-priority context): {prompt_obj.get('task_sql_feature_hints', {})}\n"
            f"Semantics summary (dataset context): {sem_summary}\n"
            f"Stats summary (dataset context): {stats_summary}\n"
            "When SQL hints indicate touched tables/columns, favor paths that cover those hints.\n"
            f"Candidate paths: {batch_payload}\n"
        )
        parsed, _raw = llm.generate_json(prompt, max_new_tokens=700, retries=1)
        if not isinstance(parsed, dict):
            failed_batches += 1
            continue
        scores = parsed.get("scores", [])
        if not isinstance(scores, list):
            scores = parsed.get("path_scores", [])
        if isinstance(scores, dict):
            scores = [{"path_id": k, "task_relevance": v} for k, v in scores.items()]
        if not isinstance(scores, list):
            alt = parsed.get("ranked_paths", [])
            if isinstance(alt, list):
                scores = alt
        if not isinstance(scores, list):
            scores = []
        total_items += len(scores)
        batch_matched = 0
        for s in scores:
            pid = None
            rel = 0.0
            conf = 0.0
            rationale = ""
            if isinstance(s, dict):
                pid = (
                    _normalize_pid(s.get("path_id"))
                    or _normalize_pid(s.get("pathId"))
                    or _normalize_pid(s.get("id"))
                )
                rel_raw = (
                    s.get("task_relevance", None)
                    if s.get("task_relevance", None) is not None
                    else s.get("score", None)
                )
                try:
                    rel = float(rel_raw if rel_raw is not None else 0.0)
                except Exception:
                    rel = 0.0
                try:
                    conf = float(s.get("confidence", 0.0))
                except Exception:
                    conf = 0.0
                rationale = str(s.get("rationale_short", ""))
            else:
                text = str(s)
                pid = _normalize_pid(text)
                mrel = re.search(r"(-?\d+(?:\.\d+)?)", text)
                if mrel:
                    try:
                        rel = float(mrel.group(1))
                    except Exception:
                        rel = 0.0
                rationale = text

            rel = max(0.0, min(10.0, rel))
            conf = max(0.0, min(1.0, conf))
            if pid in by_id:
                p = by_id[pid]
                p["llm_task_relevance"] = rel
                p["llm_confidence"] = conf
                p["llm_rationale_short"] = rationale[:240]
                batch_matched += 1

        if batch_matched == 0 and scores:
            batch_matched += _apply_order_scores(batch, scores)

        if batch_matched == 0:
            compact_prompt = (
                "Score each candidate path for task relevance.\n"
                "Return STRICT JSON only as:\n"
                "{\"relevance\":[<float0to10>, ...]}\n"
                f"The list length MUST be exactly {len(batch_payload)} in the same order as input.\n"
                f"Task definition: {prompt_obj.get('task_definition', {})}\n"
                f"Candidates (ordered): {batch_payload}\n"
            )
            parsed2, _raw2 = llm.generate_json(compact_prompt, max_new_tokens=220, retries=1)
            rel_list: list[Any] = []
            if isinstance(parsed2, dict):
                maybe = parsed2.get("relevance", [])
                if isinstance(maybe, list):
                    rel_list = maybe
                elif isinstance(maybe, dict):
                    rel_list = [v for _, v in sorted(maybe.items(), key=lambda kv: str(kv[0]))]
            if rel_list:
                total_items += len(rel_list)
                batch_matched += _apply_order_scores(batch, rel_list)

        total_matched += batch_matched
        if batch_matched == 0:
            failed_batches += 1

    scored = 0
    path_meta: list[dict[str, Any]] = []
    for p in ranked:
        struct = float(p.get("relevance_score", 0.0))
        llm_rel = float(p.get("llm_task_relevance", 0.0))
        llm_conf = float(p.get("llm_confidence", 0.0))
        depth = p.get("depth")
        depth_pen = 0.0
        if isinstance(depth, int):
            depth_pen = 0.35 * max(depth - 2, 0)
        tables_l = [str(t).lower() for t in p.get("path_tables", [])]
        hint_overlap = 0
        if hint_tables:
            hint_overlap = len(set(tables_l) & hint_tables)
        hint_bonus = min(hint_overlap, 3) * 0.30
        intermediate = tables_l[1:-1] if len(tables_l) > 2 else []
        non_hint_intermediate = [t for t in intermediate if t not in hint_tables] if hint_tables else []
        directness_pen = 0.40 * len(non_hint_intermediate)
        row_mult = float(p.get("estimated_row_multiplier", 1.0))
        cardinality_pen = 0.12 * math.log1p(max(row_mult, 1.0))
        p["final_relevance_score"] = round(
            2.2 * llm_rel
            + 0.30 * struct
            + 0.25 * llm_conf
            + hint_bonus
            - depth_pen,
            3,
        )
        p["final_relevance_score"] = round(
            float(p["final_relevance_score"]) - directness_pen - cardinality_pen,
            3,
        )
        p["hint_overlap_count"] = int(hint_overlap)
        p["directness_penalty"] = round(float(directness_pen), 3)
        p["cardinality_penalty"] = round(float(cardinality_pen), 3)
        path_meta.append(
            {
                "path_id": p.get("path_id"),
                "depth": int(depth) if isinstance(depth, int) else 99,
                "hint_overlap_count": int(hint_overlap),
                "table_count": len(tables_l),
                "tables": tables_l,
            }
        )
        if "llm_task_relevance" in p:
            scored += 1

    best_shortest: dict[tuple[str, tuple[str, ...]], tuple[int, int]] = {}
    by_pid: dict[str, dict[str, Any]] = {
        str(p.get("path_id")): p for p in ranked if isinstance(p.get("path_id"), str)
    }
    for m in path_meta:
        pid = m.get("path_id")
        if not isinstance(pid, str):
            continue
        tables = m.get("tables", [])
        if not isinstance(tables, list) or not tables:
            continue
        anchor = str(tables[0])
        hint_cov = tuple(sorted(set(t for t in tables if t in hint_tables))) if hint_tables else tuple()
        key = (anchor, hint_cov)
        depth_i = int(m.get("depth", 99))
        table_count_i = int(m.get("table_count", 999))
        prev = best_shortest.get(key)
        if prev is None or (depth_i, table_count_i) < prev:
            best_shortest[key] = (depth_i, table_count_i)

    for m in path_meta:
        pid = m.get("path_id")
        if not isinstance(pid, str):
            continue
        p = by_pid.get(pid)
        if not p:
            continue
        tables = m.get("tables", [])
        if not isinstance(tables, list) or not tables:
            continue
        anchor = str(tables[0])
        hint_cov = tuple(sorted(set(t for t in tables if t in hint_tables))) if hint_tables else tuple()
        key = (anchor, hint_cov)
        best = best_shortest.get(key)
        if not best:
            continue
        cur_depth = int(m.get("depth", 99))
        cur_table_count = int(m.get("table_count", 999))
        if (cur_depth, cur_table_count) > best:
            depth_diff = max(0, cur_depth - best[0])
            table_diff = max(0, cur_table_count - best[1])
            dominated_pen = 0.45 * depth_diff + 0.20 * table_diff
            p["dominated_path_penalty"] = round(float(dominated_pen), 3)
            p["final_relevance_score"] = round(
                float(p.get("final_relevance_score", 0.0)) - dominated_pen,
                3,
            )

    candidates.sort(
        key=lambda p: float(p.get("final_relevance_score", p.get("relevance_score", 0.0))),
        reverse=True,
    )
    return {
        "scored": scored,
        "failed_batches": failed_batches,
        "batches": batches,
        "matched_scores": total_matched,
        "returned_score_items": total_items,
    }
