from __future__ import annotations

import re
from typing import Any

import pandas as pd


def norm_identifier_tokens(name: str) -> set[str]:
    low = re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()
    toks = {t for t in low.split() if len(t) >= 2}
    expanded: set[str] = set(toks)
    for t in list(toks):
        if t.endswith("s") and len(t) > 3:
            expanded.add(t[:-1])
        if t.endswith("es") and len(t) > 4:
            expanded.add(t[:-2])
    return expanded


def infer_fk_confidence(*, overlap_ratio: float, name_compatible: bool, uniqueness: float) -> str | None:
    if overlap_ratio >= 0.98 and name_compatible and uniqueness <= 0.99:
        return "high"
    if overlap_ratio >= 0.93 and name_compatible and uniqueness <= 0.98:
        return "medium"
    return None


def infer_fk_candidates(db: Any, filtered: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    pk_map: dict[str, str | None] = {tname: t.pkey_col for tname, t in db.table_dict.items()}

    generic_id_names = {"id", "statusid", "typeid", "classid"}
    for child_name, child_df in filtered.items():
        if child_df.empty:
            continue
        for col in child_df.columns:
            col_l = col.lower()
            if not col_l.endswith("id"):
                continue
            if col_l in generic_id_names:
                continue
            child_vals = set(child_df[col].dropna().unique().tolist())
            if not child_vals:
                continue
            for parent_name, parent_df in filtered.items():
                parent_pk = pk_map.get(parent_name)
                if parent_pk is None or parent_pk not in parent_df.columns:
                    continue
                if parent_name == child_name:
                    continue
                parent_vals = set(parent_df[parent_pk].dropna().unique().tolist())
                if not parent_vals:
                    continue
                overlap = len(child_vals & parent_vals)
                if overlap == 0:
                    continue
                ratio = overlap / max(1, len(child_vals))
                child_tok = norm_identifier_tokens(col_l.replace("id", ""))
                parent_tok = norm_identifier_tokens(parent_name)
                parent_pk_tok = norm_identifier_tokens(parent_pk)
                name_compatible = bool(child_tok & (parent_tok | parent_pk_tok))
                non_null = int(child_df[col].notna().sum())
                uniq = int(child_df[col].nunique(dropna=True))
                uniqueness = float(uniq / max(1, non_null)) if non_null > 0 else 0.0

                confidence = infer_fk_confidence(
                    overlap_ratio=ratio,
                    name_compatible=name_compatible,
                    uniqueness=uniqueness,
                )
                if confidence is None:
                    continue
                out.append(
                    {
                        "child_table": child_name,
                        "child_column": col,
                        "parent_table": parent_name,
                        "parent_column": parent_pk,
                        "overlap_ratio": ratio,
                        "confidence": confidence,
                        "recommended_by_default": True,
                    }
                )
    return out
