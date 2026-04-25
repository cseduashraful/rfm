from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from phase1_data_access import Phase1DataAccess, TableMetadata
from phase1_utils import (
    is_numeric_dtype_name,
    normalize_identifier_tokens,
    quote_ident,
)


@dataclass(slots=True)
class ForeignKeyCandidate:
    child_table: str
    child_column: str
    parent_table: str
    parent_column: str
    overlap_ratio: float
    confidence: str
    recommended_by_default: bool
    source: str
    name_compatible: bool
    child_unique_ratio: float
    null_ratio: float
    exact_check: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "child_table": self.child_table,
            "child_column": self.child_column,
            "parent_table": self.parent_table,
            "parent_column": self.parent_column,
            "overlap_ratio": self.overlap_ratio,
            "confidence": self.confidence,
            "recommended_by_default": self.recommended_by_default,
            "source": self.source,
            "name_compatible": self.name_compatible,
            "child_unique_ratio": self.child_unique_ratio,
            "null_ratio": self.null_ratio,
            "exact_check": self.exact_check,
        }


def score_provided_foreign_keys(
    *,
    access: Phase1DataAccess,
    config: Any,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for table_name in access.table_names:
        meta = access.table_meta(table_name)
        for child_column, parent_table in meta.provided_foreign_keys.items():
            if parent_table not in access.table_names:
                continue
            parent_meta = access.table_meta(parent_table)
            parent_pk = parent_meta.provided_primary_key
            if parent_pk is None:
                continue
            candidate = _score_candidate_pair(
                access=access,
                child_meta=meta,
                child_column=child_column,
                parent_meta=parent_meta,
                parent_column=parent_pk,
                exact_check=True,
                source="provided",
            )
            out.append(candidate.to_dict())
    return out


def infer_fk_candidates(
    *,
    access: Phase1DataAccess,
    config: Any,
) -> list[dict[str, Any]]:
    shortlist: list[tuple[TableMetadata, str, TableMetadata, str, bool, float, float]] = []
    max_pairs = int(getattr(config, "max_fk_pairs_exact", 250))
    include_low = bool(getattr(config, "include_low_confidence_fks", True))

    for child_table in access.table_names:
        child_meta = access.table_meta(child_table)
        child_candidates = _candidate_child_fk_columns(child_meta)
        for child_column in child_candidates:
            child_dtype = child_meta.dtypes.get(child_column, "")
            child_tokens = normalize_identifier_tokens(child_column.replace("id", ""))
            for parent_table in access.table_names:
                if parent_table == child_table:
                    continue
                parent_meta = access.table_meta(parent_table)
                parent_pk = parent_meta.provided_primary_key
                if parent_pk is None or parent_pk not in parent_meta.dtypes:
                    continue
                if not _compatible_dtypes(child_dtype, parent_meta.dtypes.get(parent_pk, "")):
                    continue
                parent_tokens = normalize_identifier_tokens(parent_table) | normalize_identifier_tokens(parent_pk)
                name_compatible = bool(child_tokens & parent_tokens)
                if not name_compatible and len(child_tokens) > 0:
                    continue
                null_ratio = _column_null_ratio(access, child_meta, child_column)
                if null_ratio >= float(getattr(config, "max_fk_null_ratio", 0.98)):
                    continue
                uniqueness = _column_unique_ratio(
                    access=access,
                    table_meta=child_meta,
                    column_name=child_column,
                    sampled=True,
                    sample_size=int(getattr(config, "fk_overlap_sample_size", 50_000)),
                )
                overlap_ratio = _overlap_ratio(
                    access=access,
                    child_meta=child_meta,
                    child_column=child_column,
                    parent_meta=parent_meta,
                    parent_column=parent_pk,
                    sampled=True,
                    sample_size=int(getattr(config, "fk_overlap_sample_size", 50_000)),
                )
                shortlist.append(
                    (
                        child_meta,
                        child_column,
                        parent_meta,
                        parent_pk,
                        name_compatible,
                        uniqueness,
                        overlap_ratio,
                    )
                )

    shortlist.sort(
        key=lambda item: (
            float(item[6]),
            1.0 - float(item[5]),
            len(item[0].columns),
        ),
        reverse=True,
    )

    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for child_meta, child_column, parent_meta, parent_pk, name_compatible, uniqueness, _ in shortlist[:max_pairs]:
        null_ratio = _column_null_ratio(access, child_meta, child_column)
        scored = _score_candidate_pair(
            access=access,
            child_meta=child_meta,
            child_column=child_column,
            parent_meta=parent_meta,
            parent_column=parent_pk,
            exact_check=True,
            source="inferred",
        )
        key = (
            scored.child_table,
            scored.child_column,
            scored.parent_table,
            scored.parent_column,
        )
        if key in seen:
            continue
        seen.add(key)
        if scored.confidence == "low" and not include_low:
            continue
        if not name_compatible and scored.confidence != "high":
            continue
        scored.name_compatible = name_compatible
        scored.child_unique_ratio = uniqueness
        scored.null_ratio = null_ratio
        out.append(scored.to_dict())
    return out


def _candidate_child_fk_columns(table_meta: TableMetadata) -> list[str]:
    out: list[str] = []
    pk_lower = str(table_meta.provided_primary_key or "").lower()
    for column_name in table_meta.columns:
        lower = column_name.lower()
        if pk_lower and lower == pk_lower:
            continue
        if lower.endswith("id") or lower.endswith("_key") or lower in {"parent", "owner"}:
            out.append(column_name)
    return out


def _compatible_dtypes(child_dtype: str, parent_dtype: str) -> bool:
    child_low = str(child_dtype).lower()
    parent_low = str(parent_dtype).lower()
    if child_low == parent_low:
        return True
    if is_numeric_dtype_name(child_low) and is_numeric_dtype_name(parent_low):
        return True
    if "uuid" in child_low and "uuid" in parent_low:
        return True
    if "varchar" in child_low and "varchar" in parent_low:
        return True
    return False


def _column_null_ratio(
    access: Phase1DataAccess,
    table_meta: TableMetadata,
    column_name: str,
) -> float:
    rows = access.count_rows(table_meta.table_name, filtered=True)
    if rows <= 0:
        return 1.0
    qcol = quote_ident(column_name)
    relation_sql = access.relation_sql(table_meta.table_name, filtered=True)
    null_count = access.fetch_scalar(
        f"SELECT SUM(CASE WHEN {qcol} IS NULL THEN 1 ELSE 0 END) FROM ({relation_sql}) rel"
    )
    return float(int(null_count or 0) / max(1, rows))


def _column_unique_ratio(
    *,
    access: Phase1DataAccess,
    table_meta: TableMetadata,
    column_name: str,
    sampled: bool,
    sample_size: int,
) -> float:
    relation_sql = access.relation_sql(table_meta.table_name, filtered=True)
    if sampled and access.count_rows(table_meta.table_name, filtered=True) > sample_size:
        relation_sql = access.sampled_relation_sql(
            table_meta.table_name,
            filtered=True,
            sample_size=sample_size,
            columns=[column_name],
        )
    qcol = quote_ident(column_name)
    row = access.fetchdf(
        f"SELECT COUNT(*) AS row_count, COUNT(DISTINCT {qcol}) AS distinct_count "
        f"FROM ({relation_sql}) rel WHERE {qcol} IS NOT NULL"
    )
    if row.empty:
        return 0.0
    row_count = int(row.iloc[0].get("row_count", 0) or 0)
    distinct_count = int(row.iloc[0].get("distinct_count", 0) or 0)
    if row_count <= 0:
        return 0.0
    return float(distinct_count / row_count)


def _overlap_ratio(
    *,
    access: Phase1DataAccess,
    child_meta: TableMetadata,
    child_column: str,
    parent_meta: TableMetadata,
    parent_column: str,
    sampled: bool,
    sample_size: int,
) -> float:
    child_relation = access.relation_sql(child_meta.table_name, filtered=True, columns=[child_column])
    if sampled and access.count_rows(child_meta.table_name, filtered=True) > sample_size:
        child_relation = access.sampled_relation_sql(
            child_meta.table_name,
            filtered=True,
            sample_size=sample_size,
            columns=[child_column],
        )
    parent_relation = access.relation_sql(parent_meta.table_name, filtered=True, columns=[parent_column])
    ccol = quote_ident(child_column)
    pcol = quote_ident(parent_column)
    row = access.fetchdf(
        f"WITH child_vals AS ("
        f"    SELECT DISTINCT {ccol} AS child_value FROM ({child_relation}) child WHERE {ccol} IS NOT NULL"
        f"), parent_vals AS ("
        f"    SELECT DISTINCT {pcol} AS parent_value FROM ({parent_relation}) parent WHERE {pcol} IS NOT NULL"
        f") "
        "SELECT "
        "  (SELECT COUNT(*) FROM child_vals) AS child_distinct, "
        "  (SELECT COUNT(*) FROM child_vals c JOIN parent_vals p ON c.child_value = p.parent_value) AS matched_distinct"
    )
    if row.empty:
        return 0.0
    child_distinct = int(row.iloc[0].get("child_distinct", 0) or 0)
    matched_distinct = int(row.iloc[0].get("matched_distinct", 0) or 0)
    if child_distinct <= 0:
        return 0.0
    return float(matched_distinct / child_distinct)


def _score_candidate_pair(
    *,
    access: Phase1DataAccess,
    child_meta: TableMetadata,
    child_column: str,
    parent_meta: TableMetadata,
    parent_column: str,
    exact_check: bool,
    source: str,
) -> ForeignKeyCandidate:
    overlap_ratio = _overlap_ratio(
        access=access,
        child_meta=child_meta,
        child_column=child_column,
        parent_meta=parent_meta,
        parent_column=parent_column,
        sampled=(not exact_check),
        sample_size=50_000,
    )
    child_unique_ratio = _column_unique_ratio(
        access=access,
        table_meta=child_meta,
        column_name=child_column,
        sampled=(not exact_check),
        sample_size=50_000,
    )
    null_ratio = _column_null_ratio(access, child_meta, child_column)
    child_tokens = normalize_identifier_tokens(child_column.replace("id", ""))
    parent_tokens = normalize_identifier_tokens(parent_meta.table_name) | normalize_identifier_tokens(parent_column)
    name_compatible = bool(child_tokens & parent_tokens)
    confidence = _confidence_label(
        overlap_ratio=overlap_ratio,
        name_compatible=name_compatible,
        uniqueness=child_unique_ratio,
        null_ratio=null_ratio,
    )
    return ForeignKeyCandidate(
        child_table=child_meta.table_name,
        child_column=child_column,
        parent_table=parent_meta.table_name,
        parent_column=parent_column,
        overlap_ratio=float(overlap_ratio),
        confidence=confidence,
        recommended_by_default=confidence in {"high", "medium"},
        source=source,
        name_compatible=name_compatible,
        child_unique_ratio=float(child_unique_ratio),
        null_ratio=float(null_ratio),
        exact_check=exact_check,
    )


def _confidence_label(
    *,
    overlap_ratio: float,
    name_compatible: bool,
    uniqueness: float,
    null_ratio: float,
) -> str:
    if overlap_ratio >= 0.98 and name_compatible and uniqueness <= 0.99 and null_ratio <= 0.80:
        return "high"
    if overlap_ratio >= 0.92 and name_compatible and uniqueness <= 0.995 and null_ratio <= 0.95:
        return "medium"
    return "low"
