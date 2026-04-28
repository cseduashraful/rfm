from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional, Any, Iterable
from collections import defaultdict
import csv
import datetime
import io
import re

import numpy as np
import pandas as pd
from task_history_queries import get_task_history_feature_hints


def _parse_float_like(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        return float(value)
    text = str(value).strip()
    if not text or text == "?":
        return None
    try:
        return float(text)
    except Exception:
        return None


def _signal_feature_match_summary(
    *,
    query_features: Dict[str, Any],
    candidate_features: Dict[str, Any],
    max_items: int = 3,
) -> tuple[float, List[str]]:
    feature_names = sorted(set(query_features.keys()) & set(candidate_features.keys()))
    if not feature_names:
        return -1.0, []

    score = 0.0
    matched: List[str] = []
    compared = 0
    for feature_name in feature_names:
        q_val = query_features.get(feature_name, "")
        c_val = candidate_features.get(feature_name, "")
        if q_val in ("", "?") or c_val in ("", "?"):
            continue
        compared += 1
        q_num = _parse_float_like(q_val)
        c_num = _parse_float_like(c_val)
        if q_num is not None and c_num is not None:
            denom = max(abs(q_num), abs(c_num), 1.0)
            closeness = max(0.0, 1.0 - abs(q_num - c_num) / denom)
            score += closeness
            if closeness >= 0.85 and len(matched) < max_items:
                matched.append(feature_name)
            continue
        if str(q_val) == str(c_val):
            score += 1.0
            if len(matched) < max_items:
                matched.append(feature_name)

    if compared == 0:
        return -1.0, []
    return score / compared, matched


@dataclass
class RelBenchGraphRAGStore:
    # ---------- ID maps ----------
    node_id_map: Dict[Tuple[str, Any], int] = field(default_factory=dict)
    rev_node_id: List[Tuple[str, Any]] = field(default_factory=list)

    edge_type_map: Dict[Tuple[str, str, str], int] = field(default_factory=dict)
    rev_edge_type: List[Tuple[str, str, str]] = field(default_factory=list)

    # node_id -> list[row_id] where row has no time_col
    static_info: Dict[int, List[int]] = field(default_factory=dict)

    # ---------- Base history index ----------
    hist_indptr: Optional[np.ndarray] = None
    hist_rowid: Optional[np.ndarray] = None
    hist_ts: Optional[np.ndarray] = None

    # ---------- Base adjacency index ----------
    indptr: Optional[np.ndarray] = None
    dst: Optional[np.ndarray] = None
    ts: Optional[np.ndarray] = None
    etype: Optional[np.ndarray] = None
    rowid: Optional[np.ndarray] = None

    # reverse adjacency
    rev_indptr: Optional[np.ndarray] = None
    rev_src: Optional[np.ndarray] = None
    rev_ts: Optional[np.ndarray] = None
    rev_etype: Optional[np.ndarray] = None
    rev_rowid: Optional[np.ndarray] = None

    # ---------- Row store ----------
    row_table: List[str] = field(default_factory=list)
    row_pk: List[Any] = field(default_factory=list)
    row_time: List[Any] = field(default_factory=list)
    node_rowid_map: Dict[int, int] = field(default_factory=dict)

    # ---------- Delta overlay ----------
    delta_hist: Dict[int, List[Tuple[Any, int]]] = field(default_factory=dict)
    delta_adj: Dict[int, List[Tuple[int, Any, int, int]]] = field(default_factory=dict)

    # ---------- Basic metadata ----------
    num_nodes: int = 0
    num_edges: int = 0
    num_rows: int = 0

    # ---------- Performance ----------
    # exact-query caches
    neighbor_cache: Dict[Tuple[int, int, int, Optional[Tuple[int, ...]]], List[Dict[str, Any]]] = field(default_factory=dict)
    multihop_cache: Dict[Tuple[int, int, int, int], List[Dict[str, Any]]] = field(default_factory=dict)
    history_cache: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = field(default_factory=dict)

    # node_id -> cached adjacency slice metadata
    _neighbor_slice_cache: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # table_name -> list[node_id]
    nodes_by_table: Dict[str, List[int]] = field(default_factory=dict)

    # time offset
    edge_time_offset: int = 0

    # tuneables
    # how many candidates per direction to inspect before final top-k merge
    neighbor_scan_factor: int = 8
    neighbor_min_scan: int = 64
    max_multihop_nodes_per_hop: int = 512
    max_similarity_scan_per_table: Optional[int] = 5000

    def add_delta_rows(self, table_name: str, df_chunk) -> None:
        raise NotImplementedError

    def clear_caches(self) -> None:
        self.neighbor_cache.clear()
        self.multihop_cache.clear()
        self.history_cache.clear()
        self._neighbor_slice_cache.clear()

    def _get_or_create_node_id(self, key: Tuple[str, Any]) -> int:
        nid = self.node_id_map.get(key)
        if nid is not None:
            return nid
        nid = len(self.rev_node_id)
        self.node_id_map[key] = nid
        self.rev_node_id.append(key)
        self.num_nodes = len(self.rev_node_id)
        return nid

    def _get_or_create_edge_type(self, key: Tuple[str, str, str]) -> int:
        eid = self.edge_type_map.get(key)
        if eid is not None:
            return eid
        eid = len(self.rev_edge_type)
        self.edge_type_map[key] = eid
        self.rev_edge_type.append(key)
        return eid

    @staticmethod
    def _cutoff_to_int(cutoff_time) -> int:
        if cutoff_time is None:
            return np.iinfo(np.int64).max
        if isinstance(cutoff_time, (int, np.integer)):
            return int(cutoff_time)
        if isinstance(cutoff_time, pd.Timestamp):
            return int(cutoff_time.value)
        if hasattr(cutoff_time, "value"):
            return int(cutoff_time.value)
        raise TypeError(f"Unsupported cutoff_time type: {type(cutoff_time)}")

    @staticmethod
    def _timestamp_to_ns(ts) -> int:
        if ts is None:
            return 0
        if isinstance(ts, (int, np.integer)):
            return int(ts)
        if isinstance(ts, pd.Timestamp):
            return int(ts.value)
        if hasattr(ts, "value"):
            return int(ts.value)
        raise TypeError(f"Unsupported timestamp type: {type(ts)}")

    def build_base(self, db, task=None) -> None:
        self.clear_caches()
        self.node_id_map.clear()
        self.rev_node_id.clear()
        self.edge_type_map.clear()
        self.rev_edge_type.clear()
        self.static_info.clear()
        self.row_table.clear()
        self.row_pk.clear()
        self.row_time.clear()
        self.node_rowid_map.clear()
        self.nodes_by_table.clear()

        row_id = 0
        self.table_row_offset = {}
        offset = 0

        # ---------- pass 1: create nodes + row metadata ----------
        for table_name, table in db.table_dict.items():
            df = table.df
            self.table_row_offset[table_name] = offset
            offset += len(df)

            pk_col = table.pkey_col
            time_col = table.time_col
            col_idx = {col: idx for idx, col in enumerate(df.columns)}
            pk_idx = col_idx.get(pk_col) if pk_col is not None else None
            time_idx = col_idx[time_col] if time_col is not None else None

            row_table_append = self.row_table.append
            row_pk_append = self.row_pk.append
            row_time_append = self.row_time.append
            get_or_create_node_id = self._get_or_create_node_id

            for local_row_idx, row in enumerate(df.itertuples(index=False, name=None)):
                pk_val = row[pk_idx] if pk_idx is not None else f"__row_{local_row_idx}"
                node_key = (table_name, pk_val)
                nid = get_or_create_node_id(node_key)
                self.nodes_by_table.setdefault(table_name, []).append(nid)

                row_table_append(table_name)
                row_pk_append(pk_val)
                row_time_append(row[time_idx] if time_idx is not None else None)
                self.node_rowid_map[nid] = row_id
                row_id += 1

        self.num_rows = row_id

        # ---------- pass 2: build edges ----------
        src_list = []
        dst_list = []
        ts_list = []
        etype_list = []
        rowid_list = []

        row_id = 0
        node_id_map = self.node_id_map

        for table_name, table in db.table_dict.items():
            df = table.df
            pk_col = table.pkey_col
            time_col = table.time_col
            fks = table.fkey_col_to_pkey_table

            col_idx = {col: idx for idx, col in enumerate(df.columns)}
            pk_idx = col_idx.get(pk_col) if pk_col is not None else None
            time_idx = col_idx[time_col] if time_col is not None else None

            fk_info = []
            for fk_col, ref_table in fks.items():
                if fk_col not in col_idx:
                    continue
                fk_info.append(
                    (
                        col_idx[fk_col],
                        ref_table,
                        self._get_or_create_edge_type((table_name, fk_col, ref_table)),
                    )
                )

            src_append = src_list.append
            dst_append = dst_list.append
            ts_append = ts_list.append
            etype_append = etype_list.append
            rowid_append = rowid_list.append

            for local_row_idx, row in enumerate(df.itertuples(index=False, name=None)):
                pk_val = row[pk_idx] if pk_idx is not None else f"__row_{local_row_idx}"
                src_node = node_id_map[(table_name, pk_val)]

                if time_idx is not None:
                    tsv = row[time_idx]
                    ts_ns = 0 if tsv is None else int(tsv.value)
                else:
                    ts_ns = 0

                for fk_idx, ref_table, et in fk_info:
                    ref_val = row[fk_idx]
                    if ref_val is None:
                        continue
                    dst_node = node_id_map.get((ref_table, ref_val))
                    if dst_node is None:
                        continue

                    src_append(src_node)
                    dst_append(dst_node)
                    ts_append(ts_ns)
                    etype_append(et)
                    rowid_append(row_id)

                row_id += 1

        self.src = np.asarray(src_list, dtype=np.int32)
        self.dst = np.asarray(dst_list, dtype=np.int32)
        ts_arr = np.asarray(ts_list, dtype=np.int64)
        self.edge_time_offset = int(ts_arr.min()) if len(ts_arr) > 0 else 0
        if len(ts_arr) > 0:
            ts_arr = ts_arr - self.edge_time_offset
        self.ts = ts_arr
        self.etype = np.asarray(etype_list, dtype=np.int16)
        self.rowid = np.asarray(rowid_list, dtype=np.int32)

        self.num_edges = len(self.src)
        self._build_csr_from_edges()
        self._build_history_index(db, task)

    def _build_csr_from_edges(self) -> None:
        if self.num_edges == 0:
            self.indptr = np.zeros(self.num_nodes + 1, dtype=np.int64)
            self.rev_indptr = np.zeros(self.num_nodes + 1, dtype=np.int64)
            self.rev_src = np.zeros(0, dtype=np.int32)
            self.rev_dst = np.zeros(0, dtype=np.int32)
            self.rev_ts = np.zeros(0, dtype=np.int64)
            self.rev_etype = np.zeros(0, dtype=np.int16)
            self.rev_rowid = np.zeros(0, dtype=np.int32)
            return

        # forward: sort by (src, ts)
        order = np.lexsort((self.ts, self.src))
        self.src = self.src[order]
        self.dst = self.dst[order]
        self.ts = self.ts[order]
        self.etype = self.etype[order]
        self.rowid = self.rowid[order]

        counts = np.bincount(self.src, minlength=self.num_nodes)
        self.indptr = np.empty(self.num_nodes + 1, dtype=np.int64)
        self.indptr[0] = 0
        np.cumsum(counts, out=self.indptr[1:])

        # reverse: sort by (dst, ts)
        rev_order = np.lexsort((self.ts, self.dst))
        self.rev_dst = self.dst[rev_order]
        self.rev_src = self.src[rev_order]
        self.rev_ts = self.ts[rev_order]
        self.rev_etype = self.etype[rev_order]
        self.rev_rowid = self.rowid[rev_order]

        rev_counts = np.bincount(self.rev_dst, minlength=self.num_nodes)
        self.rev_indptr = np.empty(self.num_nodes + 1, dtype=np.int64)
        self.rev_indptr[0] = 0
        np.cumsum(rev_counts, out=self.rev_indptr[1:])

    def _build_history_index(self, db, task) -> None:
        if task is None:
            self.hist_indptr = np.zeros(self.num_nodes + 1, dtype=np.int64)
            self.hist_rowid = np.zeros(0, dtype=np.int32)
            self.hist_ts = np.zeros(0, dtype=np.int64)
            return

        entity_col = task.entity_col
        entity_table = task.entity_table

        src_nodes = []
        row_ids = []
        ts_list = []

        row_id = 0
        for table_name, table in db.table_dict.items():
            df = table.df

            if entity_col not in df.columns:
                row_id += len(df)
                continue

            time_col = table.time_col
            col_idx = {col: idx for idx, col in enumerate(df.columns)}
            ent_idx = col_idx[entity_col]
            time_idx = col_idx[time_col] if time_col is not None else None

            for row in df.itertuples(index=False, name=None):
                ent_val = row[ent_idx]
                node = self.node_id_map.get((entity_table, ent_val))
                if node is None:
                    row_id += 1
                    continue

                if time_idx is None:
                    self.static_info.setdefault(node, []).append(row_id)
                else:
                    tsv = row[time_idx]
                    if tsv is None:
                        row_id += 1
                        continue
                    src_nodes.append(node)
                    row_ids.append(row_id)
                    ts_list.append(int(tsv.value))

                row_id += 1

        if len(src_nodes) == 0:
            self.hist_indptr = np.zeros(self.num_nodes + 1, dtype=np.int64)
            self.hist_rowid = np.zeros(0, dtype=np.int32)
            self.hist_ts = np.zeros(0, dtype=np.int64)
            return

        src_nodes = np.asarray(src_nodes, dtype=np.int32)
        row_ids = np.asarray(row_ids, dtype=np.int32)
        ts_arr = np.asarray(ts_list, dtype=np.int64)

        order = np.lexsort((ts_arr, src_nodes))
        src_nodes = src_nodes[order]
        row_ids = row_ids[order]
        ts_arr = ts_arr[order]

        counts = np.bincount(src_nodes, minlength=self.num_nodes)
        self.hist_indptr = np.empty(self.num_nodes + 1, dtype=np.int64)
        self.hist_indptr[0] = 0
        np.cumsum(counts, out=self.hist_indptr[1:])

        self.hist_rowid = row_ids
        self.hist_ts = ts_arr

    def get_static_info(self, node_id: int) -> List[int]:
        return self.static_info.get(node_id, [])

    def decode_row(self, row_id: int) -> Dict[str, Any]:
        return {
            "table": self.row_table[row_id],
            "pk": self.row_pk[row_id],
            "time": self.row_time[row_id],
        }

    def debug_neighbors(self, node_id: int, limit: int = 10):
        start = self.indptr[node_id]
        end = self.indptr[node_id + 1]
        out = []
        for i in range(start, min(end, start + limit)):
            out.append(
                {
                    "src": int(self.src[i]),
                    "dst": int(self.dst[i]),
                    "ts": int(self.ts[i]) + self.edge_time_offset,
                    "etype": int(self.etype[i]),
                    "rowid": int(self.rowid[i]),
                }
            )
        return out

    def debug_history(self, node_id: int, limit: int = 10):
        start = self.hist_indptr[node_id]
        end = self.hist_indptr[node_id + 1]
        out = []
        for i in range(start, min(end, start + limit)):
            out.append(
                {
                    "rowid": int(self.hist_rowid[i]),
                    "ts": int(self.hist_ts[i]),
                }
            )
        return out

    def get_history_before(self, node_id: int, cutoff_time, k: int):
        cutoff_abs = self._cutoff_to_int(cutoff_time)
        cache_key = (node_id, cutoff_abs, k)
        cached = self.history_cache.get(cache_key)
        if cached is not None:
            return [dict(x) for x in cached]

        start = self.hist_indptr[node_id]
        end = self.hist_indptr[node_id + 1]
        if start == end:
            return []

        ts_slice = self.hist_ts[start:end]
        pos = np.searchsorted(ts_slice, cutoff_abs, side="left")
        if pos == 0:
            return []

        left = max(0, pos - k)
        idx = np.arange(pos - 1, left - 1, -1, dtype=np.int64)
        rowids = self.hist_rowid[start:end][idx]
        tss = ts_slice[idx]

        results = [
            {"rowid": int(rid), "ts": int(ts)}
            for rid, ts in zip(rowids, tss)
        ]
        self.history_cache[cache_key] = [dict(x) for x in results]
        return results

    def _get_node_neighbor_slices(self, node_id: int) -> Dict[str, Any]:
        cached = self._neighbor_slice_cache.get(node_id)
        if cached is not None:
            return cached

        f_start = int(self.indptr[node_id])
        f_end = int(self.indptr[node_id + 1])

        r_start = int(self.rev_indptr[node_id])
        r_end = int(self.rev_indptr[node_id + 1])

        cached = {
            "f_dst": self.dst[f_start:f_end],
            "f_ts": self.ts[f_start:f_end],
            "f_etype": self.etype[f_start:f_end],
            "f_rowid": self.rowid[f_start:f_end],
            "r_dst": self.rev_src[r_start:r_end],
            "r_ts": self.rev_ts[r_start:r_end],
            "r_etype": self.rev_etype[r_start:r_end],
            "r_rowid": self.rev_rowid[r_start:r_end],
        }
        self._neighbor_slice_cache[node_id] = cached
        return cached

    def _collect_direction_candidates(
        self,
        dst_arr: np.ndarray,
        ts_arr: np.ndarray,
        etype_arr: np.ndarray,
        rowid_arr: np.ndarray,
        cutoff_rel: int,
        k: int,
        etype_filter_set: Optional[set],
    ):
        if len(ts_arr) == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int16), np.empty(0, dtype=np.int32)

        pos = int(np.searchsorted(ts_arr, cutoff_rel, side="right"))
        if pos == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int16), np.empty(0, dtype=np.int32)

        scan_n = min(pos, max(self.neighbor_min_scan, self.neighbor_scan_factor * max(k, 1)))
        sl = slice(pos - scan_n, pos)

        cand_dst = dst_arr[sl]
        cand_ts = ts_arr[sl]
        cand_etype = etype_arr[sl]
        cand_rowid = rowid_arr[sl]

        if etype_filter_set is not None:
            mask = np.isin(cand_etype, np.fromiter(etype_filter_set, dtype=np.int16))
            cand_dst = cand_dst[mask]
            cand_ts = cand_ts[mask]
            cand_etype = cand_etype[mask]
            cand_rowid = cand_rowid[mask]

        if len(cand_ts) == 0:
            return cand_dst, cand_ts, cand_etype, cand_rowid

        # newest first
        order = np.argsort(cand_ts)[::-1]
        return cand_dst[order], cand_ts[order], cand_etype[order], cand_rowid[order]

    def get_neighbors_before(
        self,
        node_id: int,
        cutoff_time,
        k: int,
        etype_filter: Optional[List[int]] = None,
    ):
        cutoff_abs = self._cutoff_to_int(cutoff_time)
        filter_key = None if etype_filter is None else tuple(sorted(int(x) for x in etype_filter))
        cache_key = (node_id, cutoff_abs, k, filter_key)
        cached = self.neighbor_cache.get(cache_key)
        if cached is not None:
            return [dict(x) for x in cached]

        cutoff_rel = cutoff_abs - self.edge_time_offset
        if cutoff_rel < 0:
            cutoff_rel = 0

        etype_filter_set = None if etype_filter is None else set(int(x) for x in etype_filter)
        sl = self._get_node_neighbor_slices(node_id)

        f_dst, f_ts, f_etype, f_rowid = self._collect_direction_candidates(
            sl["f_dst"], sl["f_ts"], sl["f_etype"], sl["f_rowid"], cutoff_rel, k, etype_filter_set
        )
        r_dst, r_ts, r_etype, r_rowid = self._collect_direction_candidates(
            sl["r_dst"], sl["r_ts"], sl["r_etype"], sl["r_rowid"], cutoff_rel, k, etype_filter_set
        )

        if len(f_ts) == 0 and len(r_ts) == 0:
            self.neighbor_cache[cache_key] = []
            return []

        cand_dst = np.concatenate([f_dst, r_dst], axis=0)
        cand_ts = np.concatenate([f_ts, r_ts], axis=0)
        cand_etype = np.concatenate([f_etype, r_etype], axis=0)
        cand_rowid = np.concatenate([f_rowid, r_rowid], axis=0)

        if len(cand_ts) == 0:
            self.neighbor_cache[cache_key] = []
            return []

        order = np.argsort(cand_ts)[::-1]
        cand_dst = cand_dst[order]
        cand_ts = cand_ts[order]
        cand_etype = cand_etype[order]
        cand_rowid = cand_rowid[order]

        results = []
        seen = set()

        for dst_i, ts_i, et_i, rid_i in zip(cand_dst, cand_ts, cand_etype, cand_rowid):
            key = (int(dst_i), int(ts_i), int(et_i), int(rid_i))
            if key in seen:
                continue
            seen.add(key)

            results.append(
                {
                    "dst": int(dst_i),
                    "ts": int(ts_i) + self.edge_time_offset,
                    "etype": int(et_i),
                    "rowid": int(rid_i),
                }
            )
            if len(results) >= k:
                break

        self.neighbor_cache[cache_key] = [dict(x) for x in results]
        return results

    def get_multihop_neighbors_before(
        self,
        start_node: int,
        cutoff_time,
        num_hops: int,
        top_k: int,
    ):
        if num_hops <= 0 or top_k <= 0:
            return []

        cutoff_abs = self._cutoff_to_int(cutoff_time)
        cache_key = (start_node, cutoff_abs, num_hops, top_k)
        cached = self.multihop_cache.get(cache_key)
        if cached is not None:
            return [dict(x) for x in cached]

        results = []
        visited = {start_node}
        frontier = [(start_node, cutoff_abs)]

        for hop in range(1, num_hops + 1):
            next_frontier = []
            hop_results = []

            for node_id, node_cutoff in frontier:
                nbrs = self.get_neighbors_before(node_id, node_cutoff, top_k)
                for n in nbrs:
                    hop_results.append({**n, "hop": hop})

            if not hop_results:
                break

            # keep newest first, and avoid frontier blow-up
            hop_results.sort(key=lambda x: x["ts"], reverse=True)

            limited_hop_results = hop_results[: self.max_multihop_nodes_per_hop]
            results.extend(limited_hop_results)

            for n in limited_hop_results:
                dst = n["dst"]
                if dst in visited:
                    continue
                visited.add(dst)
                next_frontier.append((dst, int(n["ts"])))

            if not next_frontier:
                break

            frontier = next_frontier

        self.multihop_cache[cache_key] = [dict(x) for x in results]
        return results

    def build_query_context(
        self,
        node_id: int,
        cutoff_time,
        k_hist: int = 10,
        k_nbr: int = 10,
        num_hops: int = 1,
        logger=None,
    ):
        hist = self.get_history_before(node_id, cutoff_time, k_hist)

        if logger:
            logger.info(
                "History for node_id=%d at cutoff_time=%s: %s",
                node_id,
                cutoff_time,
                hist[:5],
            )

        hist_decoded = []
        neighbor_rows = []
        seen_neighbor_keys = set()

        def append_neighbor_rows(source_row, neighbors):
            for n in neighbors:
                neighbor_key = (n["dst"], n["ts"], n["rowid"], n["hop"])
                if neighbor_key in seen_neighbor_keys:
                    continue
                seen_neighbor_keys.add(neighbor_key)
                neighbor_rows.append(
                    {
                        "src_row": source_row,
                        "neighbor_entity": self.rev_node_id[n["dst"]],
                        "etype": self.rev_edge_type[n["etype"]],
                        "ts": n["ts"],
                        "rowid": n["rowid"],
                        "hop": n["hop"],
                    }
                )

        # decode history rows once
        for h in hist:
            row = self.decode_row(h["rowid"])
            hist_decoded.append(
                {
                    **row,
                    "ts": h["ts"],
                    "rowid": h["rowid"],
                }
            )

        # neighbors for history rows
        for h in hist_decoded:
            row_node = self.node_id_map.get((h["table"], h["pk"]))
            if row_node is None:
                continue

            nbrs = self.get_multihop_neighbors_before(
                row_node,
                h["ts"],
                num_hops=num_hops,
                top_k=k_nbr,
            )
            append_neighbor_rows(
                {"table": h["table"], "pk": h["pk"], "time": h["ts"]},
                nbrs,
            )

        # neighbors for query node
        query_neighbors = self.get_multihop_neighbors_before(
            node_id,
            cutoff_time,
            num_hops=num_hops,
            top_k=k_nbr,
        )
        query_src = {
            "table": self.rev_node_id[node_id][0],
            "pk": self.rev_node_id[node_id][1],
            "time": cutoff_time,
        }
        append_neighbor_rows(query_src, query_neighbors)

        static = [
            {**self.decode_row(rid), "rowid": rid}
            for rid in self.get_static_info(node_id)
        ]

        return {
            "entity": self.rev_node_id[node_id],
            "static": static,
            "history": hist_decoded,
            "neighbors": neighbor_rows,
        }

    def serialize_context(self, ctx) -> str:
        lines = []
        table, pk = ctx["entity"]
        lines.append(f"Entity: {table}({pk})")

        if ctx["static"]:
            lines.append("\nStatic Info:")
            for r in ctx["static"]:
                lines.append(f"- {r['table']}({r['pk']})")

        if ctx["history"]:
            lines.append("\nHistory:")
            for r in ctx["history"]:
                lines.append(f"- {r['table']}({r['pk']}) at {r['ts']}")

        if ctx["neighbors"]:
            lines.append("\nNeighbors:")
            for n in ctx["neighbors"]:
                etype = n["etype"]
                nbr = n["neighbor_entity"]
                lines.append(f"- via {etype} → {nbr} at {n['ts']}")

        return "\n".join(lines)


class RowFeatureExtractor:
    """
    Fast row materializer with caching.
    """
    def __init__(self, db, store):
        self.tables = {}
        self.store = store
        self.row_cache: Dict[int, Dict[str, Any]] = {}

        for name, table in db.table_dict.items():
            df = table.df
            self.tables[name] = {
                "columns": list(df.columns),
                "data": {col: df[col].values for col in df.columns},
            }

    def get_row(self, rowid: int):
        cached = self.row_cache.get(rowid)
        if cached is not None:
            return cached

        meta = self.store.decode_row(rowid)
        table_name = meta["table"]
        offset = self.store.table_row_offset[table_name]
        local_id = rowid - offset
        table = self.tables[table_name]

        row = {
            col: table["data"][col][local_id]
            for col in table["columns"]
        }
        self.row_cache[rowid] = row
        return row

    def get_many_rows(self, rowids: Iterable[int]) -> Dict[int, Dict[str, Any]]:
        out = {}
        for rid in rowids:
            out[int(rid)] = self.get_row(int(rid))
        return out


def normalize_time(ts):
    if ts is None:
        return "static"
    if isinstance(ts, pd.Timestamp):
        return str(ts.date())
    if isinstance(ts, datetime.datetime):
        return str(ts.date())
    return str(datetime.datetime.utcfromtimestamp(ts / 1e9).date())


def _is_informative_value(value):
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    return True


def _select_display_columns(row):
    priority_columns = []
    fallback_columns = []
    for key, value in row.items():
        if not _is_informative_value(value):
            continue
        key_lower = key.lower()
        if key_lower in {"id", "rowid"}:
            continue
        if "date" in key_lower or "time" in key_lower:
            priority_columns.append(key)
        elif key_lower.endswith("id"):
            priority_columns.append(key)
        elif any(
            token in key_lower
            for token in ["name", "title", "type", "status", "position", "point", "score", "value", "label"]
        ):
            priority_columns.append(key)
        else:
            fallback_columns.append(key)

    selected_columns = []
    for key in priority_columns + fallback_columns:
        if key not in selected_columns:
            selected_columns.append(key)
        if len(selected_columns) >= 6:
            break
    return selected_columns


def format_row(table_name, row, ts):
    time_str = normalize_time(ts)
    selected_columns = _select_display_columns(row)
    features = [f"{k}={row.get(k)}" for k in selected_columns]
    return f"[{time_str}] {table_name}: " + ", ".join(features)


def format_task_history_row(task_name, row, time_col):
    ts = row.get(time_col)
    time_str = normalize_time(ts)
    features = [f"{k}={v}" for k, v in row.items()]
    return f"[{time_str}] {task_name}: " + ", ".join(features)


def format_task_query_row(task_name, row, time_col, output_col):
    ts = row.get(time_col)
    time_str = normalize_time(ts)
    features = [f"{k}={v}" for k, v in row.items() if k != output_col]
    return f"[{time_str}] {task_name}: " + ", ".join(features)


def _value_signature(value):
    if not _is_informative_value(value):
        return None
    if isinstance(value, pd.Timestamp):
        return str(value)
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), 3)
    return str(value)


def _similarity_score(query_row, candidate_row):
    score = 0.0
    shared = 0
    query_keys = set(query_row.keys()) & set(candidate_row.keys())
    for key in query_keys:
        key_lower = key.lower()
        if key_lower in {"id", "rowid"} or key_lower.endswith("id"):
            continue
        q_val = _value_signature(query_row[key])
        c_val = _value_signature(candidate_row[key])
        if q_val is None or c_val is None:
            continue
        shared += 1
        if isinstance(q_val, str) and isinstance(c_val, str):
            if q_val == c_val:
                score += 2.0
        elif isinstance(q_val, (int, float)) and isinstance(c_val, (int, float)):
            denom = max(abs(float(q_val)), abs(float(c_val)), 1.0)
            score += max(0.0, 1.0 - abs(float(q_val) - float(c_val)) / denom)
        elif q_val == c_val:
            score += 1.0
    if shared == 0:
        return -1.0
    return score


def build_similar_entity_context(store, extractor, node_id, cutoff_time, top_k, k_hist):
    entity_table, _ = store.rev_node_id[node_id]
    query_rowid = store.node_rowid_map.get(node_id)
    if query_rowid is None:
        return []

    query_row = extractor.get_row(query_rowid)

    candidate_nodes = store.nodes_by_table.get(entity_table, [])
    if store.max_similarity_scan_per_table is not None and len(candidate_nodes) > store.max_similarity_scan_per_table:
        candidate_nodes = candidate_nodes[: store.max_similarity_scan_per_table]

    scored = []
    for other_node_id in candidate_nodes:
        if other_node_id == node_id:
            continue
        other_rowid = store.node_rowid_map.get(other_node_id)
        if other_rowid is None:
            continue
        other_row = extractor.get_row(other_rowid)
        score = _similarity_score(query_row, other_row)
        if score <= 0:
            continue
        scored.append((score, other_node_id, other_rowid, other_row))

    scored.sort(key=lambda item: (-item[0], store.rev_node_id[item[1]][1]))

    similar_items = []
    for score, other_node_id, _, other_row in scored[:top_k]:
        history_rows = []
        for history_item in store.get_history_before(other_node_id, cutoff_time, k_hist):
            history_row = extractor.get_row(history_item["rowid"])
            history_rows.append(
                format_row(
                    store.row_table[history_item["rowid"]],
                    history_row,
                    history_item["ts"],
                )
            )
        similar_items.append(
            {
                "entity": f"{entity_table}({store.rev_node_id[other_node_id][1]})",
                "score": round(score, 3),
                "static": format_row(entity_table, other_row, None),
                "history": history_rows,
            }
        )
    return similar_items


def build_semantic_context(
    store,
    extractor,
    node_id,
    cutoff_time,
    k_hist,
    top_k,
    num_hops,
    logger=None,
    include_semantic_retrieval=False,
):
    ctx = store.build_query_context(
        node_id,
        cutoff_time,
        k_hist=k_hist,
        k_nbr=top_k,
        num_hops=num_hops,
        logger=logger,
    )

    needed_rowids = set()
    for h in ctx["static"]:
        needed_rowids.add(h["rowid"])
    for h in ctx["history"]:
        needed_rowids.add(h["rowid"])
    for n in ctx["neighbors"]:
        needed_rowids.add(n["rowid"])

    row_map = extractor.get_many_rows(needed_rowids)

    static_info = []
    for h in ctx["static"]:
        row = row_map[h["rowid"]]
        static_info.append(
            {
                "table": h["table"],
                "time": "static",
                "text": format_row(h["table"], row, None),
            }
        )

    semantic_history = []
    for h in ctx["history"]:
        row = row_map[h["rowid"]]
        semantic_history.append(
            {
                "table": h["table"],
                "time": normalize_time(h["ts"]),
                "text": format_row(h["table"], row, h["ts"]),
            }
        )

    semantic_neighbors = []
    seen = set()
    for n in ctx["neighbors"]:
        key = (n["neighbor_entity"], n["ts"], n["rowid"], n["hop"])
        if key in seen:
            continue
        seen.add(key)

        edge_row_meta = store.decode_row(n["rowid"])
        nbr_row = row_map[n["rowid"]]

        semantic_neighbors.append(
            {
                "table": edge_row_meta["table"],
                "text": (
                    f"{format_row(edge_row_meta['table'], nbr_row, n['ts'])} "
                    f"-> {n['neighbor_entity']} via {n['etype']}"
                ),
                "source_key": (edge_row_meta["table"], edge_row_meta["pk"], int(n["ts"])),
                "source_text": format_row(edge_row_meta["table"], nbr_row, n["ts"]),
                "neighbor_entity": n["neighbor_entity"],
                "etype": n["etype"],
                "hop": n["hop"],
            }
        )

    return {
        "static": static_info,
        "history": semantic_history,
        "neighbors": semantic_neighbors,
        "similar": (
            build_similar_entity_context(
                store,
                extractor,
                node_id,
                cutoff_time,
                top_k=top_k,
                k_hist=k_hist,
            )
            if include_semantic_retrieval
            else []
        ),
    }


def format_prompt(entity_name, entity_id, context, task_desc):
    lines = []
    lines.append(f"Task: {task_desc}")
    lines.append("Return ONLY a number.")
    lines.append("")
    lines.append(f"Entity: {entity_name}({entity_id})")
    lines.append("")

    grouped = defaultdict(list)

    if context["static"]:
        lines.append("\nStatic:")
        for r in context["static"]:
            lines.append(f"- {r['text']}")

    for h in context["history"]:
        grouped[h["table"]].append(h)

    lines.append("History (most recent first):")
    for table, items in grouped.items():
        lines.append(f"\n{table}:")
        for i, item in enumerate(items[:10], 1):
            lines.append(f"{i}. {item['text']}")

    if context["neighbors"]:
        lines.append("\nNeighbors:")
        for i, n in enumerate(context["neighbors"][:10], 1):
            lines.append(f"{i}. [hop={n['hop']}] {n['text']}")

    lines.append("\nQuestion:")
    lines.append(f"Predict {task_desc}.")
    lines.append("")
    lines.append("Answer:")

    return "\n".join(lines)


def summarize_neighbors_by_hop(neighbors):
    summaries = []
    grouped = {}
    for neighbor in neighbors:
        grouped.setdefault(neighbor["hop"], []).append(neighbor)

    for hop in sorted(grouped):
        items = grouped[hop]
        table_counts = {}
        for item in items:
            table_name = item.get("table", "unknown")
            table_counts[table_name] = table_counts.get(table_name, 0) + 1

        top_tables = sorted(
            table_counts.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )[:3]
        table_text = ", ".join(f"{name} x{count}" for name, count in top_tables)
        summaries.append(
            {
                "hop": hop,
                "count": len(items),
                "tables": table_text,
            }
        )
    return summaries


def render_neighbor_graph(neighbors):
    lines = []
    grouped_by_hop = {}
    for neighbor in neighbors:
        grouped_by_hop.setdefault(neighbor["hop"], []).append(neighbor)

    for hop in sorted(grouped_by_hop):
        lines.append(f"Hop {hop}:")
        source_groups = {}
        source_order = []
        for neighbor in grouped_by_hop[hop]:
            source_key = neighbor["source_key"]
            if source_key not in source_groups:
                source_groups[source_key] = {
                    "source_text": neighbor["source_text"],
                    "edges": [],
                }
                source_order.append(source_key)
            source_groups[source_key]["edges"].append(
                f"{neighbor['etype'][1]} -> {neighbor['neighbor_entity'][0]}({neighbor['neighbor_entity'][1]})"
            )

        for source_key in source_order:
            source_group = source_groups[source_key]
            lines.append(f"- {source_group['source_text']}")
            for edge_text in source_group["edges"]:
                lines.append(f"  -> {edge_text}")

    return lines


def build_history_neighbor_context(
    store,
    extractor,
    resource,
    entity_value,
    history_rows,
    top_k,
    num_hops,
):
    node_key = (resource.entity_table, entity_value)
    node_id = store.node_id_map.get(node_key)

    static_info = []
    if node_id is not None:
        static_rowids = store.get_static_info(node_id)
        row_map = extractor.get_many_rows(static_rowids)
        for rid in static_rowids:
            row = row_map[rid]
            meta = store.decode_row(rid)
            static_info.append(
                {
                    "table": meta["table"],
                    "time": "static",
                    "text": format_row(meta["table"], row, None),
                }
            )

    semantic_history = []
    semantic_neighbors = []
    seen_neighbors = set()
    self_node_key = (resource.entity_table, entity_value)

    for history_row in history_rows:
        history_timestamp = pd.Timestamp(history_row[resource.time_col])
        semantic_history.append(
            {
                "table": resource.task,
                "time": normalize_time(history_timestamp),
                "text": format_task_history_row(resource.task, history_row, resource.time_col),
            }
        )

        history_node_id = store.node_id_map.get((resource.entity_table, history_row[resource.entity_col]))
        if history_node_id is None:
            continue

        neighbors = store.get_multihop_neighbors_before(
            history_node_id,
            history_timestamp,
            num_hops=num_hops,
            top_k=top_k,
        )
        neighbor_rowids = [n["rowid"] for n in neighbors]
        row_map = extractor.get_many_rows(neighbor_rowids)

        for neighbor in neighbors:
            neighbor_key = (neighbor["dst"], neighbor["ts"], neighbor["rowid"], neighbor["hop"])
            if neighbor_key in seen_neighbors:
                continue

            neighbor_entity = store.rev_node_id[neighbor["dst"]]
            if neighbor_entity == self_node_key:
                continue

            neighbor_row = row_map[neighbor["rowid"]]
            edge_row_meta = store.decode_row(neighbor["rowid"])
            if edge_row_meta["table"] == resource.entity_table and edge_row_meta["pk"] == entity_value:
                continue

            seen_neighbors.add(neighbor_key)
            semantic_neighbors.append(
                {
                    "table": edge_row_meta["table"],
                    "text": (
                        f"{format_row(edge_row_meta['table'], neighbor_row, neighbor['ts'])} "
                        f"-> {neighbor_entity[0]}({neighbor_entity[1]})"
                    ),
                    "source_key": (
                        edge_row_meta["table"],
                        edge_row_meta["pk"],
                        int(neighbor["ts"]),
                    ),
                    "source_text": format_row(
                        edge_row_meta["table"],
                        neighbor_row,
                        neighbor["ts"],
                    ),
                    "neighbor_entity": neighbor_entity,
                    "etype": neighbor["etype"],
                    "hop": neighbor["hop"],
                }
            )

    return {
        "static": static_info,
        "history": semantic_history,
        "neighbors": semantic_neighbors,
    }





def build_zero_shot_prompt(
    store,
    extractor,
    resource,
    query_row,
    history_rows,
    top_k,
    num_hops,
    include_hop_aggregation=True,
    include_semantic_retrieval=False,
    use_dfs=False,
    dfs_context_builder=None,
    context_workers=1,
    precomputed_entry_dfs_summaries=None,
    precomputed_entry_dfs_feature_dicts=None,
    recent_context_k=2,
    include_neighbors=True,
    include_dfs_summary=True,
    include_dfs_table=False,
    other_neighbor_entity_count=5,
    other_neighbor_history_count=3,
):
    table_only_mode = include_dfs_table and not include_neighbors and not include_dfs_summary

    def _build_table_signal_summary(
        entry_rows: List[Dict[str, Any]],
        entry_feature_dicts: List[Dict[str, Any]],
    ) -> List[str]:
        if not entry_rows or not entry_feature_dicts:
            return []

        self_items: List[tuple[Dict[str, Any], Dict[str, Any]]] = []
        other_items: List[tuple[Dict[str, Any], Dict[str, Any]]] = []
        query_item: tuple[Dict[str, Any], Dict[str, Any]] | None = None
        for idx, (row, row_features) in enumerate(zip(entry_rows, entry_feature_dicts)):
            is_query_row = idx == len(entry_rows) - 1
            if is_query_row:
                query_item = (row, row_features)
                continue
            scope = str(row.get("__example_scope", "self"))
            if scope == "other":
                other_items.append((row, row_features))
            else:
                self_items.append((row, row_features))

        lines: List[str] = []
        lines.append("Prompt Signal Summary:")
        if self_items:
            self_outputs = [
                _parse_float_like(row.get(resource.output_col))
                for row, _ in self_items
            ]
            self_outputs = [x for x in self_outputs if x is not None]
            recent_self_outputs = self_outputs[-3:]
            if recent_self_outputs:
                rendered = ", ".join(f"{val:g}" for val in recent_self_outputs)
                lines.append(
                    f"- Recent self outputs ({len(recent_self_outputs)} most recent): {rendered}."
                )
                if len(recent_self_outputs) >= 2:
                    delta = recent_self_outputs[-1] - recent_self_outputs[0]
                    if abs(delta) > 0:
                        direction = "upward" if delta > 0 else "downward"
                        lines.append(f"- Self trend over recent rows is {direction}.")
        else:
            lines.append("- No leakage-safe self history is available for this query.")

        if query_item is not None and other_items:
            query_row, query_features = query_item
            scored_neighbors: List[tuple[float, Dict[str, Any], List[str]]] = []
            for row, row_features in other_items:
                score, matched = _signal_feature_match_summary(
                    query_features=query_features,
                    candidate_features=row_features,
                )
                if score < 0:
                    continue
                scored_neighbors.append((score, row, matched))
            scored_neighbors.sort(
                key=lambda item: (
                    -item[0],
                    pd.Timestamp(item[1].get(resource.time_col)).value
                    if item[1].get(resource.time_col) is not None
                    else -1,
                )
            )
            top_neighbors = scored_neighbors[:3]
            if top_neighbors:
                rendered_neighbors: List[str] = []
                neighbor_outputs: List[float] = []
                for score, row, matched in top_neighbors:
                    out_val = _parse_float_like(row.get(resource.output_col))
                    if out_val is not None:
                        neighbor_outputs.append(out_val)
                    match_text = ", ".join(matched[:2]) if matched else "overall feature profile"
                    rendered_neighbors.append(
                        f"{row.get(resource.output_col)} at {row.get(resource.time_col)} ({match_text})"
                    )
                lines.append("- Closest peer rows to the query: " + "; ".join(rendered_neighbors) + ".")
                if neighbor_outputs:
                    lines.append(
                        f"- Closest peer output range: {min(neighbor_outputs):g} to {max(neighbor_outputs):g}."
                    )
        return lines

    def _visible_row_fields(row: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in row.items() if not str(k).startswith("__")}

    ordered_history_rows = [dict(row) for row in history_rows]
    entry_rows = ordered_history_rows + [dict(query_row)]
    entry_dfs_summaries = [[] for _ in entry_rows]
    entry_dfs_feature_dicts = [{} for _ in entry_rows]
    if use_dfs:
        if precomputed_entry_dfs_summaries is not None:
            entry_dfs_summaries = precomputed_entry_dfs_summaries
        elif dfs_context_builder is not None:
            entry_dfs_summaries = dfs_context_builder.summarize_rows(entry_rows)
        else:
            entry_dfs_summaries = [
                ["- DFS context requested but FastDFS context builder is unavailable."]
                for _ in entry_rows
            ]
        if include_dfs_table:
            if precomputed_entry_dfs_feature_dicts is not None:
                entry_dfs_feature_dicts = precomputed_entry_dfs_feature_dicts
            elif dfs_context_builder is not None:
                entry_dfs_feature_dicts = dfs_context_builder.feature_dicts_for_rows(entry_rows)

    entry_contexts = []
    if not table_only_mode:
        def _build_entry_context(row: Dict[str, Any], dfs_summary: list[str]) -> dict[str, Any] | None:
            entity_value = row[resource.entity_col]
            cutoff_time = pd.Timestamp(row[resource.time_col])
            node_id = store.node_id_map.get((resource.entity_table, entity_value))
            if node_id is None:
                return None
            context = build_semantic_context(
                store=store,
                extractor=extractor,
                node_id=node_id,
                cutoff_time=cutoff_time,
                k_hist=len(history_rows) if history_rows else top_k,
                top_k=top_k,
                num_hops=num_hops,
                include_semantic_retrieval=include_semantic_retrieval,
            )
            return {"row": row, "context": context, "dfs_summary": dfs_summary}

        max_workers = max(1, int(context_workers))
        if max_workers > 1 and len(entry_rows) > 1:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(entry_rows))) as pool:
                built_contexts = list(pool.map(_build_entry_context, entry_rows, entry_dfs_summaries))
            entry_contexts = [ctx for ctx in built_contexts if ctx is not None]
        else:
            for row, dfs_summary in zip(entry_rows, entry_dfs_summaries):
                built = _build_entry_context(row, dfs_summary)
                if built is not None:
                    entry_contexts.append(built)

    lines = []

    # =========================
    # 🔥 STRONG HEADER (CRITICAL)
    # =========================
    # lines.append(f"Task: Predict {resource.output_col} for the query node id")
    # lines.append("Return ONLY a number. No explanation.")
    # # lines.append("Follow the pattern in the examples exactly.")
    # # lines.append("Valid outputs are numeric (e.g., 10, 12.5).")
    # lines.append("The prompt contains first examples for the same entity - old to recent. and then recent interactions of the query entity from the database. Consider the old to recent pattern and recent activity and corresponing timestamps ")
    # lines.append("")
    # lines.append(f"Task: Predict {resource.output_col} for the query entity.")

    # lines.append("Output format:")
    # lines.append("- Return ONLY a single numeric value")
    # lines.append("- No explanation, no text, no symbols")

    # lines.append("Instructions:")
    # lines.append("- The examples show historical values for the SAME entity in chronological order (oldest → most recent)")
    # lines.append("- Learn how the value evolves over time based on recent activity")
    # lines.append("- Use recency: more recent interactions are more important")
    # lines.append("- Use trend: detect whether values are increasing, decreasing, or stable")
    # lines.append("- Predict the next value for the query timestamp")

    # lines.append("")

    lines.append("Task: Predict the next value of the target variable for the same entity.")
    lines.append("")
    if table_only_mode and True:
        lines.append("Guidelines:")
        lines.append("- Focus primarily on the most recent rows of the same entity.")
        lines.append("- Use earlier rows only to understand overall scale and typical behavior.")
        lines.append("- If self history is limited, use similar entities to estimate a reasonable range.")
        lines.append("- Do not overreact to a single outlier (self or neighbor).")
        lines.append("- If signals conflict, prioritize recent self behavior over long-term averages or neighbors.")
        lines.append("- Avoid defaulting to a generic mid-range value — base the prediction on actual patterns.")
        lines.append("- Allow both upward and downward changes if supported by evidence.")
        lines.append("- If the query row has feature values, compare them with recent self and neighbor rows.")
        lines.append("- Identify rows with similar feature patterns and use their outputs as guidance.")
        lines.append("- Prefer examples that are both recent and feature-similar.")
        lines.append("- Identify which features vary across examples and explain output differences.")
        lines.append("- Use similarity to adjust the prediction, not to copy values directly.")
        signal_summary_lines = _build_table_signal_summary(entry_rows, entry_dfs_feature_dicts)
        if signal_summary_lines:
            lines.append("")
            lines.extend(signal_summary_lines)
    elif table_only_mode and False:
        lines.append("Data Usage:")
        lines.append("- Use the DFS Feature Table below as the only context.")
        lines.append("- Self rows (same entity) are the PRIMARY signal when sufficient history exists.")
        lines.append("- Neighbor rows are SECONDARY, but become IMPORTANT when self history is sparse or unstable.")
        lines.append(
            f"- Use up to {int(other_neighbor_entity_count)} nearby same-type neighbor entities "
            f"and up to {int(other_neighbor_history_count)} most recent rows per neighbor."
        )
        lines.append("")
        lines.append("Core Principle:")
        lines.append("- Dynamically decide whether to trust SELF or NEIGHBORS more based on evidence strength.")
        lines.append("")
        lines.append("--------------------------------------------------")
        lines.append("Step 0: Assess Self Evidence Strength")
        lines.append("--------------------------------------------------")
        lines.append("Determine how reliable the self history is:")
        lines.append("")
        lines.append("- STRONG self evidence:")
        lines.append("  - 3 or more recent self rows, OR")
        lines.append("  - clear stable trend (increasing, decreasing, or consistent level)")
        lines.append("")
        lines.append("- WEAK self evidence:")
        lines.append("  - fewer than 3 self rows, OR")
        lines.append("  - no clear pattern")
        lines.append("")
        lines.append("--------------------------------------------------")
        lines.append("Step 1: Build Base Prediction")
        lines.append("--------------------------------------------------")
        lines.append("")
        lines.append("IF self evidence is STRONG:")
        lines.append("  - Use recent self rows to estimate level and trend")
        lines.append("  - Extrapolate conservatively")
        lines.append("")
        lines.append("IF self evidence is WEAK:")
        lines.append("  - Do NOT rely solely on the last self value")
        lines.append("  - Use neighbors to estimate a reasonable scale and behavior")
        lines.append("")
        lines.append("--------------------------------------------------")
        lines.append("Step 2: Use Neighbor Evidence (Carefully)")
        lines.append("--------------------------------------------------")
        lines.append("")
        lines.append(f"From up to {int(other_neighbor_entity_count)} similar entities:")
        lines.append("- Identify typical output range and behavior")
        lines.append("- Look for:")
        lines.append("  - common scale (e.g., low vs high values)")
        lines.append("  - whether outputs tend to drop, spike, or stay stable")
        lines.append("")
        lines.append("Rules:")
        lines.append("- Neighbors can ADJUST scale when self is weak")
        lines.append("- Neighbors can NOT override strong self trends")
        lines.append("- Do NOT copy neighbor values directly")
        lines.append("")
        lines.append("--------------------------------------------------")
        lines.append("Step 3: Detect Possible Regime Change")
        lines.append("--------------------------------------------------")
        lines.append("")
        lines.append("Consider a regime change if:")
        lines.append("- Self history is very short (1-2 points)")
        lines.append("- Neighbor values are consistently far from self value")
        lines.append("- Feature patterns resemble neighbors more than past self")
        lines.append("")
        lines.append("In such cases:")
        lines.append("- Allow larger deviation from last self value")
        lines.append("- Move prediction toward neighbor-informed scale")
        lines.append("")
        lines.append("--------------------------------------------------")
        lines.append("Step 4: Final Prediction Rules")
        lines.append("--------------------------------------------------")
        lines.append("")
        lines.append("- If strong self trend -> stay close to self trajectory")
        lines.append("- If weak self signal -> blend self + neighbor scale")
        lines.append("- Allow large change ONLY if supported by:")
        lines.append("  - weak self evidence AND")
        lines.append("  - consistent neighbor pattern")
        lines.append("")
        lines.append("--------------------------------------------------")
        lines.append("Constraints:")
        lines.append("--------------------------------------------------")
        lines.append("")
        lines.append("- Do NOT blindly copy the last value")
        lines.append("- Do NOT blindly copy neighbors")
        lines.append("- Prefer realistic scale over strict smoothness")
        lines.append("- Output must reflect the most plausible value given BOTH self and neighbors")
        lines.append("")
        lines.append("--------------------------------------------------")
        lines.append("Output:")
        lines.append("--------------------------------------------------")
        lines.append("")
        lines.append("Return a single numeric value only.")
        lines.append("No explanation, no text, no symbols.")
    else:
        lines.append("Guidelines:")
        lines.append("- Examples are ordered by time (oldest → newest)")
        lines.append("- The target value depends on recent behavior")
        lines.append("- Focus on the last few timestamps more than older ones")
        lines.append("- Identify the trend (increasing, decreasing, stable)")
        lines.append("- Extrapolate to predict the next value at the query time")

    lines.append("")

    # =========================
    # STATIC (UNCHANGED)
    # =========================
    query_node_id = store.node_id_map.get((resource.entity_table, query_row[resource.entity_col]))
    if query_node_id is not None:
        static_once = []
        static_rowids = store.get_static_info(query_node_id)
        static_rows = extractor.get_many_rows(static_rowids)

        for rid in static_rowids:
            row = static_rows[rid]
            meta = store.decode_row(rid)
            static_once.append(format_row(meta["table"], row, None))

        if static_once:
            lines.append("Static:")
            for text in static_once:
                lines.append(f"- {text}")

    # =========================
    # 🔥 KEEP ORIGINAL ICL STRUCTURE
    # =========================
    if not table_only_mode:
        for idx, item in enumerate(entry_contexts, 1):
            row = item["row"]
            context = item["context"]
            is_query = idx == len(entry_contexts)
            example_scope = str(row.get("__example_scope", "self" if not is_query else "query"))
            show_recent_context = idx > max(0, len(entry_contexts) - max(0, int(recent_context_k)))

            lines.append("")

            if is_query:
                lines.append(
                    f"Query: {format_task_query_row(resource.task, _visible_row_fields(row), resource.time_col, resource.output_col)}"
                )
            else:
                # if idx == 1:
                #     lines.append()
                lines.append(
                    f"Example {idx} [{example_scope}]: "
                    f"{format_task_query_row(resource.task, _visible_row_fields(row), resource.time_col, resource.output_col)}"
                )

            # ---- neighbors ----
            if include_neighbors and context["neighbors"] and show_recent_context:
                if include_hop_aggregation:
                    lines.append("Neighbor Summary:")
                    for summary in summarize_neighbors_by_hop(context["neighbors"]):
                        lines.append(
                            f"- hop {summary['hop']}: {summary['count']} neighbors"
                            + (f" ({summary['tables']})" if summary["tables"] else "")
                        )

                lines.append("Neighbors:")
                lines.extend(render_neighbor_graph(context["neighbors"]))

            if use_dfs and include_dfs_summary and show_recent_context:
                lines.append("DFS Summary:")
                dfs_summary_lines = item.get("dfs_summary", [])
                if dfs_summary_lines:
                    lines.extend(dfs_summary_lines)
                else:
                    lines.append("- No DFS-derived signals available for this timestamp.")

            # ---- similar ----
            if context.get("similar") and is_query:
                lines.append("Similar Entities:")
                for similar_item in context["similar"]:
                    lines.append(
                        f"- {similar_item['entity']} | similarity={similar_item['score']} | {similar_item['static']}"
                    )
                    for history_text in similar_item["history"]:
                        lines.append(f"  history: {history_text}")

            # ---- output ----
            if not is_query:
                lines.append(f"Output [{example_scope}]: {row[resource.output_col]}")

    if use_dfs and include_dfs_table:
        def _cell_text(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, float) and np.isnan(value):
                return ""
            return str(value)

        max_dfs_table_features = 24
        feature_hints = get_task_history_feature_hints(resource.dataset, resource.task) or {}
        hinted_tables = {
            str(table_name).lower(): int(weight)
            for table_name, weight in feature_hints.get("tables", {}).items()
        }
        hinted_columns = {
            (str(table_name).lower(), str(column_name).lower()): int(weight)
            for (table_name, column_name), weight in feature_hints.get("columns", {}).items()
        }

        def _normalized_tokens(text: str) -> set[str]:
            return {tok for tok in re.split(r"[^a-zA-Z0-9]+", text.lower()) if tok}

        output_tokens = _normalized_tokens(str(resource.output_col))

        def _feature_relevance_bonus(feature_name: str) -> int:
            score = 0
            lowered = feature_name.lower()

            # Reward lexical overlap with task output name.
            overlap = _normalized_tokens(lowered) & output_tokens
            score += 2 * len(overlap)

            # Reward table/column provenance from the task SQL.
            matches = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)", feature_name)
            seen_table_column = set()
            for table_name, column_name in matches:
                table_key = table_name.lower()
                column_key = column_name.lower()
                if table_key in hinted_tables:
                    score += hinted_tables[table_key]
                table_column_key = (table_key, column_key)
                if table_column_key in hinted_columns and table_column_key not in seen_table_column:
                    score += hinted_columns[table_column_key]
                    seen_table_column.add(table_column_key)

            # Light penalty for metadata-like columns unless explicitly hinted.
            metadata_tokens = {"year", "round", "statusid", "raceid", "number"}
            if any(token in lowered for token in metadata_tokens) and score == 0:
                score -= 2

            # Prefer statistical aggregates for table rendering.
            aggregate_bonus = [
                (".mean(", 4),
                (".std(", 4),
                (".sum(", 3),
                (".count(", 3),
                (".median(", 3),
                (".var(", 2),
                (".skew(", 2),
                (".quantile(", 2),
                (".percentile(", 2),
                (".num_unique(", 2),
            ]
            for token, bonus in aggregate_bonus:
                if token in lowered:
                    score += bonus
            return score

        def _is_excluded_extreme_feature(feature_name: str) -> bool:
            lowered = feature_name.lower()
            return ".max(" in lowered or ".min(" in lowered

        feature_scores: Dict[str, int] = {}
        for idx, (row, row_features) in enumerate(zip(entry_rows, entry_dfs_feature_dicts), 1):
            is_query_row = idx == len(entry_rows)
            if is_query_row:
                weight = 2
            elif str(row.get("__example_scope", "self")) == "self":
                weight = 3
            else:
                weight = 1
            for feature_name in row_features.keys():
                if _is_excluded_extreme_feature(feature_name):
                    continue
                feature_scores[feature_name] = feature_scores.get(feature_name, 0) + weight

        ranked_feature_names = sorted(
            feature_scores.keys(),
            key=lambda name: (
                -(feature_scores[name] + _feature_relevance_bonus(name)),
                -feature_scores[name],
                name,
            ),
        )
        feature_names = ranked_feature_names[:max_dfs_table_features]
        max_delta_features = 8

        def _parse_float_or_none(text: Any) -> float | None:
            if text is None:
                return None
            text_s = str(text).strip()
            if not text_s:
                return None
            try:
                return float(text_s)
            except Exception:
                return None

        def _select_delta_feature_names(candidates: list[str]) -> list[str]:
            priority_buckets = [".count(", ".mean(", ".std(", ".sum("]
            selected: list[str] = []
            seen = set()
            lowered_candidates = [(name, name.lower()) for name in candidates]

            for token in priority_buckets:
                for name, lowered in lowered_candidates:
                    if name in seen:
                        continue
                    if token in lowered:
                        selected.append(name)
                        seen.add(name)
                        if len(selected) >= max_delta_features:
                            return selected

            # Fill remaining slots with task-relevant candidates.
            for name in candidates:
                if name in seen:
                    continue
                selected.append(name)
                seen.add(name)
                if len(selected) >= max_delta_features:
                    break
            return selected

        delta_feature_names = _select_delta_feature_names(feature_names)
        delta_column_names = [f"delta::{name}" for name in delta_feature_names]

        def _render_dfs_section(
            section_title: str,
            rows_with_features: list[tuple[int, dict[str, Any], dict[str, str], bool]],
            section_scope: str,
        ) -> None:
            lines.append("")
            lines.append(section_title)
            if not rows_with_features:
                lines.append("- none")
                return

            table_stream = io.StringIO()
            writer = csv.writer(table_stream)
            header = [
                "entry_index",
                resource.entity_col,
                resource.time_col,
                "output",
            ] + feature_names + delta_column_names
            writer.writerow(header)

            prev_values_by_entity: Dict[str, Dict[str, float]] = {}
            for idx, row, row_features, is_query_row in rows_with_features:
                output_value = "?" if is_query_row else _cell_text(row.get(resource.output_col))
                entity_key = _cell_text(row.get(resource.entity_col))
                prev_values = prev_values_by_entity.get(entity_key, {})
                row_cells = [
                    str(idx),
                    _cell_text(row.get(resource.entity_col)),
                    _cell_text(row.get(resource.time_col)),
                    output_value,
                ]
                for feature_name in feature_names:
                    row_cells.append(row_features.get(feature_name, ""))
                current_values_for_entity: Dict[str, float] = dict(prev_values)
                for feature_name in delta_feature_names:
                    current_val = _parse_float_or_none(row_features.get(feature_name, ""))
                    previous_val = prev_values.get(feature_name)
                    if current_val is None or previous_val is None:
                        row_cells.append("")
                    else:
                        row_cells.append(_cell_text(current_val - previous_val))
                    if current_val is not None:
                        current_values_for_entity[feature_name] = current_val
                prev_values_by_entity[entity_key] = current_values_for_entity
                writer.writerow(row_cells)

            lines.append("```csv")
            lines.append(table_stream.getvalue().rstrip())
            lines.append("```")

            # Show MIN/MAX once per entity only for self section.
            if section_scope != "self":
                return

            minmax_candidates = sorted(
                {
                    name
                    for _, _, row_features, _ in rows_with_features
                    for name in row_features.keys()
                    if ".min(" in name.lower() or ".max(" in name.lower()
                },
                key=lambda name: (
                    -(feature_scores.get(name, 0) + _feature_relevance_bonus(name)),
                    -feature_scores.get(name, 0),
                    name,
                ),
            )[:8]

            if not minmax_candidates:
                return

            latest_row_by_entity: Dict[str, tuple[pd.Timestamp, dict[str, Any], dict[str, str]]] = {}
            for _, row, row_features, is_query_row in rows_with_features:
                if is_query_row:
                    continue
                entity_key = _cell_text(row.get(resource.entity_col))
                ts_raw = row.get(resource.time_col)
                ts = pd.Timestamp.min
                try:
                    ts = pd.Timestamp(ts_raw)
                except Exception:
                    pass
                existing = latest_row_by_entity.get(entity_key)
                if existing is None or ts >= existing[0]:
                    latest_row_by_entity[entity_key] = (ts, row, row_features)

            if not latest_row_by_entity:
                return

            def _minmax_base_key(feature_name: str) -> str:
                lowered = feature_name.lower()
                base = re.sub(r"\.min\(", ".agg(", lowered)
                base = re.sub(r"\.max\(", ".agg(", base)
                return base

            lines.append("Latest Min/Max (Self):")
            for entity_key in sorted(latest_row_by_entity.keys()):
                ts, row, row_features = latest_row_by_entity[entity_key]
                minmax_by_base: Dict[str, Dict[str, str]] = {}
                display_name_by_base: Dict[str, str] = {}
                for feature_name in minmax_candidates:
                    value = row_features.get(feature_name, "")
                    if value == "":
                        continue
                    lowered = feature_name.lower()
                    agg = "min" if ".min(" in lowered else ("max" if ".max(" in lowered else None)
                    if agg is None:
                        continue
                    base = _minmax_base_key(feature_name)
                    minmax_by_base.setdefault(base, {})[agg] = value
                    display_name_by_base.setdefault(base, feature_name)

                if not minmax_by_base:
                    continue

                rendered_pairs = []
                for base in sorted(minmax_by_base.keys()):
                    stats = minmax_by_base[base]
                    feature_label = display_name_by_base[base]
                    cleaned_label = re.sub(r"\.MIN\(", ".AGG(", feature_label)
                    cleaned_label = re.sub(r"\.MAX\(", ".AGG(", cleaned_label)
                    min_text = stats.get("min", "?")
                    max_text = stats.get("max", "?")
                    rendered_pairs.append(f"{cleaned_label}: min={min_text}, max={max_text}")

                timestamp_text = _cell_text(row.get(resource.time_col))
                lines.append(
                    f"- {resource.entity_col}={entity_key} at {timestamp_text}: "
                    + "; ".join(rendered_pairs)
                )

        self_rows: list[tuple[int, dict[str, Any], dict[str, str], bool]] = []
        other_rows: list[tuple[int, dict[str, Any], dict[str, str], bool]] = []
        query_rows: list[tuple[int, dict[str, Any], dict[str, str], bool]] = []

        for idx, (row, row_features) in enumerate(zip(entry_rows, entry_dfs_feature_dicts), 1):
            is_query_row = idx == len(entry_rows)
            scope = "query" if is_query_row else str(row.get("__example_scope", "self"))
            row_tuple = (idx, row, row_features, is_query_row)
            if scope == "self":
                self_rows.append(row_tuple)
            elif scope == "other":
                other_rows.append(row_tuple)
            else:
                query_rows.append(row_tuple)

        lines.append("")
        lines.append("DFS Table Notes:")
        lines.append("- Temporal order is oldest -> newest.")
        lines.append("- Self rows = primary evidence.")
        lines.append("- Neighbor rows = analogical evidence (not direct prediction).")
        _render_dfs_section("DFS Feature Table (Self Examples):", self_rows, "self")
        _render_dfs_section("DFS Feature Table (Other Examples):", other_rows, "other")
        _render_dfs_section("DFS Feature Table (Query Row):", query_rows, "query")

    if not table_only_mode:
        # =========================
        # 🔥 STRONG OUTPUT ANCHOR
        # =========================
        lines.append("")
        lines.append("Predict the output value for the Query using the output column from the examples.")
        lines.append("")
        lines.append("Answer (number only):")

    return "\n".join(lines)










def build_zero_shot_prompt_old(
    store,
    extractor,
    resource,
    query_row,
    history_rows,
    top_k,
    num_hops,
    include_hop_aggregation=True,
    include_semantic_retrieval=False,
    context_workers=1,  # retained for compatibility, not used
):
    ordered_history_rows = [dict(row) for row in history_rows]
    entry_rows = ordered_history_rows + [dict(query_row)]

    entry_contexts = []
    for row in entry_rows:
        entity_value = row[resource.entity_col]
        cutoff_time = pd.Timestamp(row[resource.time_col])
        node_id = store.node_id_map.get((resource.entity_table, entity_value))
        if node_id is None:
            continue

        context = build_semantic_context(
            store=store,
            extractor=extractor,
            node_id=node_id,
            cutoff_time=cutoff_time,
            k_hist=len(history_rows) if history_rows else top_k,
            top_k=top_k,
            num_hops=num_hops,
            include_semantic_retrieval=include_semantic_retrieval,
        )
        entry_contexts.append({"row": row, "context": context})

    lines = []
    lines.append(f"Task: {resource.output_col}")
    lines.append("Return ONLY a number.")
    lines.append(
        "The prompt includes prior examples of the same task for the same entity, each paired with neighborhood context from its timestamp, plus neighborhood context for the final query timestamp."
    )
    lines.append("")

    query_node_id = store.node_id_map.get((resource.entity_table, query_row[resource.entity_col]))
    if query_node_id is not None:
        static_once = []
        static_rowids = store.get_static_info(query_node_id)
        static_rows = extractor.get_many_rows(static_rowids)
        for rid in static_rowids:
            row = static_rows[rid]
            meta = store.decode_row(rid)
            static_once.append(format_row(meta["table"], row, None))

        if static_once:
            lines.append("Static:")
            for text in static_once:
                lines.append(f"- {text}")

    for idx, item in enumerate(entry_contexts, 1):
        row = item["row"]
        context = item["context"]
        lines.append("")
        is_query = idx == len(entry_contexts)

        if is_query:
            lines.append(
                f"Query {idx}: {format_task_query_row(resource.task, row, resource.time_col, resource.output_col)}"
            )
        else:
            lines.append(
                f"History Example {idx}: {format_task_query_row(resource.task, row, resource.time_col, resource.output_col)}"
            )

        if context["history"]:
            lines.append("History:")
            for history_item in context["history"]:
                lines.append(f"- {history_item['text']}")

        if context["neighbors"]:
            if include_hop_aggregation:
                lines.append("Neighbor Summary:")
                for summary in summarize_neighbors_by_hop(context["neighbors"]):
                    lines.append(
                        f"- hop {summary['hop']}: {summary['count']} neighbors"
                        + (f" ({summary['tables']})" if summary["tables"] else "")
                    )
            lines.append("Neighbors:")
            lines.extend(render_neighbor_graph(context["neighbors"]))

        if context.get("similar"):
            lines.append("Similar Entities:")
            for similar_item in context["similar"]:
                lines.append(
                    f"- {similar_item['entity']} | similarity={similar_item['score']} | {similar_item['static']}"
                )
                for history_text in similar_item["history"]:
                    lines.append(f"  history: {history_text}")

        if not is_query:
            lines.append(f"Output: {row[resource.output_col]}")

    lines.append("")
    lines.append("Question:")
    lines.append(f"Predict the {resource.output_col} for the final query.")
    lines.append("")
    lines.append("Answer:")

    return "\n".join(lines)
