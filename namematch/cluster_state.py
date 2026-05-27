"""Per-cluster aggregated state for fast greedy clustering.

The clustering step in namematch's hot loop spends most of its time slicing
pandas DataFrames to build a candidate-merge view for each potential edge.
This module replaces that with incrementally-maintained per-cluster state:
constraints query a small summary dict instead of a DataFrame.

The aggregations supported here are merge-friendly (associative + commutative),
so combining two clusters is O(1) or O(k) where k is the size of the smaller
cluster's state. Supported types:

- set_str       : union of distinct string values
- set_tuple     : union of distinct tuples (from N source columns)
- set_int       : union of distinct ints
- int_min_max   : cluster-wide (min, max)
- int_sum       : running total
- float_min_max : like int_min_max but for floats

This module is standalone. Integration with cluster.py happens in a later
phase; for now it can be exercised in isolation and unit-tested.
"""

from typing import Any, Dict, Mapping, Sequence, Tuple

import pandas as pd


class Aggregation:
    """Base class for an aggregation type.

    Subclasses must implement:
      - init_from_record(values): build state from one record's source values
      - merge(a, b): combine two states into one (must be associative)
      - empty(): identity element for merge (so empty.merge(x) == x)

    Subclasses can override unpack() if their internal storage shape differs
    from how callers expect to see the result in the summary dict.
    """

    @classmethod
    def init_from_record(cls, values: Sequence[Any]) -> Any:
        raise NotImplementedError

    @classmethod
    def merge(cls, a: Any, b: Any) -> Any:
        raise NotImplementedError

    @classmethod
    def empty(cls) -> Any:
        """Identity element: empty.merge(state) == state for any state."""
        raise NotImplementedError

    @classmethod
    def unpack(cls, name: str, state: Any) -> Dict[str, Any]:
        """Default: a single key in the summary dict with the raw state."""
        return {name: state}


class SetStr(Aggregation):
    @classmethod
    def init_from_record(cls, values):
        v = values[0]
        return {str(v)} if pd.notna(v) else set()

    @classmethod
    def merge(cls, a, b):
        return a | b

    @classmethod
    def empty(cls):
        return set()


class SetInt(Aggregation):
    @classmethod
    def init_from_record(cls, values):
        v = values[0]
        return {int(v)} if pd.notna(v) else set()

    @classmethod
    def merge(cls, a, b):
        return a | b

    @classmethod
    def empty(cls):
        return set()


class SetTuple(Aggregation):
    """Union of N-column tuples. Skips records with any NA in source cols."""

    @classmethod
    def init_from_record(cls, values):
        if any(pd.isna(v) for v in values):
            return set()
        return {tuple(values)}

    @classmethod
    def merge(cls, a, b):
        return a | b

    @classmethod
    def empty(cls):
        return set()


class IntMinMax(Aggregation):
    """Stored as (min, max) tuple. Records with NA contribute nothing."""

    @classmethod
    def init_from_record(cls, values):
        v = values[0]
        if pd.isna(v):
            return None
        i = int(v)
        return (i, i)

    @classmethod
    def merge(cls, a, b):
        if a is None:
            return b
        if b is None:
            return a
        return (min(a[0], b[0]), max(a[1], b[1]))

    @classmethod
    def empty(cls):
        return None

    @classmethod
    def unpack(cls, name, state):
        if state is None:
            return {f'{name}_min': None, f'{name}_max': None}
        return {f'{name}_min': state[0], f'{name}_max': state[1]}


class FloatMinMax(IntMinMax):
    @classmethod
    def init_from_record(cls, values):
        v = values[0]
        if pd.isna(v):
            return None
        f = float(v)
        return (f, f)


class IntSum(Aggregation):
    @classmethod
    def init_from_record(cls, values):
        v = values[0]
        return int(v) if pd.notna(v) else 0

    @classmethod
    def merge(cls, a, b):
        return a + b

    @classmethod
    def empty(cls):
        return 0


AGGREGATIONS: Dict[str, type] = {
    'set_str':       SetStr,
    'set_int':       SetInt,
    'set_tuple':     SetTuple,
    'int_min_max':   IntMinMax,
    'int_sum':       IntSum,
    'float_min_max': FloatMinMax,
}


class ClusterStateStore:
    """Per-cluster aggregated state for fast constraint evaluation.

    Built once from all_names, then mutated incrementally as clusters merge.
    Constraint code queries summaries via merge_preview() rather than slicing
    a DataFrame.

    Cluster ids are taken from the all_names record_id column at
    construction (one record per cluster initially). The caller is
    responsible for choosing a cid_keep / cid_drop convention when merging.
    """

    def __init__(
        self,
        all_names_df: pd.DataFrame,
        aggregations: Mapping[str, str],
        sources: Mapping[str, Tuple[str, ...]] = None,
        record_id_col: str = 'record_id',
        uid_col: str = 'uid',
        initial_clusters: Mapping[Any, Sequence[Any]] = None,
    ):
        """
        Args:
            all_names_df: one row per record. Must contain record_id_col and
                every source column referenced by aggregations.
            aggregations: {agg_name: agg_type_name}. agg_type_name must be a
                key in AGGREGATIONS.
            sources: {agg_name: (col_1, col_2, ...)} for multi-column aggs.
                Single-column aggs default to (agg_name,).
            record_id_col: column holding the record id.
            uid_col: column holding the uid. Always tracked as set_str.
                Set to None to skip uid aggregation entirely.
            initial_clusters: {cluster_id: [record_id, ...]} mapping. When
                provided, the store uses these cluster_ids (not record_ids)
                as its keys and aggregates each cluster's records together.
                When None (default), each record becomes its own singleton
                cluster keyed by record_id. The non-None form is how
                Cluster.main() seeds the store after must-links produce
                multi-record initial clusters.
        """
        sources = sources or {}

        self._aggs: Dict[str, type] = {}
        self._sources: Dict[str, Tuple[str, ...]] = {}
        for agg_name, type_name in aggregations.items():
            if type_name not in AGGREGATIONS:
                raise ValueError(
                    f"Unknown aggregation type '{type_name}' for "
                    f"aggregation '{agg_name}'. Valid types: "
                    f"{sorted(AGGREGATIONS)}"
                )
            self._aggs[agg_name] = AGGREGATIONS[type_name]
            cols = sources.get(agg_name, (agg_name,))
            if isinstance(cols, str):
                cols = (cols,)
            self._sources[agg_name] = tuple(cols)

        for agg_name, cols in self._sources.items():
            for col in cols:
                if col not in all_names_df.columns:
                    raise KeyError(
                        f"Aggregation '{agg_name}' references column "
                        f"'{col}' which is not in all_names_df"
                    )
        if record_id_col not in all_names_df.columns:
            raise KeyError(
                f"record_id_col '{record_id_col}' not in all_names_df"
            )
        if uid_col is not None and uid_col not in all_names_df.columns:
            raise KeyError(
                f"uid_col '{uid_col}' not in all_names_df "
                f"(pass uid_col=None to skip)"
            )

        self._uid_col = uid_col

        # cluster_id -> per-cluster state
        self.size: Dict[Any, int] = {}
        self.uid:  Dict[Any, set] = {}
        # agg_name -> {cluster_id -> state}
        self.state: Dict[str, Dict[Any, Any]] = {a: {} for a in self._aggs}

        if initial_clusters is None:
            self._init_singletons(all_names_df, record_id_col)
        else:
            self._init_with_clusters(
                all_names_df, record_id_col, initial_clusters)

    def _extract_column_arrays(self, df: pd.DataFrame, record_id_col: str):
        """Cache numpy arrays for each source column to avoid repeated
        pandas attribute lookups in the per-row init loop."""
        record_ids = df[record_id_col].values
        uid_values = df[self._uid_col].values if self._uid_col else None
        agg_col_arrays = {
            agg_name: tuple(df[c].values for c in cols)
            for agg_name, cols in self._sources.items()
        }
        return record_ids, uid_values, agg_col_arrays

    def _init_singletons(self, df: pd.DataFrame, record_id_col: str) -> None:
        """One cluster per record; cluster_id = record_id."""
        record_ids, uid_values, agg_col_arrays = \
            self._extract_column_arrays(df, record_id_col)

        n = len(df)
        for i in range(n):
            rid = record_ids[i]
            self.size[rid] = 1
            if uid_values is not None:
                u = uid_values[i]
                self.uid[rid] = {str(u)} if pd.notna(u) else set()
            else:
                self.uid[rid] = set()
            for agg_name, agg_cls in self._aggs.items():
                col_arrays = agg_col_arrays[agg_name]
                values = tuple(arr[i] for arr in col_arrays)
                self.state[agg_name][rid] = agg_cls.init_from_record(values)

    def _init_with_clusters(
        self,
        df: pd.DataFrame,
        record_id_col: str,
        initial_clusters: Mapping[Any, Sequence[Any]],
    ) -> None:
        """Multi-record-per-cluster init. cluster_ids come from the keys
        of initial_clusters; each cluster's state is the merge of its
        records' contributions."""
        # record_id -> cluster_id lookup
        rid_to_cid: Dict[Any, Any] = {}
        for cid, rids in initial_clusters.items():
            for rid in rids:
                rid_to_cid[rid] = cid

        # Seed every cluster with the identity (empty) state.
        for cid in initial_clusters:
            self.size[cid] = 0
            self.uid[cid] = set()
            for agg_name, agg_cls in self._aggs.items():
                self.state[agg_name][cid] = agg_cls.empty()

        record_ids, uid_values, agg_col_arrays = \
            self._extract_column_arrays(df, record_id_col)

        n = len(df)
        for i in range(n):
            rid = record_ids[i]
            cid = rid_to_cid.get(rid)
            if cid is None:
                # Record not present in any initial cluster - skipped.
                # This shouldn't happen in normal namematch flow because
                # get_initial_clusters returns clusters covering all of
                # all_names (singletons + must-link components).
                continue
            self.size[cid] += 1
            if uid_values is not None:
                u = uid_values[i]
                if pd.notna(u):
                    self.uid[cid].add(str(u))
            for agg_name, agg_cls in self._aggs.items():
                col_arrays = agg_col_arrays[agg_name]
                values = tuple(arr[i] for arr in col_arrays)
                contrib = agg_cls.init_from_record(values)
                self.state[agg_name][cid] = agg_cls.merge(
                    self.state[agg_name][cid], contrib)

    def summary(self, cid: Any) -> Dict[str, Any]:
        """Summary dict for a single cluster (no merge)."""
        out: Dict[str, Any] = {
            'size': self.size[cid],
            'uid':  self.uid[cid],
        }
        for agg_name, agg_cls in self._aggs.items():
            out.update(agg_cls.unpack(agg_name, self.state[agg_name][cid]))
        return out

    def merge_preview(self, cid_a: Any, cid_b: Any) -> Dict[str, Any]:
        """Compute the summary that would result from merging cid_a and
        cid_b, without modifying state. Pass this to is_valid_cluster_fast.
        """
        out: Dict[str, Any] = {
            'size': self.size[cid_a] + self.size[cid_b],
            'uid':  self.uid[cid_a] | self.uid[cid_b],
        }
        for agg_name, agg_cls in self._aggs.items():
            merged = agg_cls.merge(
                self.state[agg_name][cid_a],
                self.state[agg_name][cid_b],
            )
            out.update(agg_cls.unpack(agg_name, merged))
        return out

    def commit_merge(self, cid_keep: Any, cid_drop: Any) -> None:
        """Fold cid_drop's state into cid_keep, then remove cid_drop."""
        if cid_keep == cid_drop:
            raise ValueError("cid_keep and cid_drop must differ")
        self.size[cid_keep] += self.size[cid_drop]
        self.uid[cid_keep] = self.uid[cid_keep] | self.uid[cid_drop]
        del self.size[cid_drop]
        del self.uid[cid_drop]
        for agg_name, agg_cls in self._aggs.items():
            self.state[agg_name][cid_keep] = agg_cls.merge(
                self.state[agg_name][cid_keep],
                self.state[agg_name][cid_drop],
            )
            del self.state[agg_name][cid_drop]

    def __contains__(self, cid: Any) -> bool:
        return cid in self.size

    def __len__(self) -> int:
        return len(self.size)
