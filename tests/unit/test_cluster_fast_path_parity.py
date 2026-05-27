"""Parity test: fast path produces identical cluster_assignments as legacy
when given semantically-equivalent constraints.

This is the regression guard for phase 3. If any future change to either
path breaks the equivalence on this synthetic case, the test fails loudly.
"""

import os
import types

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from namematch.cluster import Cluster, Constraints
from namematch.cluster_state import ClusterStateStore


# ---------------------------------------------------------------------------
# Small synthetic fixture: 12 records across 3 "true" people, plus some
# decoys. Edges connect plausible candidates with varying phat scores.
# ---------------------------------------------------------------------------

def make_records():
    return pd.DataFrame([
        # "Person A": John Doe, 1990, uid=A
        {'record_id': 'r0', 'first_name': 'JOHN',  'last_name': 'DOE',  'dob': '1990-01-01', 'age': 36, 'uid': 'A'},
        {'record_id': 'r1', 'first_name': 'JOHN',  'last_name': 'DOE',  'dob': '1990-02-01', 'age': 36, 'uid': 'A'},
        {'record_id': 'r2', 'first_name': 'JON',   'last_name': 'DOE',  'dob': '1990-01-01', 'age': 36, 'uid': 'A'},
        # "Person B": Jane Smith, 1985, uid=B
        {'record_id': 'r3', 'first_name': 'JANE',  'last_name': 'SMITH', 'dob': '1985-05-01', 'age': 41, 'uid': 'B'},
        {'record_id': 'r4', 'first_name': 'JAYNE', 'last_name': 'SMITH', 'dob': '1985-05-01', 'age': 41, 'uid': 'B'},
        # "Person C": Bob Lee, 1972, uid=C
        {'record_id': 'r5', 'first_name': 'BOB',   'last_name': 'LEE',   'dob': '1972-03-01', 'age': 54, 'uid': 'C'},
        {'record_id': 'r6', 'first_name': 'ROBERT','last_name': 'LEE',   'dob': '1972-03-01', 'age': 54, 'uid': 'C'},
        # Decoys with valid edges that should NOT merge
        {'record_id': 'r7', 'first_name': 'JOHN',  'last_name': 'DOE',   'dob': '1960-01-01', 'age': 66, 'uid': 'D'},
        {'record_id': 'r8', 'first_name': 'JANE',  'last_name': 'SMITH', 'dob': '1940-01-01', 'age': 86, 'uid': 'E'},
        # Records that should stay singletons
        {'record_id': 'r9',  'first_name': 'ALICE', 'last_name': 'JONES', 'dob': '1995-01-01', 'age': 31, 'uid': 'F'},
        {'record_id': 'r10', 'first_name': 'CARL',  'last_name': 'NG',    'dob': '1980-01-01', 'age': 46, 'uid': 'G'},
        {'record_id': 'r11', 'first_name': 'EVE',   'last_name': 'WU',    'dob': '1988-01-01', 'age': 38, 'uid': 'H'},
    ])


def make_edges():
    """Edges ranked by phat (descending), as the clustering loop expects.
    Mix of true positives, ground-truth, and edges that should be rejected
    by the cluster-level constraints we'll define below."""
    return pd.DataFrame([
        # True person-A triplet, high confidence
        {'record_id_1': 'r0', 'record_id_2': 'r1', 'phat': 0.95, 'gt': 0},
        {'record_id_1': 'r0', 'record_id_2': 'r2', 'phat': 0.93, 'gt': 0},
        # Person-B pair
        {'record_id_1': 'r3', 'record_id_2': 'r4', 'phat': 0.91, 'gt': 0},
        # Person-C pair
        {'record_id_1': 'r5', 'record_id_2': 'r6', 'phat': 0.90, 'gt': 0},
        # Decoy edges that pass the model but should fail the cluster-level
        # age-range constraint (>3y gap):
        {'record_id_1': 'r0', 'record_id_2': 'r7', 'phat': 0.85, 'gt': 0},  # 36 vs 66
        {'record_id_1': 'r3', 'record_id_2': 'r8', 'phat': 0.82, 'gt': 0},  # 41 vs 86
        # Random edges that should be rejected by phat-level OR by the
        # cluster size cap (we set cap = 4):
        {'record_id_1': 'r9',  'record_id_2': 'r10', 'phat': 0.75, 'gt': 0},
        {'record_id_1': 'r10', 'record_id_2': 'r11', 'phat': 0.74, 'gt': 0},
    ])


def write_edges_parquet(edges_df, path):
    """Write edges in the format cluster_potential_edges expects."""
    pq.write_table(pa.Table.from_pandas(edges_df, preserve_index=False), path)


# ---------------------------------------------------------------------------
# Two semantically-equivalent constraint modules: one legacy, one fast.
# Both reject merges where:
#   - cluster size > 4
#   - age range (max - min) > 3
# ---------------------------------------------------------------------------

def make_legacy_constraints():
    mod = types.ModuleType('legacy_test_constraints')

    def get_columns_used():
        return {'age': 'int', 'dob': 'str', 'first_name': 'str', 'last_name': 'str'}

    def is_valid_link(df):
        df['valid'] = True
        return df.valid

    def is_valid_cluster(cluster_df, phat):
        if cluster_df.shape[0] > 4:
            return False
        if cluster_df['age'].astype(int).max() - cluster_df['age'].astype(int).min() > 3:
            return False
        return True

    def apply_link_priority(df):
        return df.sort_values(by=['phat'], ascending=False)

    mod.get_columns_used = get_columns_used
    mod.is_valid_link = is_valid_link
    mod.is_valid_cluster = is_valid_cluster
    mod.apply_link_priority = apply_link_priority
    return mod


def make_fast_constraints():
    mod = types.ModuleType('fast_test_constraints')

    def get_columns_used():
        return {'age': 'int', 'dob': 'str', 'first_name': 'str', 'last_name': 'str'}

    def is_valid_link(df):
        df['valid'] = True
        return df.valid

    def is_valid_cluster(cluster_df, phat):
        # Kept for completeness; should not be called when fast API in use.
        return True

    def apply_link_priority(df):
        return df.sort_values(by=['phat'], ascending=False)

    required_aggregations = {'age': 'int_min_max'}

    def is_valid_cluster_fast(summary, phat):
        if summary['size'] > 4:
            return False
        if summary['age_max'] - summary['age_min'] > 3:
            return False
        return True

    mod.get_columns_used = get_columns_used
    mod.is_valid_link = is_valid_link
    mod.is_valid_cluster = is_valid_cluster
    mod.apply_link_priority = apply_link_priority
    mod.required_aggregations = required_aggregations
    mod.is_valid_cluster_fast = is_valid_cluster_fast
    return mod


# ---------------------------------------------------------------------------
# Test scaffolding: a minimal Cluster instance set up to run only the
# loop, not the full Cluster.main() pipeline.
# ---------------------------------------------------------------------------

class _FakeParams:
    cluster_batch_size = 100
    allow_clusters_w_multiple_unique_ids = True   # uid uniqueness off (CLUE config)
    leven_thresh = None
    verbose = None


def make_cluster(edges_parquet_path):
    c = Cluster.__new__(Cluster)
    c.enable_lprof = False
    c.params = _FakeParams()
    c.stats_dict = {}
    c.edges = edges_parquet_path
    return c


def initial_state(records_df):
    """Singletons: cluster_id = record_id, each cluster holds one record."""
    cluster_assignments = {rid: rid for rid in records_df.record_id}
    clusters = {rid: [rid] for rid in records_df.record_id}
    return clusters, cluster_assignments


# ---------------------------------------------------------------------------
# Parity test
# ---------------------------------------------------------------------------

def _legacy_constraints_instance(records_df):
    """Build a Constraints instance + matching cluster_info DataFrame."""
    legacy_mod = make_legacy_constraints()
    logic = Constraints()
    logic.get_columns_used = legacy_mod.get_columns_used
    logic.is_valid_link = legacy_mod.is_valid_link
    logic.is_valid_cluster = legacy_mod.is_valid_cluster
    logic.apply_link_priority = legacy_mod.apply_link_priority
    cluster_info = records_df.set_index('record_id').copy()
    cluster_info['age'] = cluster_info['age'].astype(int)
    return logic, cluster_info


def _fast_constraints_instance(records_df):
    fast_mod = make_fast_constraints()
    logic = Constraints()
    logic.get_columns_used = fast_mod.get_columns_used
    logic.is_valid_link = fast_mod.is_valid_link
    logic.is_valid_cluster = fast_mod.is_valid_cluster
    logic.apply_link_priority = fast_mod.apply_link_priority
    logic.is_valid_cluster_fast = fast_mod.is_valid_cluster_fast
    logic.required_aggregations = fast_mod.required_aggregations
    logic.aggregation_sources = None
    cluster_info = records_df.set_index('record_id').copy()
    cluster_info['age'] = cluster_info['age'].astype(int)
    return logic, cluster_info


def test_fast_path_matches_legacy_assignments(tmp_path):
    records_df = make_records()
    edges_df = make_edges()
    edges_path = os.path.join(str(tmp_path), 'edges_to_cluster.parquet')
    write_edges_parquet(edges_df, edges_path)

    # ---- LEGACY RUN ----
    cluster_legacy = make_cluster(edges_path)
    legacy_logic, legacy_ci = _legacy_constraints_instance(records_df)
    clusters_L, assignments_L = initial_state(records_df)
    legacy_assignments = cluster_legacy.cluster_potential_edges(
        clusters_L, assignments_L, original_cluster_ids=None,
        cluster_info=legacy_ci, cluster_logic=legacy_logic,
        uid_cols=['uid'], eid_col=None, cluster_state=None,
    )

    # ---- FAST RUN ----
    cluster_fast = make_cluster(edges_path)
    fast_logic, fast_ci = _fast_constraints_instance(records_df)
    clusters_F, assignments_F = initial_state(records_df)
    state = ClusterStateStore(
        records_df.copy(),
        aggregations=fast_logic.required_aggregations,
        sources=fast_logic.aggregation_sources or {},
        record_id_col='record_id',
        uid_col='uid',
        initial_clusters=clusters_F,
    )
    fast_assignments = cluster_fast.cluster_potential_edges(
        clusters_F, assignments_F, original_cluster_ids=None,
        cluster_info=fast_ci, cluster_logic=fast_logic,
        uid_cols=['uid'], eid_col=None, cluster_state=state,
    )

    # ---- ASSERT PARITY ----
    # Same set of records assigned to clusters (sanity)
    assert set(legacy_assignments) == set(fast_assignments)

    # Cluster_assignments map each record to a (string) cluster id. The
    # numeric id may differ between runs because of how dict iteration
    # picks the "first" id, but the partition must be the same. So we
    # compare PARTITIONS, not cluster_id-to-cluster_id equality.
    from collections import defaultdict

    def partition(assignments):
        parts = defaultdict(set)
        for rid, cid in assignments.items():
            parts[cid].add(rid)
        return frozenset(frozenset(s) for s in parts.values())

    legacy_partition = partition(legacy_assignments)
    fast_partition = partition(fast_assignments)
    assert legacy_partition == fast_partition, (
        f"Partition mismatch.\n"
        f"  Legacy: {sorted([sorted(s) for s in legacy_partition])}\n"
        f"  Fast:   {sorted([sorted(s) for s in fast_partition])}"
    )

    # Sanity: the partition is what we expect from the constraints.
    # Person-A trio should merge (size 3, age range 0).
    # Decoy edges (r0-r7, r3-r8) should be REJECTED (age range > 3).
    # Person-B pair, Person-C pair merge.
    # r9, r10, r11 stay singletons or merge if size-cap permits.
    expected = {
        frozenset({'r0', 'r1', 'r2'}),
        frozenset({'r3', 'r4'}),
        frozenset({'r5', 'r6'}),
        frozenset({'r7'}),
        frozenset({'r8'}),
    }
    # Match the expected sub-partition (others may be singletons or
    # small merges depending on edge order — keep the assertion focused
    # on the key constraint-driven groupings).
    for grp in expected:
        assert grp in fast_partition, f"Expected group {grp} not in fast partition"


def test_fast_path_stats_match_legacy(tmp_path):
    """Same setup, also verify the recorded merge/invalid_cluster counts."""
    records_df = make_records()
    edges_df = make_edges()
    edges_path = os.path.join(str(tmp_path), 'edges_to_cluster.parquet')
    write_edges_parquet(edges_df, edges_path)

    cluster_legacy = make_cluster(edges_path)
    legacy_logic, legacy_ci = _legacy_constraints_instance(records_df)
    clusters_L, assignments_L = initial_state(records_df)
    cluster_legacy.cluster_potential_edges(
        clusters_L, assignments_L, original_cluster_ids=None,
        cluster_info=legacy_ci, cluster_logic=legacy_logic,
        uid_cols=['uid'], eid_col=None, cluster_state=None,
    )

    cluster_fast = make_cluster(edges_path)
    fast_logic, fast_ci = _fast_constraints_instance(records_df)
    clusters_F, assignments_F = initial_state(records_df)
    state = ClusterStateStore(
        records_df.copy(),
        aggregations=fast_logic.required_aggregations,
        sources=fast_logic.aggregation_sources or {},
        record_id_col='record_id',
        uid_col='uid',
        initial_clusters=clusters_F,
    )
    cluster_fast.cluster_potential_edges(
        clusters_F, assignments_F, original_cluster_ids=None,
        cluster_info=fast_ci, cluster_logic=fast_logic,
        uid_cols=['uid'], eid_col=None, cluster_state=state,
    )

    # Same number of merges and rejections recorded
    assert cluster_legacy.stats_dict['n_invalid_clusters'] == \
           cluster_fast.stats_dict['n_invalid_clusters']
    assert cluster_legacy.stats_dict['n_clusters'] == \
           cluster_fast.stats_dict['n_clusters']
