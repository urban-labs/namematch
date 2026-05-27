"""Tests for Constraints fast-API detection and get_cluster_logic plumbing.

Phase 2 work: the loader recognizes is_valid_cluster_fast +
required_aggregations, validates that they're declared together, and exposes
the new fields on the Constraints instance. No behavior change in the hot
loop yet.
"""

import os
import sys
import tempfile
import textwrap
import types

import pandas as pd
import pytest

from namematch.cluster import Constraints, Cluster


# ---------------------------------------------------------------------------
# Constraints.uses_fast_api validation
# ---------------------------------------------------------------------------

def make_constraints(**fields):
    """Build a Constraints instance with arbitrary fields set, used to
    test the detection logic without needing to import a user module."""
    c = Constraints()
    for k, v in fields.items():
        setattr(c, k, v)
    return c


class TestUsesFastApi:
    def test_neither_field_present_returns_false(self):
        c = make_constraints()
        assert c.uses_fast_api() is False

    def test_both_fields_present_returns_true(self):
        c = make_constraints(
            is_valid_cluster_fast=lambda summary, phat: True,
            required_aggregations={'dob': 'set_str'},
        )
        assert c.uses_fast_api() is True

    def test_only_function_present_raises(self):
        c = make_constraints(is_valid_cluster_fast=lambda s, p: True)
        with pytest.raises(ValueError, match="required_aggregations"):
            c.uses_fast_api()

    def test_only_aggregations_present_raises(self):
        c = make_constraints(required_aggregations={'dob': 'set_str'})
        with pytest.raises(ValueError, match="is_valid_cluster_fast"):
            c.uses_fast_api()

    def test_aggregation_sources_alone_does_not_enable_fast_api(self):
        # aggregation_sources is metadata, not a trigger
        c = make_constraints(aggregation_sources={'fn_ln_dob': ('first_name', 'last_name', 'dob')})
        assert c.uses_fast_api() is False


# ---------------------------------------------------------------------------
# get_cluster_logic: detects fast API from a user constraints module on disk
# ---------------------------------------------------------------------------

def _write_constraints_module(tmpdir, body):
    """Write a Python source file under tmpdir and return its path."""
    path = os.path.join(tmpdir, 'user_constraints.py')
    with open(path, 'w') as f:
        f.write(textwrap.dedent(body))
    return path


@pytest.fixture
def cluster_instance():
    """A minimally-initialized Cluster instance, enough to call
    get_cluster_logic() in isolation. We don't go through the full
    NamematchBase init path - the loader only needs self.enable_lprof."""
    c = Cluster.__new__(Cluster)
    c.enable_lprof = False
    return c


class TestGetClusterLogicLegacy:
    def test_legacy_module_yields_no_fast_api(self, cluster_instance, tmp_path):
        path = _write_constraints_module(str(tmp_path), '''
            def get_columns_used():
                return "all"

            def is_valid_link(df):
                df["valid"] = True
                return df.valid

            def is_valid_cluster(cluster_df, phat):
                return True

            def apply_link_priority(df):
                return df
        ''')
        logic = cluster_instance.get_cluster_logic(path)
        assert logic.uses_fast_api() is False
        # Legacy callables still work
        assert callable(logic.is_valid_cluster)
        assert callable(logic.is_valid_link)
        # Fast-API attrs are None
        assert logic.is_valid_cluster_fast is None
        assert logic.required_aggregations is None
        assert logic.aggregation_sources is None


class TestGetClusterLogicFastApi:
    def test_fast_api_module_is_detected(self, cluster_instance, tmp_path):
        path = _write_constraints_module(str(tmp_path), '''
            def get_columns_used():
                return "all"

            def is_valid_link(df):
                df["valid"] = True
                return df.valid

            def is_valid_cluster(cluster_df, phat):
                # legacy fallback - shouldn't be called when fast is on
                return True

            def apply_link_priority(df):
                return df

            required_aggregations = {
                "dob": "set_str",
                "age": "int_min_max",
            }

            def is_valid_cluster_fast(summary, phat):
                if summary["size"] > 300:
                    return False
                return True
        ''')
        logic = cluster_instance.get_cluster_logic(path)
        assert logic.uses_fast_api() is True
        assert logic.required_aggregations == {
            "dob": "set_str",
            "age": "int_min_max",
        }
        assert callable(logic.is_valid_cluster_fast)
        # Calling with a summary dict works
        assert logic.is_valid_cluster_fast({"size": 5}, 0.8) is True
        assert logic.is_valid_cluster_fast({"size": 999}, 0.8) is False

    def test_fast_api_with_aggregation_sources(self, cluster_instance, tmp_path):
        path = _write_constraints_module(str(tmp_path), '''
            def get_columns_used():
                return "all"

            def is_valid_link(df):
                df["valid"] = True
                return df.valid

            def is_valid_cluster(df, phat):
                return True

            def apply_link_priority(df):
                return df

            required_aggregations = {"fn_ln_dob": "set_tuple"}
            aggregation_sources = {
                "fn_ln_dob": ("first_name", "last_name", "dob"),
            }

            def is_valid_cluster_fast(summary, phat):
                return len(summary["fn_ln_dob"]) <= 6
        ''')
        logic = cluster_instance.get_cluster_logic(path)
        assert logic.uses_fast_api() is True
        assert logic.aggregation_sources == {
            "fn_ln_dob": ("first_name", "last_name", "dob"),
        }

    def test_half_declaration_raises(self, cluster_instance, tmp_path):
        # is_valid_cluster_fast present, required_aggregations absent
        path = _write_constraints_module(str(tmp_path), '''
            def get_columns_used():
                return "all"

            def is_valid_link(df):
                df["valid"] = True
                return df.valid

            def is_valid_cluster(df, phat):
                return True

            def apply_link_priority(df):
                return df

            def is_valid_cluster_fast(summary, phat):
                return True
        ''')
        with pytest.raises(ValueError, match="required_aggregations"):
            cluster_instance.get_cluster_logic(path)


# ---------------------------------------------------------------------------
# load_cluster_info: when fast API is on, source columns are added to the
# column-load list automatically (we don't need to also list them in
# get_columns_used).
# ---------------------------------------------------------------------------

class _FakeVariables:
    def __init__(self, names):
        self._names = list(names)

    def get_an_column_names(self):
        return self._names

    def get_variables_where(self, attr, attr_value):
        return []  # no uid/eid cols in the fake


class _FakeSchema:
    def __init__(self, names):
        self.variables = _FakeVariables(names)


def test_load_cluster_info_includes_fast_api_source_columns(tmp_path):
    """The loader must add aggregation source columns when the fast API
    is opted into, so the ClusterStateStore can find them."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Build a small parquet with extra columns beyond what get_columns_used
    # asks for. The fast-API source cols ('age') should still get loaded.
    df = pd.DataFrame({
        'record_id': ['r1', 'r2', 'r3'],
        'dataset':   ['gt', 'gt', 'gt'],  # required by load_cluster_info
        'first_name': ['JOHN', 'JANE', 'JIM'],
        'last_name':  ['DOE', 'DOE', 'DOE'],
        'dob':        ['1990-01-01', '1990-02-01', '1991-03-01'],
        'age':        ['36', '36', '35'],
        'extra_col':  ['x', 'y', 'z'],  # should NOT be loaded
    })
    parquet_path = os.path.join(str(tmp_path), 'all_names.parquet')
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), parquet_path)

    # Fake cluster instance with the minimum scaffolding load_cluster_info
    # needs (schema for the 'all' fallback, but we won't hit that branch
    # because get_columns_used returns a dict).
    cluster = Cluster.__new__(Cluster)
    cluster.enable_lprof = False
    cluster.schema = _FakeSchema(names=list(df.columns))

    logic = Constraints()
    logic.get_columns_used = lambda: {'dob': 'str'}
    logic.is_valid_link = lambda df: pd.Series([True] * len(df))
    logic.is_valid_cluster = lambda df, phat: True
    logic.apply_link_priority = lambda df: df
    logic.is_valid_cluster_fast = lambda summary, phat: True
    logic.required_aggregations = {'age': 'int_min_max'}
    logic.aggregation_sources = None  # default sources -> ('age',)

    ci = cluster.load_cluster_info(parquet_path, [], None, logic)
    # dob was explicit in get_columns_used; age must be auto-added because
    # required_aggregations references it.
    assert 'age' in ci.columns
    assert 'dob' in ci.columns
    # extra_col was neither requested nor needed for aggregation -> excluded
    assert 'extra_col' not in ci.columns
