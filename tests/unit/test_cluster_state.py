"""Unit tests for ClusterStateStore and aggregation primitives."""

import pandas as pd
import pytest

from namematch.cluster_state import (
    AGGREGATIONS,
    ClusterStateStore,
    FloatMinMax,
    IntMinMax,
    IntSum,
    SetInt,
    SetStr,
    SetTuple,
)


def make_df():
    """Small fixture: 4 records covering happy + missing-value cases."""
    return pd.DataFrame([
        {'record_id': 'r1', 'first_name': 'JOHN', 'last_name': 'DOE',
         'dob': '1990-01-01', 'age': 36, 'uid': 'u1'},
        {'record_id': 'r2', 'first_name': 'JANE', 'last_name': 'DOE',
         'dob': '1990-02-01', 'age': 36, 'uid': 'u1'},
        {'record_id': 'r3', 'first_name': 'JON',  'last_name': 'DOE',
         'dob': '1991-01-01', 'age': 35, 'uid': 'u2'},
        {'record_id': 'r4', 'first_name': 'JANE', 'last_name': 'DOE',
         'dob': None,         'age': None, 'uid': None},
    ])


class TestPrimitiveAggregations:
    def test_set_str_init(self):
        assert SetStr.init_from_record(['JOHN']) == {'JOHN'}
        assert SetStr.init_from_record([None]) == set()
        # ints coerced to str
        assert SetStr.init_from_record([42]) == {'42'}

    def test_set_str_merge(self):
        assert SetStr.merge({'A'}, {'B'}) == {'A', 'B'}
        assert SetStr.merge({'A'}, set()) == {'A'}
        assert SetStr.merge(set(), set()) == set()

    def test_set_int_init(self):
        assert SetInt.init_from_record([5]) == {5}
        assert SetInt.init_from_record([None]) == set()

    def test_set_int_merge(self):
        assert SetInt.merge({1, 2}, {2, 3}) == {1, 2, 3}

    def test_set_tuple_init_all_present(self):
        assert SetTuple.init_from_record(['JOHN', 'DOE', '1990-01-01']) == \
            {('JOHN', 'DOE', '1990-01-01')}

    def test_set_tuple_init_any_missing_returns_empty(self):
        # Single missing source value -> the whole tuple is dropped, not
        # partially recorded. Prevents partial tuples polluting the set.
        assert SetTuple.init_from_record(['JOHN', None, '1990']) == set()
        assert SetTuple.init_from_record([None, None, None]) == set()

    def test_set_tuple_merge(self):
        a = {('JOHN', 'DOE')}
        b = {('JANE', 'DOE')}
        assert SetTuple.merge(a, b) == {('JOHN', 'DOE'), ('JANE', 'DOE')}

    def test_int_min_max_init(self):
        assert IntMinMax.init_from_record([36]) == (36, 36)
        assert IntMinMax.init_from_record([None]) is None

    def test_int_min_max_merge(self):
        assert IntMinMax.merge((10, 20), (15, 25)) == (10, 25)
        # nan/None propagation: None side contributes nothing
        assert IntMinMax.merge((10, 20), None) == (10, 20)
        assert IntMinMax.merge(None, (10, 20)) == (10, 20)
        assert IntMinMax.merge(None, None) is None

    def test_int_min_max_unpack(self):
        # Internal (min, max) tuple expands to two summary keys
        assert IntMinMax.unpack('age', (10, 20)) == \
            {'age_min': 10, 'age_max': 20}
        assert IntMinMax.unpack('age', None) == \
            {'age_min': None, 'age_max': None}

    def test_float_min_max(self):
        assert FloatMinMax.init_from_record([3.5]) == (3.5, 3.5)
        assert FloatMinMax.merge((1.0, 2.0), (1.5, 3.0)) == (1.0, 3.0)

    def test_int_sum(self):
        assert IntSum.init_from_record([5]) == 5
        assert IntSum.init_from_record([None]) == 0
        assert IntSum.merge(3, 4) == 7

    def test_registry_completeness(self):
        # Every aggregation type referenced in docstring is registered
        expected = {'set_str', 'set_int', 'set_tuple',
                    'int_min_max', 'int_sum', 'float_min_max'}
        assert set(AGGREGATIONS) == expected


class TestClusterStateStoreInit:
    def test_singleton_clusters_after_init(self):
        df = make_df()
        store = ClusterStateStore(
            df,
            aggregations={'dob': 'set_str', 'age': 'int_min_max'},
        )
        assert len(store) == 4
        for rid in ('r1', 'r2', 'r3', 'r4'):
            assert rid in store
            assert store.size[rid] == 1

    def test_summary_singleton(self):
        df = make_df()
        store = ClusterStateStore(
            df,
            aggregations={'dob': 'set_str', 'age': 'int_min_max'},
        )
        s = store.summary('r1')
        assert s['size'] == 1
        assert s['uid'] == {'u1'}
        assert s['dob'] == {'1990-01-01'}
        assert s['age_min'] == 36
        assert s['age_max'] == 36

    def test_missing_values_handled(self):
        df = make_df()
        store = ClusterStateStore(
            df,
            aggregations={'dob': 'set_str', 'age': 'int_min_max'},
        )
        s = store.summary('r4')
        assert s['size'] == 1
        assert s['uid'] == set()  # uid was None
        assert s['dob'] == set()
        assert s['age_min'] is None
        assert s['age_max'] is None

    def test_multi_column_aggregation(self):
        df = make_df()
        store = ClusterStateStore(
            df,
            aggregations={'fn_ln_dob': 'set_tuple'},
            sources={'fn_ln_dob': ('first_name', 'last_name', 'dob')},
        )
        assert store.summary('r1')['fn_ln_dob'] == \
            {('JOHN', 'DOE', '1990-01-01')}
        # Missing dob -> empty for r4
        assert store.summary('r4')['fn_ln_dob'] == set()

    def test_uid_col_none_skips_uid(self):
        df = make_df()
        store = ClusterStateStore(
            df,
            aggregations={'dob': 'set_str'},
            uid_col=None,
        )
        assert store.summary('r1')['uid'] == set()

    def test_unknown_aggregation_type_raises(self):
        df = make_df()
        with pytest.raises(ValueError, match="Unknown aggregation type"):
            ClusterStateStore(df, aggregations={'dob': 'bogus_type'})

    def test_missing_source_column_raises(self):
        df = make_df()
        with pytest.raises(KeyError, match="nonexistent_col"):
            ClusterStateStore(
                df,
                aggregations={'foo': 'set_str'},
                sources={'foo': ('nonexistent_col',)},
            )

    def test_missing_uid_col_raises_unless_none(self):
        df = make_df().drop(columns=['uid'])
        with pytest.raises(KeyError, match="uid_col 'uid'"):
            ClusterStateStore(df, aggregations={'dob': 'set_str'})
        # Should succeed with uid_col=None
        store = ClusterStateStore(
            df, aggregations={'dob': 'set_str'}, uid_col=None,
        )
        assert len(store) == 4


class TestMergePreview:
    def test_does_not_mutate_state(self):
        df = make_df()
        store = ClusterStateStore(df, aggregations={'dob': 'set_str'})
        preview = store.merge_preview('r1', 'r2')
        assert preview['size'] == 2
        assert preview['dob'] == {'1990-01-01', '1990-02-01'}
        assert preview['uid'] == {'u1'}
        # Original cluster state untouched
        assert store.size['r1'] == 1
        assert store.size['r2'] == 1
        assert 'r2' in store
        assert store.summary('r1')['dob'] == {'1990-01-01'}

    def test_preview_unions_uids(self):
        df = make_df()
        store = ClusterStateStore(df, aggregations={'dob': 'set_str'})
        preview = store.merge_preview('r1', 'r3')
        assert preview['uid'] == {'u1', 'u2'}

    def test_preview_min_max_across_records(self):
        df = make_df()
        store = ClusterStateStore(df, aggregations={'age': 'int_min_max'})
        preview = store.merge_preview('r1', 'r3')  # ages 36 and 35
        assert preview['age_min'] == 35
        assert preview['age_max'] == 36


class TestCommitMerge:
    def test_basic_merge(self):
        df = make_df()
        store = ClusterStateStore(df, aggregations={'age': 'int_min_max'})
        store.commit_merge('r1', 'r3')  # r3 folded into r1
        assert store.size['r1'] == 2
        assert 'r3' not in store
        s = store.summary('r1')
        assert s['age_min'] == 35
        assert s['age_max'] == 36
        assert s['uid'] == {'u1', 'u2'}

    def test_merge_same_id_raises(self):
        df = make_df()
        store = ClusterStateStore(df, aggregations={'dob': 'set_str'})
        with pytest.raises(ValueError, match="must differ"):
            store.commit_merge('r1', 'r1')

    def test_cascading_merges_size_correct(self):
        df = make_df()
        store = ClusterStateStore(df, aggregations={'dob': 'set_str'})
        store.commit_merge('r1', 'r2')
        store.commit_merge('r1', 'r3')
        store.commit_merge('r1', 'r4')
        assert store.size['r1'] == 4
        assert len(store) == 1
        # r4 had null dob -> empty contribution; final set has 3 elements
        assert store.summary('r1')['dob'] == \
            {'1990-01-01', '1990-02-01', '1991-01-01'}


class TestAssociativity:
    """All aggregations must be associative so merge order is irrelevant."""

    def test_set_str_associativity(self):
        df = make_df()

        store_a = ClusterStateStore(df, aggregations={'dob': 'set_str'})
        store_a.commit_merge('r1', 'r2')
        store_a.commit_merge('r1', 'r3')

        store_b = ClusterStateStore(df, aggregations={'dob': 'set_str'})
        store_b.commit_merge('r2', 'r3')
        store_b.commit_merge('r1', 'r2')

        assert store_a.summary('r1')['dob'] == store_b.summary('r1')['dob']
        assert store_a.summary('r1')['uid'] == store_b.summary('r1')['uid']
        assert store_a.summary('r1')['size'] == store_b.summary('r1')['size']

    def test_int_min_max_associativity_with_nulls(self):
        df = make_df()

        store_a = ClusterStateStore(df, aggregations={'age': 'int_min_max'})
        store_a.commit_merge('r1', 'r4')  # r4 has null age
        store_a.commit_merge('r1', 'r3')

        store_b = ClusterStateStore(df, aggregations={'age': 'int_min_max'})
        store_b.commit_merge('r3', 'r4')
        store_b.commit_merge('r1', 'r3')

        assert store_a.summary('r1')['age_min'] == \
            store_b.summary('r1')['age_min']
        assert store_a.summary('r1')['age_max'] == \
            store_b.summary('r1')['age_max']

    def test_set_tuple_associativity(self):
        df = make_df()

        store_a = ClusterStateStore(
            df,
            aggregations={'fn_ln_dob': 'set_tuple'},
            sources={'fn_ln_dob': ('first_name', 'last_name', 'dob')},
        )
        store_a.commit_merge('r1', 'r2')
        store_a.commit_merge('r1', 'r3')

        store_b = ClusterStateStore(
            df,
            aggregations={'fn_ln_dob': 'set_tuple'},
            sources={'fn_ln_dob': ('first_name', 'last_name', 'dob')},
        )
        store_b.commit_merge('r2', 'r3')
        store_b.commit_merge('r1', 'r2')

        assert store_a.summary('r1')['fn_ln_dob'] == \
            store_b.summary('r1')['fn_ln_dob']
