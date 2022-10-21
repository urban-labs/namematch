from unittest.mock import patch

import pandas as pd
import numpy as np
from namematch.cluster import Cluster, Constraints


def test_get_initial_clusters(params_and_schema, all_names_clustering_df, must_links_df, logger_for_testing):
    # test from scratch
    params, schema = params_and_schema
    with patch('namematch.cluster.logger', logger_for_testing) as mock_debug:
        cluster = Cluster(
                params,
                schema,
                constraints=None,
                must_links_file=None,
                potential_edges_dir=None,
                flipped0_edges_file=None,
                all_names_file=None,
                cluster_assignments=None,
        )
        eid_col = None

        clusters, cluster_assignments, original_cluster_ids = \
                cluster.get_initial_clusters(must_links_df, all_names_clustering_df, eid_col)

        assert original_cluster_ids is None
        assert 3 == len(clusters)
        assert {"A", "B", "C", "D", "E", "F", "G", "H"} == set(cluster_assignments.keys())

        # test incremental

        # TODO


def test_auto_is_valid_edge(params_and_schema):

    edges_df = pd.DataFrame({
        'ir_1':['12345', np.NaN, '45678', '343252'],
        'ir_2':['12346', np.NaN, '45678', '982834']
    })
    params, schema = params_and_schema
    auto_valid_potential_edges = Cluster(params, schema).auto_is_valid_edge(
        edges_df.copy(), uid_cols=['ir'], allow_clusters_w_multiple_unique_ids=False,
        leven_thresh=None, eid_col=None)
    assert len(auto_valid_potential_edges) == 2

    auto_valid_potential_edges = Cluster(params, schema).auto_is_valid_edge(
        edges_df.copy(), uid_cols=['ir'], allow_clusters_w_multiple_unique_ids=False,
        leven_thresh=1, eid_col=None)
    assert len(auto_valid_potential_edges) == 3

    edges_df = pd.DataFrame({
        'ir_1':['12345', 'melissa', np.NaN, '45678', '343252', '17'],
        'ir_2':['12346', 'merissa', np.NaN, '45678', '982834', '17'],
        'ssn_1':['333', 'abc', 'mcne ill', '999', np.NaN, np.NaN],
        'ssn_2':['333', 'abc', 'mcneill', np.NaN, np.NaN, np.NaN],
    })
    auto_valid_potential_edges = Cluster(params, schema).auto_is_valid_edge(
        edges_df.copy(), uid_cols=['ir', 'ssn'], allow_clusters_w_multiple_unique_ids=False,
        leven_thresh=None, eid_col=None)
    assert len(auto_valid_potential_edges) == 4

    auto_valid_potential_edges = Cluster(params, schema).auto_is_valid_edge(
        edges_df.copy(), uid_cols=['ir', 'ssn'], allow_clusters_w_multiple_unique_ids=False,
        leven_thresh=1, eid_col=None)
    assert len(auto_valid_potential_edges) == 5


def test_auto_is_valid_cluster():
    pass


def test_get_potential_edges():
    pass


def test_load_cluster_info(all_names_parquet_file, params_and_schema):

    params, schema = params_and_schema

    def default_get_cols_used():
        return 'all'

    def no_get_cols_used():
        return {}

    def some_get_cols_used():
        return {'dob':'date'}

    default = Constraints()
    default.get_columns_used = default_get_cols_used

    no_cols = Constraints()
    no_cols.get_columns_used = no_get_cols_used

    some_cols = Constraints()
    some_cols.get_columns_used = some_get_cols_used

    # test default get_columns_used() function
    cluster_info = Cluster(params, schema).load_cluster_info(
        all_names_parquet_file, uid_cols=['uid'], eid_col=None, cluster_logic=default)
    assert cluster_info.shape[1] == 12
    assert (cluster_info.dtypes == 'object').all()

    # test empty get_columns_used() function
    cluster_info = Cluster(params, schema).load_cluster_info(
        all_names_parquet_file, uid_cols=['uid'], eid_col=None, cluster_logic=no_cols)
    assert cluster_info.shape[1] == 2
    assert (cluster_info.dtypes == 'object').all()

    # test standard get_columns_used() function
    cluster_info = Cluster(params, schema).load_cluster_info(
        all_names_parquet_file, uid_cols=['uid'], eid_col=None, cluster_logic=some_cols)
    assert cluster_info.shape[1] == 3
    assert (cluster_info.dtypes == 'object').sum() == 2
    assert (cluster_info.dtypes == 'datetime64[ns]').sum() == 1


def test_cluster_potential_edges():
    pass


def test_pandas_sorting_default():

    df = pd.DataFrame(data={
        'a':[1, 2, 3, 4, 5],
        'b':[1, .99, .98, .97, .8]
    })
    df_dict_list = df.to_dict('records')

    a_vec = [d['a'] for d in df_dict_list]
    b_vec = [d['b'] for d in df_dict_list]

    assert df.a.tolist() == a_vec
    assert df.b.tolist() == b_vec


