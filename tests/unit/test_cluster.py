from unittest.mock import patch

import pandas as pd

from namematch.cluster import Cluster


def test_get_initial_clusters(params_and_schema, all_names_clustering_df, must_links_df, logger_for_testing):
    # test from scratch
    params, schema = params_and_schema
    with patch('namematch.cluster.logger', logger_for_testing) as mock_debug:
        cluster = Cluster(
                params,
                schema,
                constraints_file=None,
                must_links_file=None,
                potential_edges_dir=None,
                flipped0_edges_file=None,
                all_names_file=None,
                output_file=None,
        )
        eid_col = None

        clusters, cluster_assignments, original_cluster_ids = \
                cluster.get_initial_clusters(must_links_df, all_names_clustering_df, eid_col)

        assert original_cluster_ids is None
        assert 3 == len(clusters)
        assert {"A", "B", "C", "D", "E", "F", "G", "H"} == set(cluster_assignments.keys())

        # test incremental

        # TODO


def test_auto_is_valid_edge():
    pass


def test_auto_is_valid_cluster():
    pass


def test_get_potential_edges():
    pass


def test_load_cluster_info():
    pass


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


