import pandas as pd
import numpy as np
import pickle
import pytest

from collections import Counter
from unittest.mock import patch

from namematch.block import (
    Block,
    get_common_name_penalties,
    read_an,
    get_second_index_nn_strings,
)

from namematch.data_structures.parameters import Parameters


def test_get_params_and_schema(params_and_schema):
    params, schema = params_and_schema

    print(params)
    print(schema)
    assert params.verbose == 50000

def test_get_blocking_columns():

    #get_blocking_columns(blocking_scheme)
    pass


def test_read_an():

    #read_an(an_file, nn_cols, ed_col, absval_col)
    pass


def test_get_nn_string_counts():

    #get_nn_string_counts(an)
    pass


def test_get_common_name_penalties(an_df, logger_for_testing):

    # load fake data
    last_names = an_df.last_name
    max_penalty = 0.4
    bins = 3

    # get the threshholds
    with patch('namematch.block.logger', logger_for_testing) as mock_debug:
        common = get_common_name_penalties(last_names, max_penalty, bins)

    # test
    assert last_names.nunique() == len(common)
    assert bins == len(Counter(common.values()))
    for k in common.keys():
        assert common[k] <= max_penalty


def test_read_all_names_parquet(all_names_parquet_file):
    an = read_an(all_names_parquet_file, ['first_name', 'last_name'], 'dob', None)
    assert an.shape == (23, 9)


def test_split_last_names(params_and_schema, all_names_parquet_file, logger_for_testing):
    params, schema = params_and_schema
    an = read_an(all_names_parquet_file, ['first_name', 'last_name'], 'dob', None)

    with patch('namematch.process_input_data.logger', logger_for_testing) as mock_debug:
        block = Block(
            params,
            schema,
            all_names_file=all_names_parquet_file,
            must_links_file=None,
            og_blocking_index_file=None,
            candidate_pairs_file=None,
        )
        an_split_ln = block.split_last_names(an, 'last_name', params.blocking_scheme)

        assert "orig_last_name" in an_split_ln
        assert "orig_record" in an_split_ln


def test_convert_all_names_to_blockstring_info(params_and_schema, all_names_parquet_file, logger_for_testing):
    _, schema = params_and_schema
    an = read_an(all_names_parquet_file, ['first_name', 'last_name'], 'dob', None)

    with patch('namematch.block.logger', logger_for_testing) as mock_debug:
        params = Parameters({
                "last_name_column": 'last_name',
                "blocking_scheme": {
                    'cosine_distance' : { 'variables' : ['first_name', 'last_name'] },
                    'edit_distance' : { 'variable' : 'dob' }
                },
                'blocking_thresholds': {'common_name_max_penalty' : .1},
                'split_names' : True})

        block = Block(
            params,
            schema,
            all_names_file=all_names_parquet_file,
            must_links_file=None,
            og_blocking_index_file=None,
            candidate_pairs_file=None,
        )

        nn_string_info, nn_string_expanded_df = block.convert_all_names_to_blockstring_info(an, None, params)

        assert nn_string_info[nn_string_info.nn_string == 'BEN::SMITH'].n_new.iloc[0] == 2
        assert len(nn_string_info) == 19
        assert len(nn_string_expanded_df) == 21


def test_get_all_shingles():

    #get_all_shingles()
    pass


def test_generate_shingles_matrix(params_and_schema, all_names_parquet_file, logger_for_testing):
    params, schema = params_and_schema

    with patch('namematch.block.logger', logger_for_testing) as mock_debug:
        block = Block(
            params,
            schema,
            all_names_file=all_names_parquet_file,
            must_links_file=None,
            og_blocking_index_file=None,
            candidate_pairs_file=None,
        )

        nn_strings = ['JOHN::SMITH', 'TAYLOR::JOHNSON', 'JAMES::JOHNS']
        alpha = 0.5
        power = 1
        matrix_type = "test"
        verbose = False
        m = block.generate_shingles_matrix(nn_strings, alpha, power, matrix_type, verbose)

        # sum of m's rows should equal 1 + alpha (but also floating point error)
        summed = m.sum()
        assert pytest.approx(summed) == (1 + alpha) * len(nn_strings)


def test_prep_index():

    #prep_index()
    pass


def test_load_main_index_nn_strings():

    #load_main_index_nn_strings(og_blocking_index_file)
    pass


def test_load_main_index():

    #load_main_index(index_file)
    pass


def test_generate_index():

    #generate_index(nn_strings, num_workers, M, efC, post, alpha, power, print_progress=True)
    pass


def test_get_second_index_nn_strings():

    all_strings = ["a", "b", "c"]
    main_strings = ["c"]
    expected_strings = ["a", "b"]
    strings = get_second_index_nn_strings(all_strings, main_strings)
    assert sorted(expected_strings) == sorted(strings)


def test_save_main_index():

    #save_main_index(main_index, main_index_nn_strings, main_index_file)
    pass


def test_get_indices():

    #get_indices(params, all_nn_strings, og_blocking_index_file)
    pass


def test_apply_blocking_filter_exact(params_and_schema, blocking_exact_df, nn_string_full_df, nn_string_ed_string_df, thresholds_dict):

    expanded_info = nn_string_full_df.join(nn_string_ed_string_df) # TEMP
    # exact matches
    params, schema = params_and_schema
    out = Block(params, schema).apply_blocking_filter(blocking_exact_df, thresholds_dict, expanded_info, nns_match=True)
    expected_names = ['blockstring_1', 'blockstring_2', 'cos_dist', 'edit_dist', 'covered_pair']
    assert expected_names == list(out)
    assert out.shape[0] == out.covered_pair.sum()


def test_apply_blocking_filter_not_exact(params_and_schema, blocking_notexact_df, nn_string_full_df, nn_string_ed_string_df, thresholds_dict):
    # not exact matches-- could be tested more exhaustively
    expanded_info = nn_string_full_df.join(nn_string_ed_string_df)
    expected_names = ['blockstring_1', 'blockstring_2', 'cos_dist', 'edit_dist', 'covered_pair']
    params, schema = params_and_schema
    out = Block(params, schema).apply_blocking_filter(blocking_notexact_df, thresholds_dict, expanded_info, nns_match=False)
    assert expected_names == list(out)
    assert blocking_notexact_df.shape[0] >= out.shape[0]


def test_disallow_switched_pairs():

    #disallow_switched_pairs(df, incremental, nn_strings_to_query)
    pass


def test_get_actual_candidates(params_and_schema, near_neighbors_df, nn_string_full_df, nn_string_ed_string_df, nn_strings_to_query, thresholds_dict):
    expanded_info = nn_string_full_df.join(nn_string_ed_string_df) # TEMP
    start_ix_worker = 0
    end_ix_worker = near_neighbors_df.shape[0]
    incremental = False

    # get pairs
    params, schema = params_and_schema
    cand_pairs = Block(params, schema).get_actual_candidates(
        near_neighbors_df.iloc[start_ix_worker : end_ix_worker],
        expanded_info,
        nn_strings_to_query,
        thresholds_dict,
        incremental)

    # test
    assert cand_pairs.shape[0] <= near_neighbors_df.shape[0]


def test_get_near_neighbors_df(params_and_schema):
    # changing this test involves re-calculating the near_neighbors_list using
    # functions like generate_indices and get_all_shingles and knnQueryBatch
    near_neighbors_list = [
        (np.array([3, 4]), np.array([0., 0.18983728])),
        (np.array([4, 3]), np.array([1.1920929e-07, 1.8983728e-01])),
        (np.array([5, 6]), np.array([5.9604645e-08, 2.2517276e-01])),
        (np.array([6, 5]), np.array([0., 0.22517282])),
    ]

    k = len(near_neighbors_list[0][0])

    nn_string_info = pd.DataFrame({
        "nn_string": ["MALI::H", "ZUB::JELVEH", "J::SMITS", "JOHN::SMITH", "JON::SMITH", "ABBY::LI", "ZABBIE::LI"],
        "commonness_penalty": [0, 0, 0, 0, 0, 0, 0],
        "n_new": [3, 6, 1, 2, 6, 8, 9]
    })

    nn_strings_this_index = nn_string_info.nn_string.tolist()

    start_ix_batch = 3
    end_ix_batch = 8
    nn_strings_queried_this_batch = nn_strings_this_index[start_ix_batch:end_ix_batch]

    # get the df
    params, schema = params_and_schema
    near_neighbors_df = Block(params, schema).get_near_neighbors_df(
        near_neighbors_list,
        nn_string_info,
        nn_strings_this_index,
        nn_strings_queried_this_batch)

    # test
    assert k * len(nn_strings_queried_this_batch) == near_neighbors_df.shape[0]

    js = {"JOHN::SMITH", "JON::SMITH"}
    azl = {"ABBY::LI", "ZABBIE::LI"}
    for (i, row) in near_neighbors_df.iterrows():
        if (row.nn_string_1 == row.nn_string_2):
            assert 0 == pytest.approx(row.cos_dist, abs=1e-5)
        else:
            if (row.nn_string_1 in js):
                assert row.nn_string_2 in js
            elif (row.nn_string_2 in azl):
                assert row.nn_string_2 in azl


def test_get_exact_match_candidate_pairs():

    #get_exact_match_candidate_pairs(
    #    nn_string_info_multi, nn_string_expanded_df, blocking_thresholds)
    pass

def test_get_query_strings():

    #get_query_strings(nn_string_info, blocking_scheme)
    pass

def test_write_some_cps():

    #write_some_cps(cand_pairs, output_file, header=False)
    pass

def test_generate_candidate_pairs():

    # generate_candidate_pairs(
    #     params, nn_strings_to_query, shingles_to_query,
    #     nn_string_info, nn_string_expanded_df,
    #     main_index, main_index_nn_strings,
    #     second_index, second_index_nn_strings,
    #     output_file)
    pass

def test_generate_true_pairs():

    #generate_true_pairs(must_links_df)
    pass

def test_compute_cosine_sim():

    #compute_cosine_sim(blockstrings_in_pairs, pairs_df, shingles_matrix)
    pass

def test_evaluate_blocking():

    #evaluate_blocking(cp_df, tp_df, blocking_scheme)
    pass

def test_add_uncovered_pairs():

    #add_uncovered_pairs(candidate_pairs_df, uncovered_pairs_df)
    pass

def test_pandas_default_min():

    # point is to test that min() default skipna=True doesn't change
    # for some reason, explicitely passing skinpna=True makes
    # the function a lot slower

    df = pd.DataFrame(data={
        'a':[1, 1, 2, 3, 4],
        'b':[np.NaN, 3, 1, 5, np.NaN]
    })
    df = df.groupby('a').b.min()

    assert df.isnull().sum(), 1

