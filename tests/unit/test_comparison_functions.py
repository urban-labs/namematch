from unittest.mock import patch
import warnings

import numpy as np

from namematch.generate_data_rows import GenerateDataRows
from namematch.comparison_functions import (
    get_name_probabilities,
    try_switch_first_last_name,
    compare_strings,
    compare_numbers,
    compare_categories,
    compare_dates,
    compare_geographies,
    generate_label,
)


def test_get_name_probabilities(an_df, side_by_side_df, params_and_schema, logger_for_testing):
    params, schema = params_and_schema

    # too many warnings
    warnings.filterwarnings("ignore")

    first_name_col = "first_name"
    last_name_col = "last_name"

    with patch('namematch.generate_data_rows.logger', logger_for_testing) as mock_debug:
        generate_data_rows = GenerateDataRows(
            params,
            schema,
            all_names_file=an_df,
            candidate_pairs_file=None,
            output_dir=None,
        )

        np_object = generate_data_rows.generate_name_probabilities_object(an_df, first_name_col, last_name_col)

    # get probabilities
    probs = get_name_probabilities(side_by_side_df, np_object, first_name_col, last_name_col)

    # test
    expected = [
        "prob_name_1",
        "prob_name_2",
        "prob_rev_name_1",
        "prob_rev_name_2",
        "count_pctl_name_1",
        "count_pctl_name_2",
        "count_pctl_fn_1",
        "count_pctl_fn_2",
        "count_pctl_ln_1",
        "count_pctl_ln_2",
        "prob_same_name",
        "prob_same_name_rev_1",
        "prob_same_name_rev_2"
        ]
    assert list(side_by_side_df) + expected == list(probs)

    for i, row in probs.iterrows():

        # this is true of the artificial data that is being used in this test
        # obviously it won't always be true

        if (row.first_name_1.startswith("J")):
            assert row.prob_rev_name_1 < row.prob_name_1
        else:
            assert row.prob_rev_name_1 > row.prob_name_1

        if (row.first_name_2.startswith("J")):
            assert row.prob_rev_name_2 < row.prob_name_2
        else:
            assert row.prob_rev_name_2 > row.prob_name_2


def test_try_switch_first_last_name(side_by_side_df_with_probs):
    # switch
    switched = try_switch_first_last_name(side_by_side_df_with_probs, "first_name", "last_name")

    # test
    for i, row in switched.iterrows():
        assert row.first_name_1.startswith("J")
        assert row.first_name_2.startswith("J")
        assert row.last_name_1.startswith("D")
        assert row.last_name_2.startswith("D")


def test_compare_strings(side_by_side_df):
    varname = "string"

    # compare
    compared = compare_strings(side_by_side_df, varname)

    # test
    assert 0 == compared.string_missing.sum()
    assert side_by_side_df.shape[0], compared.shape[0]
    assert (side_by_side_df.string_1 == side_by_side_df.string_2).sum() == compared.string_exact_match.sum()
    assert compared.string_exact_match.sum() < compared.string_soundex.sum()


def test_compare_numbers(side_by_side_df):
    df = side_by_side_df.astype({"number_1": str,"number_2": str})
    varname = "number"

    # compare
    compared = compare_numbers(df, varname)

    # test
    assert 0 == compared.number_missing.sum()
    assert df.shape[0] == compared.shape[0]


def test_compare_categories(side_by_side_df):
    side_by_side_df = side_by_side_df.fillna('')
    varname = "cat"

    # compare
    compared = compare_categories(side_by_side_df, varname)

    # test
    assert side_by_side_df.shape[0] == compared.shape[0]
    assert 0 == compared.cat_missing.isna().sum()
    assert 0 != compared.cat_exact_match.isna().sum()


def test_compare_dates(side_by_side_df):
    varname = "date"

    # compare
    compared = compare_dates(side_by_side_df, varname)

    # test
    assert side_by_side_df.shape[0] == compared.shape[0]
    assert 0 == compared.date_missing.isna().sum()
    assert 0 == compared.date_missing.sum()
    assert 0 == (compared.date_day_diff > 365).sum()


def test_compare_geographies(side_by_side_df):
    varname = "geo"

    # compare
    compared = compare_geographies(side_by_side_df, varname)

    # test
    assert side_by_side_df.shape[0] == compared.shape[0]
    assert 0 == compared.geo_missing.isna().sum()
    assert 0 == compared.geo_missing.sum()


def test_generate_label(ids_df):
    # the data is laid out in the following way:
    # uniqueid         ID        uid
    #    match      match      match
    #    match      match    missing
    #    match      match   mismatch
    #    match      match  nearmatch
    #    match  nearmatch   mismatch
    #  missing    missing  nearmatch

    uid_vars = ['uid']

    # test with no leven_thresh
    labels = generate_label(ids_df, uid_vars, leven_thresh=None)

    np.testing.assert_array_equal(['1', '', '0', '0', '0', '0'], labels.tolist())

    # test with leven_thresh
    labels = generate_label(ids_df, uid_vars, leven_thresh=1)
    np.testing.assert_array_equal(['1', '', '0', '', '0', ''], labels.tolist())

