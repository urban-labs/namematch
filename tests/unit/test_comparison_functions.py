import json
import numpy as np
import pandas as pd
import pickle
import random
import unittest
import warnings

from NameProbability import NameMatcher
from unittest.mock import Mock

from namematch.generate_data_rows import generate_name_probabilities_object
from namematch.comparison_functions import *
from namematch.data_structures.parameters import *
from namematch.data_structures.variable import *

logging_config = yaml.load(open('tests/logging_config.yaml', 'r'), Loader=yaml.FullLoader)
setup_logging(logging_config, None)
logger = logging.getLogger()
logging.disable(logging.CRITICAL)


class TestComparisonFunctions(unittest.TestCase):

    PATH = "tests/unit/data/"

    def test_get_name_probabilities(self):

        # too many warnings
        warnings.filterwarnings("ignore")

        first_name_col = "first_name"
        last_name_col = "last_name"

        # load fake data
        an = pd.read_csv(self.PATH + 'an.csv')
        np_object = generate_name_probabilities_object(an, first_name_col, last_name_col)
        df = pd.read_csv(self.PATH + "side_by_side_df.csv")

        # get probabilities
        probs = get_name_probabilities(df, np_object, first_name_col, last_name_col)

        # test

        expected = [
            "prob_name_1",
            "prob_name_2",
            "prob_rev_name_1",
            "prob_rev_name_2",
            "count_pctl_name_1",
            "count_pctl_name_2",
            "prob_same_name",
            "prob_same_name_rev_1",
            "prob_same_name_rev_2"
            ]
        self.assertEqual(list(df) + expected, list(probs))

        for i, row in probs.iterrows():

            # this is true of the artificial data that is being used in this test
            # obviously it won't always be true

            if (row.first_name_1.startswith("J")):
                self.assertTrue(row.prob_rev_name_1 < row.prob_name_1)
            else:
                self.assertTrue(row.prob_rev_name_1 > row.prob_name_1)

            if (row.first_name_2.startswith("J")):
                self.assertTrue(row.prob_rev_name_2 < row.prob_name_2)
            else:
                self.assertTrue(row.prob_rev_name_2 > row.prob_name_2)


    def test_try_switch_first_last_name(self):

        # load fake data
        df = pd.read_csv(self.PATH + "side_by_side_df_with_probs.csv")

        # switch
        switched = try_switch_first_last_name(df, "first_name", "last_name")

        # test
        for i, row in switched.iterrows():
            self.assertTrue(row.first_name_1.startswith("J"))
            self.assertTrue(row.first_name_2.startswith("J"))
            self.assertTrue(row.last_name_1.startswith("D"))
            self.assertTrue(row.last_name_2.startswith("D"))


    def test_compare_strings(self):

        # load fake data
        df = pd.read_csv(self.PATH + "side_by_side_df.csv")
        varname = "string"

        # compare
        compared = compare_strings(df, varname)

        # test
        self.assertEqual(0, compared.string_missing.sum())
        self.assertEqual(df.shape[0], compared.shape[0])
        self.assertEqual((df.string_1 == df.string_2).sum(), compared.string_exact_match.sum())
        self.assertTrue(compared.string_exact_match.sum() < compared.string_soundex.sum())


    def test_compare_numbers(self):

        # load fake data
        df = pd.read_csv(
            self.PATH + "side_by_side_df.csv",
            dtype={
                "number_1": str,
                "number_2": str
            }
        )
        varname = "number"

        # compare
        compared = compare_numbers(df, varname)

        # test
        self.assertEqual(0, compared.number_missing.sum())
        self.assertEqual(df.shape[0], compared.shape[0])


    def test_compare_categories(self):

        # load fake data
        df = pd.read_csv(
            self.PATH + "side_by_side_df.csv",
            keep_default_na=False)
        varname = "cat"

        # compare
        compared = compare_categories(df, varname)

        # test
        self.assertEqual(df.shape[0], compared.shape[0])
        self.assertEqual(0, compared.cat_missing.isna().sum())
        self.assertNotEqual(0, compared.cat_exact_match.isna().sum())


    def test_compare_dates(self):

        # load fake data
        df = pd.read_csv(self.PATH +"side_by_side_df.csv")
        varname = "date"

        # compare
        compared = compare_dates(df, varname)

        # test
        self.assertEqual(df.shape[0], compared.shape[0])
        self.assertEqual(0, compared.date_missing.isna().sum())
        self.assertEqual(0, compared.date_missing.sum())
        self.assertEqual(0, (compared.date_day_diff > 365).sum())


    def test_compare_geographies(self):

        # load fake data
        df = pd.read_csv(self.PATH + "side_by_side_df.csv")
        varname = "geo"

        # compare
        compared = compare_geographies(df, varname)

        # test
        self.assertEqual(df.shape[0], compared.shape[0])
        self.assertEqual(0, compared.geo_missing.isna().sum())
        self.assertEqual(0, compared.geo_missing.sum())


    def test_generate_label(self):

        # load fake data

        df = pd.read_csv(
            self.PATH + "ids.csv",
            dtype=object, na_filter=False
        )

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
        labels = generate_label(df, uid_vars, leven_thresh=None)

        np.testing.assert_array_equal(['1', '', '0', '0', '0', '0'], labels.tolist())

        # test with leven_thresh
        labels = generate_label(df, uid_vars, leven_thresh=1)
        np.testing.assert_array_equal(['1', '', '0', '', '0', ''], labels.tolist())

