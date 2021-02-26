import pandas as pd
import numpy as np
import pickle
import unittest

from collections import Counter
from unittest.mock import Mock

from namematch.block import *
from namematch.data_structures.schema import *
from namematch.data_structures.data_file import *
from namematch.data_structures.parameters import *
from namematch.data_structures.variable import *

logging_config = yaml.load(open('tests/logging_config.yaml', 'r'), Loader=yaml.FullLoader)
setup_logging(logging_config, None)
logger = logging.getLogger()
logging.disable(logging.CRITICAL)


class TestBlocking(unittest.TestCase):

    PATH = "tests/unit/data/"

    def test_get_blocking_columns(self):

        #get_blocking_columns(blocking_scheme)
        pass

    def test_read_an(self):

        #read_an(an_file, nn_cols, ed_col, absval_col)
        pass

    def test_get_nn_string_counts(self):

        #get_nn_string_counts(an)
        pass

    def test_get_common_name_penalties(self):

        # load fake data
        last_names = pd.read_csv(self.PATH + "an.csv").last_name
        max_penalty = 0.4
        bins = 3

        # get the threshholds
        common = get_common_name_penalties(last_names, max_penalty, bins)

        # test
        self.assertEqual(last_names.nunique(), len(common))
        self.assertEqual(bins, len(Counter(common.values())))
        for k in common.keys():
            self.assertTrue(common[k] <= max_penalty)

    def test_split_last_names(self):

        #split_last_names(df, last_name_column, blocking_scheme)
        pass

    def test_convert_all_names_to_blockstring_info(self):

        an = read_an(self.PATH + "an.parquet", ['first_name', 'last_name'], 'dob', None)

        params = Parameters({
                "last_name_column": 'last_name',
                "blocking_scheme": {
                    'cosine_distance' : { 'variables' : ['first_name', 'last_name'] },
                    'edit_distance' : { 'variable' : 'dob' }
                },
                'blocking_thresholds': {'common_name_max_penalty' : .1},
                'split_names' : True})

        nn_string_info, nn_string_expanded_df = convert_all_names_to_blockstring_info(an, None, params)

        self.assertEqual(nn_string_info[nn_string_info.nn_string == 'BEN::SMITH'].n_new.iloc[0], 2)
        self.assertEqual(len(nn_string_info), 19)
        self.assertEqual(len(nn_string_expanded_df), 21)

    def test_get_all_shingles(self):

        #get_all_shingles()
        pass

    def test_generate_shingles_matrix(self):

        nn_strings = ['JOHN::SMITH', 'TAYLOR::JOHNSON', 'JAMES::JOHNS']
        alpha = 0.5
        power = 1
        matrix_type = "test"
        verbose = False
        m = generate_shingles_matrix(nn_strings, alpha, power, matrix_type, verbose)

        # sum of m's rows should equal 1 + alpha (but also floating point error)
        summed = m.sum()
        self.assertAlmostEqual(summed, (1 + alpha) * len(nn_strings))

    def test_prep_index(self):

        #prep_index()
        pass

    def test_load_main_index_nn_strings(self):

        #load_main_index_nn_strings(og_blocking_index_file)
        pass

    def test_load_main_index(self):

        #load_main_index(index_file)
        pass

    def test_generate_index(self):

        #generate_index(nn_strings, num_threads, M, efC, post, alpha, power, print_progress=True)
        pass

    def test_get_second_index_nn_strings(self):

        all_strings = ["a", "b", "c"]
        main_strings = ["c"]
        expected_strings = ["a", "b"]
        strings = get_second_index_nn_strings(all_strings, main_strings)
        self.assertEqual(sorted(expected_strings), sorted(strings))

    def test_save_main_index(self):

        #save_main_index(main_index, main_index_nn_strings, main_index_file)
        pass

    def test_get_indices(self):

        #get_indices(params, all_nn_strings, og_blocking_index_file)
        pass

    def test_apply_blocking_filter(self):

        # load fake data
        df_exact = pd.read_csv(self.PATH + "apply_blocking_exact.csv")
        df_notexact = pd.read_csv(self.PATH + "apply_blocking_notexact.csv")
        full_info = pd.read_csv(self.PATH + "nn_string_full_df.csv").set_index("nn_string")
        ed_info = pd.read_csv(self.PATH + "nn_string_ed_string_df.csv").set_index("nn_string")
        expanded_info = full_info.join(ed_info) # TEMP
        with open(self.PATH + "thresholds.json") as j:
            thresholds = json.load(j)

        # exact matches
        out = apply_blocking_filter(df_exact, thresholds, expanded_info, nns_match=True)
        expected_names = ['blockstring_1', 'blockstring_2', 'cos_dist', 'edit_dist', 'covered_pair']
        self.assertEqual(expected_names, list(out))
        self.assertEqual(out.shape[0], out.covered_pair.sum())

        # not exact matches-- could be tested more exhaustively
        out = apply_blocking_filter(df_notexact, thresholds, expanded_info, nns_match=False)
        self.assertEqual(expected_names, list(out))
        self.assertTrue(df_notexact.shape[0] >= out.shape[0])

    def test_disallow_switched_pairs(self):

        #disallow_switched_pairs(df, incremental, nn_strings_to_query)
        pass

    def test_get_actual_candidates(self):

        # load fake data
        near_neighbors_df = pd.read_csv(self.PATH + "near_neighbors_df.csv")
        nn_string_full_df = pd.read_csv(self.PATH + "nn_string_full_df.csv").set_index("nn_string")
        nn_string_ed_string_df = pd.read_csv(self.PATH + "nn_string_ed_string_df.csv").set_index("nn_string")
        nn_strings_to_query = pd.read_csv(self.PATH + "nn_string_info.csv").nn_string.tolist()
        expanded_info = nn_string_full_df.join(nn_string_ed_string_df) # TEMP
        with open(self.PATH + "thresholds.json") as j:
            thresholds = json.load(j)
        start_ix_worker = 0
        end_ix_worker = near_neighbors_df.shape[0]
        incremental = False

        # get pairs
        cand_pairs = get_actual_candidates(
            near_neighbors_df.iloc[start_ix_worker : end_ix_worker],
            expanded_info,
            nn_strings_to_query,
            thresholds,
            incremental)

        # test
        self.assertTrue(cand_pairs.shape[0] <= near_neighbors_df.shape[0])

    def test_get_near_neighbors_df(self):

        # load fake data

        # changing this test involves re-calculating the near_neighbors_list using
        # functions like generate_indices and get_all_shingles and knnQueryBatch
        with open(self.PATH + "near_neighbors_list.pkl", "rb") as fp:
            near_neighbors_list = pickle.load(fp)
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
        near_neighbors_df = get_near_neighbors_df(
            near_neighbors_list,
            nn_string_info,
            nn_strings_this_index,
            nn_strings_queried_this_batch)

        # test
        self.assertEqual(k * len(nn_strings_queried_this_batch), near_neighbors_df.shape[0])
        js = {"JOHN::SMITH", "JON::SMITH"}
        azl = {"ABBY::LI", "ZABBIE::LI"}
        for (i, row) in near_neighbors_df.iterrows():
            if (row.nn_string_1 == row.nn_string_2):
                self.assertAlmostEqual(0, row.cos_dist, places=5)
            else:
                if (row.nn_string_1 in js):
                    self.assertTrue(row.nn_string_2 in js)
                elif (row.nn_string_2 in azl):
                    self.assertTrue(row.nn_string_2 in azl)

    def test_get_exact_match_candidate_pairs(self):

        #get_exact_match_candidate_pairs(
        #    nn_string_info_multi, nn_string_expanded_df, blocking_thresholds)
        pass

    def test_get_query_strings(self):

        #get_query_strings(nn_string_info, blocking_scheme)
        pass

    def test_write_some_cps(self):

        #write_some_cps(cand_pairs, output_file, header=False)
        pass

    def test_generate_candidate_pairs(self):

        # generate_candidate_pairs(
        #     params, nn_strings_to_query, shingles_to_query,
        #     nn_string_info, nn_string_expanded_df,
        #     main_index, main_index_nn_strings,
        #     second_index, second_index_nn_strings,
        #     output_file)
        pass

    def test_generate_true_pairs(self):

        #generate_true_pairs(must_links_df)
        pass

    def test_compute_cosine_sim(self):

        #compute_cosine_sim(blockstrings_in_pairs, pairs_df, shingles_matrix)
        pass

    def test_evaluate_blocking(self):

        #evaluate_blocking(cp_df, tp_df, blocking_scheme)
        pass

    def test_add_uncovered_pairs(self):

        #add_uncovered_pairs(candidate_pairs_df, uncovered_pairs_df)
        pass

    def test_pandas_default_min(self):

        # point is to test that min() default skipna=True doesn't change
        # for some reason, explicitely passing skinpna=True makes
        # the function a lot slower

        df = pd.DataFrame(data={
            'a':[1, 1, 2, 3, 4],
            'b':[np.NaN, 3, 1, 5, np.NaN]
        })
        df = df.groupby('a').b.min()

        self.assertEqual(df.isnull().sum(), 1)

