import os
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


class TestUtils(unittest.TestCase):

    PATH = "tests/unit/data/"

    def test_create_nm_record_id(self):
        pass

    def test_clean_nn_string(self):

        # test JR and .
        self.assertEqual(clean_nn_string("JOHN SMITH JR."), 'JOHN SMITH')

         # test STRIP, III, and -
        self.assertEqual(clean_nn_string(" JOHN  SM-ITH III"), 'JOHN SMITH')

        # test strip non A-Z (uppercase)
        self.assertEqual(clean_nn_string("JOHN Smith"), 'JOHN S')

        # test that uppercase isn't handled
        self.assertNotEqual(clean_nn_string("JOHN smith"), 'JOHN SMITH')


    def test_build_blockstring(self):

        # load fake data
        # (this function deals with it in all_names format)
        with open(self.PATH + "blocking_scheme.json") as j:
            scheme = json.load(j)
        df = pd.read_csv(self.PATH + "an.csv")

        # get blockstrings
        blockstring_col = build_blockstring(df, scheme)

        # test
        self.assertEqual(df.shape[0], len(blockstring_col))
        expected_num_cols = len(scheme['cosine_distance']['variables'] + [scheme['edit_distance']['variable']])
        num_cols = len(blockstring_col[0].split("::"))
        self.assertEqual(expected_num_cols, num_cols)


    def test_determine_model_to_use(self):
        pass


    def test_get_nn_string_from_blockstring(self):
        pass


    def test_get_ed_string_from_blockstring(self):
        pass


    def test_get_endpoints(self):
        pass


