import numpy as np
import random
import time
import unittest
import yaml

from unittest.mock import MagicMock


from namematch.fit_model import *

logging_config = yaml.load(open('tests/logging_config.yaml', 'r'), Loader=yaml.FullLoader)
setup_logging(logging_config, None)
logger = logging.getLogger()
logging.disable(logging.CRITICAL)


class TestModeling(unittest.TestCase):

    PATH = "tests/unit/data/"

    def test_get_feature_info(self):
        pass


    def test_fit_model(self):
        pass


    def test_fit_models(self):
        pass


    def test_define_necessary_models(self):

        # fake data
        dr_file_list = [self.PATH + "data_rows.parquet"]
        output_dir = ''
        missing_field = "first_name"

        # missing
        model_info = define_necessary_models(dr_file_list, output_dir, missing_field)
        self.assertEqual(2, len(model_info))
        self.assertTrue("no_{}".format(missing_field) in model_info)

        # not missing
        model_info = define_necessary_models(dr_file_list, output_dir, None)
        self.assertEqual(1, len(model_info))


    def test_get_match_train_eligible_flag(self):
        pass


    def test_load_model_data(self):
        pass


    def test_get_flipped0_potential_edges(self):
        pass


