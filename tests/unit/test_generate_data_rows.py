import json
import pandas as pd
import unittest

from namematch.data_structures.data_file import *
from namematch.generate_data_rows import *
from namematch.data_structures.variable import *

logging_config = yaml.load(open('tests/logging_config.yaml', 'r'), Loader=yaml.FullLoader)
setup_logging(logging_config, None)
logger = logging.getLogger()
logging.disable(logging.CRITICAL)


class TestGenerateDataRows(unittest.TestCase):

    PATH = "tests/unit/data/"

    def test_generate_name_probabilities_object(self):

        prob_obj = generate_name_probabilities_object("fake_an_obj", None, "column_name")
        self.assertTrue(prob_obj is None)


    def test_find_valid_training_records(self):
        pass


    def test_generate_actual_data_rows(self):
        pass


    def test_generate_data_row_files(self):
        pass
