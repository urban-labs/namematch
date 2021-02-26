import json
import pandas as pd
import unittest

from namematch.data_structures.schema import Schema
from namematch.data_structures.parameters import Parameters
from namematch.utils.utils import *

logging_config = yaml.load(open('tests/logging_config.yaml', 'r'), Loader=yaml.FullLoader)
setup_logging(logging_config, None)
logger = logging.getLogger()
logging.disable(logging.CRITICAL)


class TestGenerateMLPairs(unittest.TestCase):

    PATH = "tests/unit/data/"

    def test_build_ml_var_df(self):
        pass


    def test_get_must_links(self):
        pass

