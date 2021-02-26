import pandas as pd
import numpy as np
import pickle
import unittest

from collections import Counter
from unittest.mock import Mock

from namematch.process_config import *
from namematch.data_structures.schema import *
from namematch.data_structures.parameters import *
from namematch.data_structures.data_file import *
from namematch.data_structures.variable import *
from namematch.utils.utils import load_yaml

logging_config = yaml.load(open('tests/logging_config.yaml', 'r'), Loader=yaml.FullLoader)
setup_logging(logging_config, None)
logger = logging.getLogger()
logging.disable(logging.CRITICAL)


class TestProcessConfig(unittest.TestCase):

    PATH = "tests/unit/data/"
    config_file = PATH + 'config.yaml'
    private_config_file = PATH + 'private_config.yaml'

    def test_params_init(self):

        config = load_yaml(self.config_file)
        logging_params = {}
        output_dir = self.PATH + "output/"

        # test passing
        params = Parameters.init(
            config,
            self.private_config_file,
            logging_params)
        schema = Schema.init(config, params)

        # test wrong set_missing type
        config['variables'][0]['drop'] = ','.join(config['variables'][0]['drop'])
        params = Parameters.init(
                config,
                self.private_config_file,
                logging_params)
        fine = 1
        try:
            schema = Schema.init(config, params)
        except:
            fine = 0
        self.assertTrue(fine == 0)


