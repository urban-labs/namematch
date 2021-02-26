import argparse
import csv
import logging
import numpy as np
import os
import pandas as pd
import sys
from datetime import datetime
import yaml

from namematch.data_structures.schema import Schema
from namematch.data_structures.parameters import Parameters
from namematch.utils import utils

logger = logging.getLogger()

try:
    profile
except:
    from line_profiler import LineProfiler
    profile = LineProfiler()


def main__process_config(config_file, nm_code_dir, logging_params, output_dir):
    '''Initialize and validate the parameters and matching schema using both the 
    user-input config and the private config. 

    Args: 
        config_file (str): path to user-input config file
        nm_code_dir (str): path to Name Match source code
        logging_params (dict): dictionary of logging parameter values
        output_dir (str): path to Name Match's output_temp folder
    '''

    # logger.stat(f'start: "{datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}"')

    private_config_file = os.path.join(args.nm_code_dir, 'utils/private_config.yaml')

    private_config = yaml.load(open(private_config_file, 'r'), Loader=yaml.FullLoader)

    try:
        # load config file
        config = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)
    except:
        logger.error("Could not load configuration file. "
                     "Please specify a path to valid yaml file.")
        raise


    params = Parameters.init(
             config,
             private_config,
             )

    schema = Schema.init(config, params)
    schema.write(os.path.join(output_dir, 'schema.yaml'))

    params.validate(schema.variables)
    params.write(os.path.join(output_dir, 'parameters.yaml'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')
    parser.add_argument('--nm_code_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--log_file')
    args = parser.parse_args()

    logging_params = utils.load_yaml(os.path.join(args.nm_code_dir, 'utils/logging_config.yaml'))
    utils.setup_logging(logging_params, args.log_file)
    logging_params['filters']['stat_filter']['()'] = 'StatLogFilter'
    logger = logging.getLogger()

    logger.debug("Start: process_config")
    main__process_config(args.config_file,
         args.nm_code_dir,
         logging_params,
         args.output_dir)
    logger.debug("End: process_config")
