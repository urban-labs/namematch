import pytest
import tempfile
import logging
import pickle

from contextlib import contextmanager

import pandas as pd

from namematch.data_structures.schema import Schema
from namematch.data_structures.parameters import Parameters
from namematch.utils.utils import setup_logging

from tests.utils import make_temp_parquet_file

PATH = "tests/unit/data/"

@pytest.fixture
def default_params_dict():
    return {
        # basic
        'verbose' : None,
        'num_workers' : 1,
        'parallelize' : False,
        # read-in rate (memory tradeoff)
        'input_data_batch_size': 50000,
        'data_rows_batch_size': 500,
        # variables
        'required_variables': ['first_name', 'last_name', 'dob'],
        'first_name_column': 'first_name',
        'last_name_column': 'last_name',
        'exact_match_variables': ['first_name', 'last_name', 'dob'],
        'negate_exact_match_variables' : ['middle_initial'],
        # data
        'split_names': True,
        'auto_drop_logic': [{'first_name': 'JOHN', 'last_name': 'DOE'},
                            {'first_name': 'JANE', 'last_name': 'DOE'}],
        # blocking
        'blocking_scheme': {'cosine_distance': {'variables': ['first_name','last_name']},
        'edit_distance': {'variable': 'dob'},
        'absvalue_distance': {'variable': 'age'},
        'alpha': 1.4,
        'power': 0.1},
        'blocking_thresholds': {
            'common_name_max_penalty': 0.1,
            'nodob_cosine_bar': 0.26,
            'high_cosine_bar': 0.3,
            'low_cosine_bar': 0.4,
            'high_editdist_bar': 1,
            'low_editdist_bar': 2,
            'absvalue_bar': 3
        },
        'index': {
            'rebuild_main_index': 'if_secondary_index_exceeds_limit',
            'secondary_index_limit': 500000
        },
        'nmslib': {'M': 100, 'efC': 1000, 'post': 0, 'efS': 750, 'k': 500},
        # modeling
        'pct_train' : 0.9,
        'use_uncovered_phats': False,
        'missingness_model': 'dob',
        'max_match_train_n': 3000000,
        'max_selection_train_eval_n': 1000000,
        'weight_using_selection_model': False,
        'default_threshold': 0.7,
        'optimize_threshold': True,
        'match_train_criteria': {'data_rows': {'covered_pair': 1}},
        'initialize_from_ground_truth_1s': True,
        'fscore_beta': 0.5,
        'allow_clusters_w_multiple_unique_ids': False
    }


@pytest.fixture
def config_dict():
    return {
        # input data files
        'data_files': {
            'raw_data': {
                'filepath': 'tests/unit/data/raw_data.csv',
                'record_id_col': 'row_id',
                'delim': ',',
                'cluster_type': 'cluster',
                'output_name': 'input_data'
            }
        },
        'variables': [
            {
                'name': 'first_name',
                'compare_type': 'String',
                'raw_data_col': 'firstname',
                'drop': ['', ' ', 'FIRM', 'BUSINESS', 'STATE OF', 'UNIVERSITY']
            },
            {
                'name': 'last_name',
                'compare_type': 'String',
                'raw_data_col': 'lastname'
            },
            {
                'name': 'dob',
                'compare_type': 'Date',
                'raw_data_col': 'birthdate',
                'check': 'Date - %m/%d/%Y'
            },
            {
                'name' : 'age',
                'compare_type': 'Numeric',
                'raw_data_col' : 'age_2021',
            },
            {
                'name': 'gender',
                'compare_type': 'Categorical',
                'raw_data_col': 'gender',
                'check': 'm,f'
            },
            {
                'name': 'address',
                'compare_type': 'Address',
                'raw_data_col': 'address'
            },
            {
                'name': 'uid',
                'compare_type': 'UniqueID',
                'raw_data_col': 'uid'}
        ],
        'verbose': 50000,
        'num_workers': 10,
        'allow_clusters_w_multiple_unique_ids': False,
        'leven_thresh': None,
        'pct_train': 0.6,
        'missingness_model': None
    }

@pytest.fixture
def logger_for_testing():
    logging_params_dict = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'simple': {'format': '%(levelname)-8s %(message)s'},
            'colored_console': {
                '()': 'coloredlogs.ColoredFormatter',
                'format': "%(asctime)s - %(levelname)-8s %(message)s",
            },
            'detailed': {
                'format': '%(asctime)s %(levelname)-8s %(message)s',
                'datefmt': '%m/%d/%Y %H:%M:%S'
            },
            'message': {'format': '%(message)s'}
        },
        'filters': {
            'stat_filter': {'()': 'StatLogFilter'}
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'colored_console',
            'stream': 'ext://sys.stdout'
            },
            'file_handler': {
                'class': 'logging.handlers.WatchedFileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'encoding': 'utf8'
            },
            'file_handler_stat_memory': {
                'class': 'logging.StreamHandler',
                'level': 'STAT',
                'formatter': 'message'
            }
        },
        'root': {
            'level': 'DEBUG',
            'handlers': ['console', 'file_handler']
        }
    }

    log_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.log', delete=False)
    setup_logging(logging_params_dict, log_file.name, tempfile.gettempdir())
    logging_params_dict['filters']['stat_filter']['()'] = 'StatLogFilter'
    logger = logging.getLogger()
    logging.disable(logging.CRITICAL)
    return logger


@pytest.fixture
def raw_data_df():
    df = pd.read_csv(PATH + "raw_data.csv")
    return df


@pytest.fixture
def an_df():
    df = pd.read_csv(PATH + "an.csv")
    return df


@pytest.fixture
def params_and_schema(config_dict, default_params_dict):
    # test passing
    params = Parameters.init(
        config_dict,
        default_params_dict)

    schema = Schema.init(config_dict, params)

    return params, schema


@pytest.fixture
def all_names_parquet_file():
    with make_temp_parquet_file(PATH + "an.csv") as an_fn:
        yield an_fn


@pytest.fixture
def blocking_exact_df():
    df_exact = pd.read_csv(PATH + "apply_blocking_exact.csv")
    return df_exact


@pytest.fixture
def blocking_notexact_df():
    df_notexact = pd.read_csv(PATH + "apply_blocking_notexact.csv")
    return df_notexact


@pytest.fixture
def nn_string_full_df():
    full_info = pd.read_csv(PATH + "nn_string_full_df.csv").set_index("nn_string")
    return full_info


@pytest.fixture
def nn_string_ed_string_df():
    ed_info = pd.read_csv(PATH + "nn_string_ed_string_df.csv").set_index("nn_string")
    return ed_info


@pytest.fixture
def thresholds_dict():
    return {
        "high_editdist_bar": 5,
        "low_editdist_bar": 0,
        "high_cosine_bar": 0.5,
        "nodob_cosine_bar": 0.5,
        "low_cosine_bar": 0,
        "absvalue_bar": 3,
        "common_name_max_penalty": 0.1
    }


@pytest.fixture
def near_neighbors_df():
    near_neighbors_df = pd.read_csv(PATH + "near_neighbors_df.csv")
    return near_neighbors_df


@pytest.fixture
def nn_strings_to_query():
    nn_strings_to_query = pd.read_csv(PATH + "nn_string_info.csv").nn_string.tolist()
    return nn_strings_to_query


@pytest.fixture
def data_rows_parquet_file():
    with make_temp_parquet_file(PATH + "data_rows.csv") as data_rows_fn:
        yield data_rows_fn


@pytest.fixture
def all_names_clustering_df():
    an = pd.read_csv(PATH + "all_names_clustering.csv")
    return an


@pytest.fixture
def must_links_df():
    ml = pd.read_csv(PATH + "gt_edges_clustering.csv")
    return ml


@pytest.fixture
def side_by_side_df():
    df = pd.read_csv(PATH + "side_by_side_df.csv")
    return df


@pytest.fixture
def side_by_side_df_with_probs():
    df = pd.read_csv(PATH + "side_by_side_df_with_probs.csv")
    return df


@pytest.fixture
def ids_df():
    df = pd.read_csv(PATH + "ids.csv",dtype=object, na_filter=False)
    return df
