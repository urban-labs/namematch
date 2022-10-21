import hashlib
import json
import logging.config
import logging
import os
import numpy as np
import pandas as pd
import pickle
import random
import re
import time
import yaml
import io

from collections.abc import Mapping
from functools import partial, wraps, update_wrapper

import pyarrow as pa
import pyarrow.parquet as pq

from ruamel.yaml.scalarstring import DoubleQuotedScalarString
from memory_profiler import memory_usage


class StatLogFilter():
    def __init__(self):
        self.__level = 100

    def filter(self, logRecord):
        return logRecord.levelno < self.__level


def setup_logging(log_params, log_filepath, output_temp_dir, filter_stats=False, logging_level='INFO'):
    """Setup logging configuration.

    Args:
        log_params (dict): contains info for logging setup
        log_filepath (str): path to store logs
    """

    if log_filepath is not None:
        log_params['handlers']['file_handler']['filename'] = log_filepath

    if not os.path.exists(os.path.dirname(log_filepath)):
        os.makedirs(os.path.dirname(log_filepath))

    # mappingproxy type is not hashable!
    log_params['filters']['stat_filter']['()'] = StatLogFilter

    if filter_stats:
        log_params['handlers']['file_handler']['filters'] = ['stat_filter']
        log_params['handlers']['console']['filters'] = ['stat_filter']

    runtime_num = 8
    logging.addLevelName(runtime_num, "RUNTIME")

    def runtime(self, message, *args, **kws):
        self._log(runtime_num, message, args, **kws)
    logging.Logger.runtime = runtime

    memory_num = 7
    logging.addLevelName(memory_num, "MEMORY")

    def memory(self, message, *args, **kws):
        self._log(memory_num, message, args, **kws)
    logging.Logger.memory = memory

    stat_num = 6
    logging.addLevelName(stat_num, "STAT")

    def stat(self, message, *args, **kws):
        self._log(stat_num, message, args, **kws)

    def stat_dict(self, message, *args, **kws):
        self._log(stat_num, yaml.dump(message, default_flow_style=False), args, **kws)
    logging.Logger.stat = stat
    logging.Logger.stat_dict = stat_dict

    trace_num = 5
    logging.addLevelName(trace_num, "TRACE")

    def trace(self, message, *args, **kws):
        self._log(trace_num, message, args, **kws)
    logging.Logger.trace = trace

    log_params['handlers']['console']['level'] = logging_level.upper()
    log_params['handlers']['file_handler']['level'] = logging_level.upper()
    logging.config.dictConfig(log_params)


def log_stat(human_desc, yaml_desc, value):
    '''Log a statistic in the log and in the stats yaml.

    Args:
        human_desc (str): human readable description of the stat (could be a phrase)
        yaml_desc (str): concise yaml-key compatible description of the stat
        value (float or str): value of the stat
    '''

    logger = logging.getLogger()
    logger.info(f'{human_desc}: {value}')


def log_runtime_and_memory(method):
    '''Decorator that logs time to execute functions and records max memory usage in GB.

    Args:
        method (function): function to measure/log runtime and memory usage

    Returns:
        value returned by the function being decorated
    '''
    @wraps(method)
    def inner_log_runtime_and_memory(*args, **kw):

        from datetime import timedelta
        logger = logging.getLogger()

        ts = time.time()
        memory_val, result = memory_usage(proc=(method, args, kw),
                retval=True, max_usage=True, max_iterations=1)
        te = time.time()

        elapsed_time = round(te - ts)
        elapsed_time_str = str(timedelta(seconds = elapsed_time))

        try:
            # for class method
            task = args[0]
            task_name = task.__module__.split('.')[1]
        except:
            # for normal function
            task_name = method.__module__

        if "main" in method.__name__:
            try:
                task.stats_dict[f"runtime__main__{task_name}"] = elapsed_time_str
                task.stats_dict[f"ram__main__{task_name}"] = f"{round(memory_val/1000., 2)} GB"

            except:
                raise Exception("Stats failed to record!")

        logger.runtime('%s: %s', f"runtime__{task_name}_{method.__name__}", elapsed_time_str)
        if type(memory_val) == list:
            memory_val = memory_val[0]
        logger.memory('%s: %s GB', f"memory__{task_name}_{method.__name__}", round(memory_val/1000., 2))

        return result

    return inner_log_runtime_and_memory


def load_yaml(yaml_file):
    '''Load a yaml file into a dictionary.

    Args:
        yaml_file (str): path to yaml file

    Returns:
        dict: dictionary version of input yaml file
    '''

    with open(yaml_file, 'r') as f:
        yaml_as_dict = yaml.load(f, Loader=yaml.FullLoader)

    return yaml_as_dict


def dump_yaml(dict_to_write, yaml_file):
    '''Write a dictionary into a yaml file.

    Args:
        dict_to_write (dict): dict to write to yaml
        yaml_file (str): path to output yaml file
    '''

    with open(yaml_file, 'w') as f:
        yaml.dump(dict_to_write, f, default_flow_style=False)


def to_dict(obj):
    '''Convert an object (i.e. instance of a user-defined class) into a dictionary
    to make writing easier.

    Args:
        obj (object): class instance to convert to dict
    '''
    if obj:
        return json.loads(json.dumps(obj, default=lambda o: o.__dict__))

    return {}

def create_nm_record_id(nickname, record_id_series):

    record_id_series = nickname + '__' + record_id_series
    return record_id_series


def clean_nn_string(n):
    '''Removes JR, SR, II, extra spaces, etc. from nn strings. The original string
    in the dataframe keeps punctuation and suffixes.

    Args:
        n (str): raw name value

    Returns:
        str: clean version of the input name
    '''

    # remove suffixes
    clean = re.sub("\\bSR\\b", "", n)
    clean = re.sub("\\bJR\\b", "", clean)
    clean = re.sub("\\bIII\\b", "", clean)
    clean = re.sub("\\bII\\b", "", clean)
    clean = re.sub("\\bIV\\b", "", clean)

    # drop all non letters and spaces
    regex = re.compile('[^A-Z ]')
    clean = regex.sub('', clean)

    # clean up spacing
    clean = clean.strip()
    clean = re.sub("\s+", " ", clean)
    return clean


def build_blockstring(df, blocking_scheme, incl_ed_string=True):
    '''Create blockstrings (values for blocking separated by ::, such as
    JOHN::SMITH::1993-07-23) from all-names data.

    Args:
        df (pd.DataFrame): all-names table

            =====================   =======================================================
            record_id               unique record identifier
            file_type               either "new" or "existing"
            <fields for matching>   both for the matching model and for constraint checking
            drop_from_nm            flag, 1 if met any "to drop" criteria 0 otherwise
            =====================   =======================================================

        blocking_scheme (dict): contains info about fields to block on
        incl_ed_string (bool): True if the blockstring should end with the edit-distance string (e.g. dob)

    Returns:
        pd.Series: blockstrings
    '''

    df = df.copy()

    blocking_cols = blocking_scheme['cosine_distance']['variables']
    if incl_ed_string:
        blocking_cols = \
                blocking_scheme['cosine_distance']['variables'] + \
                [blocking_scheme['edit_distance']['variable']]

    df['bs'] = ''
    for i, col in enumerate(blocking_cols):

        if i == 0:
            df['bs'] = df[col]
        else:
            df['bs'] = df.bs + '::' + df[col]

    return df['bs']


def get_nn_string_from_blockstring(blockstring):
    '''Parse out the near-neighbor string (e.g. first-name and last-name) from a blockstring.

    Args:
        blockstring (str): string with info for blocking (e.g. JOHN::SMITH::1993-07-23)

    Returns:
        str: near-neighbor string (e.g. JOHN::SMITH)
    '''

    return re.sub('::[A-Z0-9\-]*$', '', blockstring)


def get_ed_string_from_blockstring(blockstring):
    '''Parse out the edit-distance string (e.g. dob) from a blockstring.

    Args:
        blockstring (str): string with info for blocking (e.g. JOHN::SMITH::1993-07-23)

    Returns:
        str: edit-distance string (e.g. 1993-07-23)
    '''

    return re.sub('[A-Z ]*::[A-Z ]*::', '', blockstring)


def get_endpoints(n, num_chunks):
    '''Divide a number into some number of chunks/intervals.

    Args:
        n (int): number to divide into chunks/intervals
        num_chunks(int): number of chunks/intervals to create


    Returns:
        list of int tuples: list of start and end points to cover entire range
    '''

    end_points = []
    factor = int(n / num_chunks)
    for i in range(num_chunks):
        start_ix_worker = i * factor # create chunks for parallelization
        end_ix_worker = start_ix_worker + factor
        if i == num_chunks - 1:
            end_points.append([start_ix_worker, n])
        else:
            end_points.append([start_ix_worker, end_ix_worker])

    return end_points


def load_sample(csv_path, pct, cols=None):
    '''Load a random sample of a csv into pandas.

    Args:
        csv_path (str): path to csv file
        pct (float): what percent of the file to randomly read
        cols (list): columns to load

    Returns:
        pd.DataFrame: random subset of the input csv
    '''

    df_sample = pd.read_csv(csv_path, usecols=cols, header=0,
            skiprows=lambda i: i>0 and random.random() > pct)

    return df_sample


def load_csv_list(df_file_list, cols=None, conditions_dict={}, sample=1):
    '''Read a list of .csv files into a single pd.DataFrame.

    Args:
        df_file_list (list of str): list of .csv files to read
        cols (list): columns to keep in the dataframe
        conditions_dict (dict): conditions for row filtering
        sample (float): share of rows to randomly sample from the final dataframe

    Return:
        pd.DataFrame: filtered sampled dataframe read in from the .csv files
    '''

    all_df = pd.DataFrame()
    for df_file in df_file_list:

        if sample == 1:
            df = pd.read_csv(df_file, usecols=cols)
        else:
            df = load_sample(df_file, usecols=cols, pct=sample)

        for col, acceptable_value in conditions_dict.items():
            df = df[df[col] == acceptable_value]

        all_df = pd.concat([all_df, df])

    all_df = all_df.reset_index(drop=True)

    return all_df

def load_parquet(df_file, cols=None, conditions_dict={}):
    '''Read a .parquet file into a pd.DataFrame.

    Args:
        df_file (str): .parquet file to read
        cols (list): columns to keep in the dataframe
        conditions_dict (dict): conditions for row filtering

    Return:
        pd.DataFrame: filtered dataframe read in from the .parquet file
    '''
    filters = []

    if conditions_dict:
        for col, acceptable_value in conditions_dict.items():
            filters.append((col, '==', acceptable_value))
    else:
        filters = None

    pf = pq.read_table(df_file, columns=cols, filters=filters, use_threads=True)

    return pf.to_pandas()


def load_parquet_list(df_file_list, cols=None, conditions_dict={}, sample=1):
    '''Read a list of .parquet files into a single pd.DataFrame.

    Args:
        df_file_list (list of str): list of .parquet files to read
        cols (list): columns to keep in the dataframe
        conditions_dict (dict): conditions for row filtering
        sample (float): share of rows to randomly sample from the final dataframe

    Return:
        pd.DataFrame: filtered sampled dataframe read in from the .parquet files
    '''

    all_df = pd.DataFrame()
    for df_file in df_file_list:
        df = load_parquet(df_file, cols, conditions_dict)
        all_df = pd.concat([all_df, df])

    if sample == 1:
        return all_df.reset_index(drop=True)
    else:
        return all_df.sample(frac=sample).reset_index(drop=True)


def determine_model_to_use(dr_df, model_info, verbose=False):
    '''Assign a model to each data row based on which fields are available.

    Args:
        dr_df (pd.DataFrame): data rows

            ========================   ===============================================================
            record_id_1                unique identifier for the first record in the pair
            record_id_2                unique identifier for the second record in the pair
            <distance metric fields>   distance metrics between the two records' matching fields
            label                      flag, "1" if the records are a match, "0" if not, "" if unknown
            ========================   ===============================================================

        model_info (dict): information about models and their universes
        verbose (bool): flag controlling logging statement (set according to which function calls this one)

    Returns:
        pd.Series: string indicating which model to use for a given record pair
    '''

    logger = logging.getLogger()
    dr_df = dr_df.copy()

    # each data row gets assigned a model based on the values available in each field
    # TODO need to rank non-basic models if more than one and not mutually exclusive
    dr_df['model_to_use'] = 'basic'
    for this_model_name, this_model_info in model_info.items():

        # basic is the default
        if this_model_name == 'basic':
            continue

        if verbose:
            logger.debug(f'Determining rows for {this_model_name} model.')

        phat_universe = this_model_info["actual_phat_universe"]

        if len(phat_universe) == 0:
            logger.error("Special model types must have a limited actual_phat_universe.")
            raise ValueError

        acceptable = f'meets_criteria__{this_model_name}'
        dr_df[acceptable] = True
        for col, accepted_values in phat_universe.items():
            if not isinstance(accepted_values, list):
                accepted_values = [accepted_values]
            dr_df[acceptable] = (dr_df[acceptable]) & (dr_df[col].isin(accepted_values))

        dr_df.loc[dr_df[acceptable], 'model_to_use'] = this_model_name

    return dr_df['model_to_use']


def load_models(model_info_file, selection=False):
    '''Load pre-trained models (selection and match, as available)

    Args:
        model_info_file (str): path to original model config
        selection (bool): if True, try to load a corresponding selection model

    Returns:
        dict: maps model name (e.g. basic or no-dob) to a fit model object
        dict: dict with information about how to fit the model
    '''

    model_info = load_yaml(model_info_file)

    selection_models = {}
    match_models = {}
    for model_name, this_model_info in model_info.items():
        if selection:
            try:
                with open(this_model_info['selection_model_path'], 'rb') as mf:
                    selection_models[model_name] = pickle.load(mf)
            except:
                pass
        with open(this_model_info['match_model_path'], 'rb') as mf:
            match_models[model_name] = pickle.load(mf)

    if selection:
        return selection_models, match_models, model_info
    else:
        return match_models, model_info


def recursively_convert_tuple_to_list(value):
    if isinstance(value, Mapping):
        return dict((k, recursively_convert_tuple_to_list(v)) for k, v in value.items())

    elif isinstance(value, tuple):
        return list(recursively_convert_tuple_to_list(v) for v in value)
    return value


def luigi_dict_parameter_to_dict(d):
    out = {}
    for key, value in d.items():
        out[key] = recursively_convert_tuple_to_list(value)
    return out


def filename_friendly_hash(inputs):
    import datetime
    def dt_handler(x):
        if isinstance(x, datetime.datetime) or isinstance(x, datetime.date):
            return x.isoformat()
        raise TypeError("Unknown type")

    return hashlib.md5(
        json.dumps(inputs, default=dt_handler, sort_keys=True).encode("utf-8")
    ).hexdigest()


def load_logging_params(logging_params_file=None):
    default_logging_params_file = os.path.join(os.path.dirname(__file__), 'logging_config.yaml')
    logging_params_file = logging_params_file if logging_params_file else default_logging_params_file
    return yaml.load(open(logging_params_file, 'r'), Loader=yaml.FullLoader)


def reformat_dict(d: dict):
    '''make all the string values in the yaml file have double quotes'''
    d = d.copy()
    for k, v in d.items():
        if isinstance(v, str):
            d[k] = DoubleQuotedScalarString(v)
    return d


def camel_to_snake(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
