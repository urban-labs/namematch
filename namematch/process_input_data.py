import logging
import numpy as np
import os
import pandas as pd
import yaml

import pyarrow as pa
import pyarrow.parquet as pq

from collections import defaultdict
from datetime import datetime
from streetaddress import StreetAddressParser

from namematch.base import NamematchBase
from namematch.data_structures.schema import Schema
from namematch.data_structures.parameters import Parameters
from namematch.utils.utils import (
    build_blockstring,
    clean_nn_string,
    create_nm_record_id,
    log_runtime_and_memory,
    load_yaml,
)
from namematch.utils.profiler import Profiler

profile = Profiler()

logger = logging.getLogger()

class ProcessInputData(NamematchBase):
    def __init__(
        self,
        params,
        schema,
        all_names_file="all_names.parquet",
        *args,
        **kwargs
    ):
        super(ProcessInputData, self).__init__(params, schema, *args, **kwargs)

        self.all_names_file = all_names_file

    @property
    def output_files(self):
        return [
            self.all_names_file
        ]

    @log_runtime_and_memory
    def main(self, **kw):
        '''Follow the instructions in the schema and params objects to build the all-names file from
        the raw input file(s).

        Args:
            params (Parameters object): contains parameter values
            schema (Schema object): contains match schema info (files to match, variables to use, etc.)
            all_names_file (str): path to the all-names file
        '''
        self.stats_dict["start"] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        self.stats_dict.yaml_add_eol_comment("process_input_data", "start")

        if self.params.incremental:
            logger.info("This is an incremental run.")
            self.stats_dict["run_type"] = "incremental"

        else:
            logger.info("This is not an incremental run.")
            self.stats_dict["run_type"] = "from scratch"

        logger.trace("Start: process_input_data")

        data_file_list = self.schema.data_files.get_all_data_files()

        # store categorical variables for report
        cat_vars = [v.name for v in self.schema.variables.varlist if v.compare_type == "Categorical"]
        self.stats_dict["categorical_variables"] = cat_vars

        n_an_rows = 0
        n_valid_an_rows = 0
        for j, data_file in enumerate(data_file_list):

            logger.info(f'Reading {data_file.nickname} data.')

            raw_columns_data = self.schema.variables.get_columns_to_read(data_file)

            for i, df in enumerate(pd.read_csv(
                    data_file.filepath,
                    chunksize=self.params.input_data_batch_size,
                    usecols=raw_columns_data,
                    dtype=object,
                    encoding="ISO-8859-1", 
                    na_filter=False)):

                if len(df) == 0:
                    break

                if self.params.verbose != None:
                    logger.info(f'  Processing: read {(i+1) * self.params.input_data_batch_size} rows.')

                df = self.process_data(df, self.schema.variables, data_file, self.params)

                first_iteration = (os.path.exists(self.all_names_file) == False)
                if len(df) > 0:
                    n_an_rows += len(df)
                    n_valid_an_rows += sum(df.drop_from_nm == 0)

                    if i == 0 and j == 0:
                        table = pa.Table.from_pandas(df)
                        pqwriter = pq.ParquetWriter(self.all_names_file, table.schema)
                    else:
                        table = pa.Table.from_pandas(df, schema=table.schema)

                    pqwriter.write_table(table)

                    # NOTE: append is okay even for first iteration because always
                    #       deleting existing files above

            logger.info(f'Done writing {data_file.nickname} data.')

        if pqwriter:
            pqwriter.close()

        logger.info(f"Number of input records: {n_an_rows}")
        logger.info(f"Number of valid input records: {n_valid_an_rows}")
        self.stats_dict["n_an"] = n_an_rows
        self.stats_dict["n_valid_an"] = n_valid_an_rows
        logger.trace("End: process_input_data")

        if self.enable_lprof:
            self.write_line_profile_stats(profile.line_profiler)

    @profile
    def process_geo_column(self, df, variable):
        '''Take dataframe of geographic data (either in "lat,lon" format or in
        "lat", "lon" format) and ensure it has just one column.

        Args:
            df (pd.DataFrame): df of address input data (columns are strings)
            variable (Variable object): contains naming info for new geo column

        Returns:
            pd.Dataframe: DataFrame of clean geographic information for all_names file
        '''

        df = df.copy()
        variable = variable.copy()

        ncols = len(df.columns)

        if ncols == 2:
            # concatenate two values (either x and y or lat and long)
            # if either column is empty, set the new column to be empty
            col1 = df.columns.tolist()[0]
            col2 = df.columns.tolist()[1]
            df[variable.name] = ''
            df.loc[(df[col1] != '') & (df[col2] != ''), variable.name] = \
                    df[col1].astype(str).str.strip() + ',' + df[col2].astype(str).str.strip()

        elif ncols == 1:
            # must already have comma in row (e.g. "12254231,95236928")

            df[variable.name] = df[df.columns.tolist()[0]]
            df['no_comma'] = ((df[variable.name] != '') & \
                    (df[variable.name].str.contains(',') == False)).astype(int)
            if df.no_comma.sum() > 0:
                logger.trace(f'Setting {variable.name} variable to missing for '
                             f'{df.no_comma.sum()} rows (misspecified).')
            df.loc[df.no_comma == 1, variable.name] = ''

        else:
            logger.error(f"Variables of compare_type Geography take "
                         f"exactly 0, 1, or 2 column names.")
            raise ValueError

        return df[[variable.name]]


    @profile
    def parse_address(self, address):
        '''Parse an address string into distinct parts.

        Args:
            address (str): string of full address (e.g. 54 East 18th Rd.)

        Returns:
            tuple: (address number, street name, street suffix)
        '''

        add_map = defaultdict(str, {
            "ave": "avenue", "avenue": "avenue",
            "blvd": "boulevard", "boulevard": "boulevard",
            "ctr": "center", "center": "center",
            "cir": "circle", "circle": "circle",
            "ct": "court", "court": "court",
            "cv": "cove", "cove": "cove",
            "dr": "drive", "drive": "drive",
            "expy": "expressway", "expressway": "expressway",
            "hts": "heights", "heights": "heights",
            "hwy": "highway", "highway": "highway",
            "jct": "junction", "junction": "junction",
            "ln": "lane", "lane": "lane",
            "lp": "loop", "loop": "loop",
            "pkwy": "parkway", "parkway": "parkway",
            "pl": "place", "place": "place",
            "rd": "road", "road": "road",
            "sq": "square", "square": "square",
            "st": "street", "street": "street"
        })
        # TODO need ter for terrace (but would have to change that in the street address parser...)
        # TODO need park (but would have to change that in the street address parser...)
        # TODO addresses with no street type (can lead to city name getting included in street name)
        # TODO should I not include street suffix as feature? don't really want two things to get
        #      "points" because both live on an avenue...

        address_parser = StreetAddressParser()
        address = address.lower()
        if address == 'redacted' or address == 'redact':
            address = ''
        ap = address_parser.parse(address)
        ap['house'] = '' if not ap['house'] else ap['house']
        ap['street_name'] = '' if not ap['street_name'] else ap['street_name']
        ap['street_type'] = '' if not ap['street_type'] else add_map[ap['street_type']]

        return ap['house'], ap['street_name'], ap['street_type']


    @profile
    def process_address_column(self, df, logger=None):
        '''Take dataframe of address data (either in "123 Main St." format or
        "123", "Main", "St." format, order matters) and parse as needed to produce
        three clean columns: street number, street name, and street type.

        Args:
            df (pd.DataFrame): df of address input data

        Returns:
            pd.DataFrame: Dataframe of clean address information for all_names file
        '''

        df = df.copy()

        ncols = len(df.columns)
        addr_cols = ['address_street_number', 'address_street_name', 'address_street_type']

        if ncols == 3:
            # simply rename columns
            df.columns = addr_cols

        elif ncols == 1:
            # parse address
            parsed_df = np.vectorize(self.parse_address)(df[df.columns.tolist()[0]].values)
            parsed_df = list(zip(*parsed_df))
            df[addr_cols] = pd.DataFrame(parsed_df, index=df.index)

        else:
            logger.error("Variables of compare_type Address take exactly 0, 1, or 3 column names.")
            raise ValueError

        return df[addr_cols]


    @profile
    def process_check(self, s, variable):
        '''Check the validity of the values in a given all-names column (according to the data type
        and config instructions) and set the series name correctly.

        Args:
            s (pd.Series): series to process (will be an all-names column)
            variable (Variable object): contains info on how to validate data in series

        Returns:
            pd.Series: Processed series
        '''

        s = s.copy()
        variable = variable.copy()

        if variable.compare_type != 'Address':
            s.name = variable.name

        if variable.check == 'Numeric':
            not_numeric = ((s != '') & (pd.to_numeric(s, errors='coerce')).isnull())
            if not_numeric.sum() > 0:
                logger.trace(f'Setting {s.name} variable to missing for {not_numeric.sum()} '
                             f'rows (fails check).')
            s.loc[not_numeric] = ''

        elif variable.check.startswith('Date'):
            before = s.isnull().sum()
            date_format = variable.check.split('-', 1)[1].strip()
            s = pd.to_datetime(s, format=date_format, errors="coerce")
            after = s.isnull().sum()
            if (before - after) > 0:
                logger.trace(f'Setting {s.name} variable to missing for '
                             f'{before - after} rows (fails check).')
            # convert to string in python date format: e.g. 2014-05-31
            s = s.dt.strftime('%Y-%m-%d')
            s = s.replace('NaT', '')
            s = s.replace(np.NaN, '')

        elif variable.compare_type == 'Categorical' and variable.check != '':
            options = variable.check.split(',')
            not_in_options = s[s.isin(options) == False].index.tolist()
            if len(not_in_options) > 0:
                logger.trace(f'Setting {s.name} variable to missing for '
                             f'{len(not_in_options)} rows (fails check).')
            s.loc[not_in_options] = ''

        else:
            try:
                s = s.str.strip()
                s = s.str.upper()
            except:
                raise

        return s

    @profile
    def process_data(self, df, variables, data_file, params):
        '''Read in part of an input file and process it according to the config in order
        to create part of the all-names file.

        Args:
            df (pd.DataFrame): chunk of an input data file
            variables (VariableList object): contains info about the fields for matching (from config)
            data_file (DataFile object): contains info about the input data set
            params (dict): dictionary of param values

        Returns:
            pd.DataFrame: a chunk of the all-names table (one row per input record)

            =====================   =======================================================
            record_id               unique record identifier
            file_type               either "new" or "existing"
            <fields for matching>   both for the matching model and for constraint checking
            <raw name fields>       pre-cleaning version of first and last name
            blockstring             concatenated version of blocking columns (sep by ::)
            drop_from_nm            flag, 1 if met any "to drop" criteria 0 otherwise
            =====================   =======================================================
        '''

        df = df.copy()
        variables = variables.copy()
        data_file = data_file.copy()
        params = params.copy()

        if data_file.use_record_id_as_is:
            df['record_id'] = df[data_file.record_id_col]
        else:
            df['record_id'] = create_nm_record_id(
                    data_file.nickname,
                    df[data_file.record_id_col])
        df.set_index('record_id', inplace=True)

        # create a new data frame to store all the processed variables
        an = pd.DataFrame(index=df.index)
        an['file_type'] = data_file.file_type
        an['dataset'] = data_file.nickname

        # track the records that will need to be dropped-- don't drop until the end
        all_records_to_drop = []

        for variable in variables.varlist:

            relevant_columns = variable.get_columns_to_read(data_file.nickname)

            # handle case where data file doesn't contain column
            if (variable.colname_dict[data_file.nickname] == ''):
                if (variable.compare_type == "Address"):
                    an["address_street_number"] = ""
                    an["address_street_name"] = ""
                    an["address_street_type"] = ""
                else:
                    an[variable.name] = ''
                continue

            # separate or combine columns as needed
            if (variable.compare_type == 'Geography'):
                relevant_df = self.process_geo_column(df[relevant_columns], variable)
            elif (variable.compare_type == 'Address'):
                relevant_df = self.process_address_column(df[relevant_columns])
            else:
                # if the compare_type isn't Geography or Address, there should only be one column
                if (len(relevant_columns) > 1):
                    logger.error(f'Only variables of compare_type "Geography" and "Address" '
                                 f'can take multiple column names.')
                    raise ValueError
                relevant_df = df[[relevant_columns[0]]]

            # NOTE: the only variable that should cause relevant_df to
            # have more than one column is Address
            for col in relevant_df:

                s = relevant_df[col].copy()
                raw_s = s.copy()

                # clean up last names
                if variable.name == params.last_name_column:
                    s = s.str.replace("MC ", "MC")
                    s = s.str.replace("-", " ")

                if len(variable.set_missing_list) > 0:
                    s = process_set_missing(s, variable.set_missing_list)

                s = self.process_check(s, variable)

                # clean up strings for blocking
                if variable.name in params.blocking_scheme['cosine_distance']['variables']:
                    an[f'tmp_raw__{s.name}'] = raw_s
                    s = s.apply(clean_nn_string)

                # NOTE: always drop last (in case other processing causes
                # value to go into the drop_list)
                if len(variable.drop_list) > 0:
                    records_to_drop = process_drop(s, variable.drop_list)
                    all_records_to_drop.extend(records_to_drop)

                an[s.name] = s

        # clean up the index
        an.reset_index(inplace=True)

        # process auto_drops
        auto_drop_records = process_auto_drops(an, all_records_to_drop, params.auto_drop_logic)
        all_records_to_drop = all_records_to_drop + auto_drop_records

        an['drop_from_nm'] = an.record_id.isin(all_records_to_drop).astype(int)
        an['blockstring'] = build_blockstring(an, params.blocking_scheme)

        return an

def process_set_missing(s, set_missing_list):
    '''Set values in a series to missing as needed.

    Args:
        s (pd.Series): strings to process
        set_missing_list (list): list of strings that are disallowed

    Returns:
        pd.Series: Processed series
    '''

    s = s.copy()
    missing = s.isin(set_missing_list)
    if (missing.sum() > 0):
        logger.trace(f'Setting {s.name} variable to missing for {missing.sum()} '
                     f'rows (in set_missing).')
    s.loc[missing] = ''

    return s


def process_drop(s, drop_list):
    '''Get the records in a series that have invalid values.

    Args:
        s (pd.Series): series being processed
        drop_list (list of str): invalid values

    Returns:
        list: Indices of records that are not valid
    '''

    s = s.copy()
    records_to_drop = s[s.isin(drop_list)].index.tolist()
    if (len(records_to_drop) > 0):
        logger.trace(f'Adding {len(records_to_drop)} rows to drop list because '
                     f'of the {s.name} variable.')

    return records_to_drop


def process_auto_drops(an, existing_drop_list, drop_logic):
    '''Get the records in all-names that have invalid values due to combination
    of multiple columns (based on logic in the private config).

    Args:
        an (pd.DataFrame): all-names chunk being processed
        existing_drop_list (list of str): records already known to be invalid
        drop_logic (list of dicts): logic for what makes a record invalid

    Returns:
        list: Indices of records that are not valid
    '''

    an = an.copy()
    existing_drop_list = existing_drop_list[:]
    drop_logic = drop_logic[:]

    an = an[an.record_id.isin(existing_drop_list) == False]

    to_drop_set = set()

    for invalid_reason in drop_logic:

        an['invalid'] = 1
        for col, value in invalid_reason.items():
            an.loc[an[col] != value, 'invalid'] = 0

        to_drop_set.update(an[an.invalid == 1].record_id.tolist())

    auto_drop_records = list(to_drop_set)

    return auto_drop_records



