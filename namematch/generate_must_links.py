import csv
import logging
import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq

from namematch.base import NamematchBase
from namematch.data_structures.schema import Schema
from namematch.data_structures.parameters import Parameters
from namematch.utils.utils import log_runtime_and_memory, load_yaml
from namematch.utils.profiler import Profiler

profile = Profiler()
logger = logging.getLogger()


class GenerateMustLinks(NamematchBase):
    def __init__(
        self,
        params,
        schema,
        all_names_file,
        must_links,
        *args,
        **kwargs
    ):

        super(GenerateMustLinks, self).__init__(params, schema, *args, **kwargs)
        self.must_links = must_links
        self.all_names_file = all_names_file

    @property
    def output_files(self):
        return [
            self.must_links
        ]

    # @log_runtime_and_memory
    def main(self, **kw):
        '''Generate the list of must-link pairs using UniqueID info .

        Args:
            params (Parameters object): contains parameter values
            schema (Schema object): contains match schema info (files to match, variables to use, etc.)
            all_names_file (str): path to the all-names file
            must_links (str): path to the must-links file
        '''
        logger.info('Generating "must-link" record pairs.')
        # get UniqueID variables
        uid_vars_list = self.schema.variables.get_variables_where(
                attr='compare_type', attr_value='UniqueID')

        # get records with non-missing unique identifiers
        ml_var_df = self.build_ml_var_df(
                self.all_names_file, uid_vars_list)

        # get the "must-link" record pairs
        must_links_df = self.get_must_links(
                ml_var_df, uid_vars_list)

        # true record pairs
        must_links_df.to_parquet(
            self.must_links,
            index=False
        )

        if self.enable_lprof:
            self.write_line_profile_stats(profile.line_profiler)

    def build_ml_var_df(self, all_names_file, uid_vars_list, **kw):
        '''Load the all-names file and limit it to the rows that have
        a non-missing UniqueID value.

        Args:
            all_names_file (str): path to the all-names file
            uid_vars_list (list of strings): all-name columns with compare_type "UniqueID"

        Returns:
            pd.DataFrame: a subset of the all-names file, relevant colums only

            ======================   =======================================================
            record_id                unique record identifier
            blockstring              concatenation of all the blocking variables (sep by ::)
            drop_from_nm             flag, 1 if met any "to drop" criteria 0 otherwise
            new_record               either True or False
            <UniqueID column(s)>     variables of compare_type UniqueID
            ======================   =======================================================
        '''

        cols = ['record_id', 'blockstring',  'drop_from_nm', 'file_type'] + uid_vars_list

        table = pq.read_table(all_names_file)
        an = table.to_pandas()[cols]

        an['new_record'] = an.file_type == 'new'

        an['has_ml_var'] = False
        for ml_var in uid_vars_list:
            an.loc[an[ml_var] != '', 'has_ml_var'] = True

        an = an[an.has_ml_var == True].drop(columns=['has_ml_var', 'file_type'])

        return an

    # @log_runtime_and_memory
    @profile
    def get_must_links(self, ml_var_df, uid_vars_list, **kw):
        '''Expand the list of records with must-link information to pairs of records
        that must be linked togehter in the final match.

        Args:
            ml_var_df (pd.DataFrame):
                ======================   =======================================================
                record_id                unique record identifier
                blockstring              concatenation of all the blocking variables (sep by ::)
                drop_from_nm             flag, 1 if met any "to drop" criteria 0 otherwise
                new_record               either True or False
                <UniqueID column(s)>     variables of compare_type UniqueID
                ======================   =======================================================

            uid_vars_list (list of strings): all-name columns with compare_type "UniqueID"

        Returns:
            pd.DataFrame: list of must-link record pairs

            ===================   =======================================================
            record_id_1           unique identifier for the first record in the pair
            record_id_2           unique identifier for the second record in the pair
            blockstring_1         blockstring for the first record in the pair
            blockstring_2         blockstring for the second record in the pair
            drop_from_nm_1        flag, True if the first record in the pair was not eligible for matching
            drop_from_nm_2        flag, True if the second record in the pair was not eligible for matching
            ===================   =======================================================
        '''

        ml_var_df = ml_var_df.copy()

        # for each UniqueID variable, merge the data frame on itself to get must-links
        must_link_df_list = []
        for ml_var in uid_vars_list:

            logger.info(f"Getting must-link pairs from {ml_var}...")

            # warn if any uids are used more than n times (might be
            # sign of misspecified missing uid)
            if ml_var in uid_vars_list:
                uid_counts = ml_var_df[ml_var_df[ml_var] != ''].groupby(ml_var).size()
                uid_counts_high = uid_counts[uid_counts > 200].index.tolist()
                if len(uid_counts_high) > 0:
                    logger.warning(f"The following {ml_var} values have over 200 unique "
                                   f"values; please ensure strings such as 'NA' are not "
                                   f"getting coded as values.")
                    logger.info(uid_counts_high)
                if uid_counts.max() > 1000:
                    logger.error("There are uids with more than 1000 records")
                    raise()

            ml_var_nonmissing_ix = ml_var_df[ml_var] != ''
            must_link_df = pd.merge(
                    ml_var_df.loc[ml_var_nonmissing_ix],
                    ml_var_df.loc[ml_var_nonmissing_ix],
                    on=ml_var, suffixes=['_1', '_2'])

            must_link_df = must_link_df[
                    (must_link_df.blockstring_1 < must_link_df.blockstring_2) |
                    ((must_link_df.blockstring_1 == must_link_df.blockstring_2) &
                     (must_link_df.record_id_1 < must_link_df.record_id_2))]

            # tweak for incremental matching
            if self.params.incremental:
                if ml_var in uid_vars_list:
                    must_link_df = must_link_df[
                            (must_link_df.new_record_1 == 1) |
                            (must_link_df.new_record_2 == 1)]

            must_link_df = must_link_df[[
                    'record_id_1', 'record_id_2', 'blockstring_1', 'blockstring_2',
                    'drop_from_nm_1', 'drop_from_nm_2']]

            must_link_df_list.append(must_link_df.copy())

        must_link_df = pd.concat(must_link_df_list)

        # remove duplicates (would occur if records match on multiple Unique IDs)
        len_pre_drop_duplicates = len(must_link_df)
        must_link_df = must_link_df.drop_duplicates(subset=['record_id_1', 'record_id_2'])
        len_post_drop_duplicates = len(must_link_df)
        if (len_pre_drop_duplicates > len_post_drop_duplicates):
            logger.trace(f'Dropped {len_pre_drop_duplicates - len_post_drop_duplicates} '
                         f'duplicate ground truth rows')

        return must_link_df
