import logging
import os

from collections import defaultdict
import multiprocessing as mp
from math import ceil

import numpy as np
import pandas as pd
from namematch.name_probability import nm_prob

import pyarrow.parquet as pq
import pyarrow as pa

from namematch.utils.utils import *
from namematch.comparison_functions import *
from namematch.data_structures.schema import Schema
from namematch.data_structures.parameters import Parameters
from namematch.base import NamematchBase
from namematch.utils.profiler import Profiler

profile = Profiler()
logger = logging.getLogger()


class GenerateDataRows(NamematchBase):
    def __init__(
        self,
        params,
        schema,
        output_dir,
        all_names_file,
        candidate_pairs_file,
        *args,
        **kwargs
    ):
        super(GenerateDataRows, self).__init__(params, schema, *args, **kwargs)

        self.all_names_file = all_names_file
        self.candidate_pairs_file = candidate_pairs_file
        self.output_dir = output_dir

    @property
    def output_files(self):
        return [os.path.join(self.output_dir, f'data_rows_{i}.parquet') for i in range(self.params.num_workers)]

    @log_runtime_and_memory
    def main(self, **kw):
        '''Take candidate pairs and merge on the all-names records (twice) to get a dataset at the
        record pair level. Compute distance metrics between the records in the pair -- these are the
        features for modeling.

        Args:
            params (Parameters object): contains parameter values
            schema (Schema object): contains match schema info (files to match, variables to use, etc.)
            all_names_file (str): path to the all-names file
            candidate_pairs_file (str): path to the candidate-pairs file
            output_dir (str): path to the data-rows dir
        '''

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        an_columns = ['blockstring'] + self.schema.variables.get_an_column_names()
        table = pq.read_table(self.all_names_file)
        an = table.to_pandas()[an_columns]

        an = an[an.drop_from_nm == 0]
        an = an.drop_duplicates(['record_id'])

        logger.info('Building name probability object.')
        name_probs = self.generate_name_probabilities_object(
                an, self.params.first_name_column, self.params.last_name_column)

        logger.info('Generating data rows...')

        # remove previous data rows files if they exist
        for dr_file in os.listdir(self.output_dir):
            os.remove(os.path.join(self.output_dir, dr_file))

        table = pq.read_table(self.candidate_pairs_file)
        cp_df = table.to_pandas()[['blockstring_1', 'blockstring_2', 'covered_pair']]

        # get data rows
        end_points = get_endpoints(len(cp_df), self.params.num_workers)
        if self.params.parallelize:
            jobs = [mp.Process(
                        target=self.generate_data_row_files,
                        args=(self.params,
                              self.schema,
                              an,
                              cp_df.iloc[end_point[0]:end_point[1]],
                              name_probs,
                              end_point[0],
                              end_point[1],
                              os.path.join(self.output_dir, f'data_rows_{i}.parquet'))
                    ) for i, end_point in enumerate(end_points)]

            for job in jobs:
                job.start()
            for job in jobs:
                t = job.join()
            failure_occurred = sum([job.exitcode != 0 for job in jobs])
            if failure_occurred:
                logger.error(f"Error occurred in {failure_occurred} worker(s).")
                raise Exception(f"Error occurred in {failure_occurred} worker(s).")

        else:

            for i, end_point in enumerate(end_points):
                self.generate_data_row_files(
                        self.params, self.schema,
                        an, cp_df.iloc[end_point[0]:end_point[1]], name_probs,
                        end_point[0], end_point[1],
                        os.path.join(self.output_dir, f'data_rows_{i}.parquet'))

        if self.enable_lprof:
            self.write_line_profile_stats(profile.line_profiler)

    @log_runtime_and_memory
    def generate_name_probabilities_object(self, an, fn_col=None, ln_col=None, **kw):
        '''The generate_name_probabilites function uses a list of names (from all_names
        file) and creates an object containing queryable probability
        information for each name.

        Args:
            an (pd.DataFrame): all-names, just the name columns
            fn_col (str): name of first name column
            ln_col (str): name of last name column

        Return:
            name probability object
        '''

        if (fn_col is None) or (ln_col is None):
            return None

        an['fn'] = an[fn_col].str.replace(' ', '')
        an['ln'] = an[ln_col].str.replace(' ', '')
        an['name_prob_str'] = '*' + an['fn'] + ' ' + an['ln'] + '*'

        np_object = nm_prob.NameProbability(name_list=an.name_prob_str.tolist())

        np_object.n_name_appearances_dict = an.groupby('name_prob_str').size().rank(pct=True, method='min').round(2).to_dict()
        np_object.n_firstname_appearances_dict = an.groupby('fn').size().rank(pct=True, method='min').round(2).to_dict()
        np_object.n_lastname_appearances_dict = an.groupby('ln').size().rank(pct=True, method='min').round(2).to_dict()

        return np_object

    @log_runtime_and_memory
    def find_valid_training_records(self, an, an_match_criteria, **kw):

        an['meets_match_criteria'] = 1
        for col, accepted_values in an_match_criteria.items():
            if not isinstance(accepted_values, list):
                accepted_values = [accepted_values]
            accepted_values = [str(av) for av in accepted_values]
            an['meets_match_criteria'] = (an['meets_match_criteria']) & (an[col].isin(accepted_values))

        return an['meets_match_criteria']

    @profile
    def generate_actual_data_rows(self, params, schema, sbs_df, np_object, first_iter):
        '''Create modeling dataframe by comparing each variable (via numerous distance metrics).

        Args:
            params (Parameters object): contains matching parameters
            schema (Schema object): contains matching schema (data files and variables)
            sbs_df (pd.DatFrame): side-by-side table (record pair level, with info from both an records)

                ==============================   =======================================================
                record_id (_1, _2)               unique record identifier
                blockstring (_1, _2)             concatenated version of blocking columns (sep by ::)
                file_type (_1, _2)               either "new" or "existing"
                candidate_pair_ix                index from candidate-pairs list
                covered_pair                     flag, 1 if blockstring pair passed blocking 0 otherwise
                <fields for matching> (_1, _2)   both for the matching model and for constraint checking
                ==============================   =======================================================
            np_object (nm_prob.NameProbability object): contains information about name probabilities

        Returns:
            pd.DataFrame: chunk of the data-rows file

            =====================   =======================================================
            dr_id                   unique record pair identifier (record_id_1__record_id_2)
            record_id (_1, _2)      unique record identifiers
            <distance metrics>      how similar are the different matching fields between recrods
            label                   "1" if the records refer to the same person, "0" if not, "" otherwise
            =====================   =======================================================
        '''

        sbs_df = sbs_df.copy()

        uid_cols = schema.variables.get_variables_where(attr='compare_type', attr_value='UniqueID')

        # don't compare a record to itself
        sbs_df = sbs_df[sbs_df.record_id_1 != sbs_df.record_id_2]

        # if blockstrings are same, don't want to compare A to B then B to A
        sbs_df = sbs_df[
                (sbs_df.blockstring_1 != sbs_df.blockstring_2) |
                (sbs_df.record_id_1 < sbs_df.record_id_2)]

        if params.incremental:

            # dont compare if both are from existing (this would work outside of
            # if statement, but would always be true -- do it here to save time)
            sbs_df = sbs_df[
                    (sbs_df.file_type_1 == 'new') |
                    (sbs_df.file_type_2 == 'new')]

        if params.drop_mixed_label_pairs:
            for uid_col in uid_cols:
                sbs_df = sbs_df[(sbs_df[f"{uid_col}_1"] == '') == (sbs_df[f"{uid_col}_2"] == '')]

        if len(sbs_df) == 0:
            return None

        non_feature_cols = ['candidate_pair_ix', 'record_id_1', 'record_id_2', 'covered_pair']

        data_rows_df = sbs_df[non_feature_cols].copy()

        data_rows_df.reset_index(drop=True, inplace=True)
        sbs_df.reset_index(drop=True, inplace=True)

        if np_object is not None:

            # get name probabilities (both as feature and for knowing which
            # name to switch if a switch is needed)
            sbs_df = get_name_probabilities(sbs_df, np_object,
                    params.first_name_column, params.last_name_column)

            # determine if name switch need to happen
            sbs_df = try_switch_first_last_name(sbs_df,
                    params.first_name_column, params.last_name_column)

        if any(data_rows_df.index != sbs_df.index):
            logger.error('Index mismatch when creating data rows.')
            raise ValueError

        # add name probability columns to data_rows_df
        try:
            #data_rows_df['prob_name1'] = sbs_df['prob_name_1']
            #data_rows_df.loc[(sbs_df.switched_name == 1), 'prob_name1'] = sbs_df.prob_rev_name_1
            #data_rows_df['prob_name2'] = sbs_df['prob_name_2']
            #data_rows_df.loc[(sbs_df.switched_name == 2), 'prob_name2'] = sbs_df['prob_rev_name_2']
            #data_rows_df['prob_same_name'] = sbs_df['prob_same_name']
            #data_rows_df.loc[sbs_df.switched_name == 1, 'prob_same_name'] = sbs_df['prob_same_name_rev_1']
            #data_rows_df.loc[sbs_df.switched_name == 2, 'prob_same_name'] = sbs_df['prob_same_name_rev_2']
            #data_rows_df['max_prob_name'] = data_rows_df[['prob_name1', 'prob_name2']].max(axis=1)
            #data_rows_df['count_pctl_name_1'] = sbs_df['count_pctl_name_1']
            #data_rows_df['count_pctl_name_2'] = sbs_df['count_pctl_name_2']
            #data_rows_df['max_count_pctl_name'] = data_rows_df[['count_pctl_name_1', 'count_pctl_name_2']].max(axis=1)
            data_rows_df['diff_count_pctl_name'] = (sbs_df['count_pctl_name_1'] - sbs_df['count_pctl_name_2']).abs()
            data_rows_df['max_count_pctl_name'] = sbs_df[['count_pctl_name_1', 'count_pctl_name_2']].max(axis=1)
            data_rows_df['diff_count_pctl_fn'] = (sbs_df['count_pctl_fn_1'] - sbs_df['count_pctl_fn_2']).abs()
            data_rows_df['max_count_pctl_fn'] = sbs_df[['count_pctl_fn_1', 'count_pctl_fn_2']].max(axis=1)
            data_rows_df['diff_count_pctl_ln'] = (sbs_df['count_pctl_ln_1'] - sbs_df['count_pctl_ln_2']).abs()
            data_rows_df['max_count_pctl_ln'] = sbs_df[['count_pctl_ln_1', 'count_pctl_ln_2']].max(axis=1)

        except:
            if first_iter:
                logger.info('No name probability features generated.')

        for variable in schema.variables.varlist:

            if variable.compare_type == 'String':
                features_df = compare_strings(sbs_df, variable.name)
            elif variable.compare_type == 'Numeric':
                features_df = compare_numbers(sbs_df, variable.name)
            elif variable.compare_type == 'Categorical':
                features_df = compare_categories(sbs_df, variable.name)
            elif variable.compare_type == 'Date':
                features_df = compare_dates(sbs_df, variable.name)
            elif variable.compare_type == 'Geography':
                features_df = compare_geographies(sbs_df, variable.name)
            elif variable.compare_type == 'Address':
                features_df_1 = compare_strings(sbs_df, 'address_street_number')
                features_df_2 = compare_strings(sbs_df, 'address_street_name')
                features_df_3 = compare_categories(sbs_df, 'address_street_type')
                features_df = pd.concat([features_df_1, features_df_2, features_df_3], axis=1)
            elif variable.compare_type == "LastName":
                ##features_df_1 = compare_strings(sbs_df, variable.name)
                ##features_df_2 = compare_last_name(sbs_df, variable.name)
                ##features_df = pd.concat([features_df_1, features_df_2], axis=1)
                # TODO add back in some sort of JR, SR check
                features_df = compare_strings(sbs_df, variable.name)
            else:
                features_df = None

            if features_df is not None:
                data_rows_df = pd.concat([data_rows_df, features_df], axis=1)

        data_rows_df.columns = ['var_' +  colname if colname not in non_feature_cols else colname
                for colname in data_rows_df.columns]

        try:
            data_rows_df['exactmatch'] = 1
            for exact_match_col in params.exact_match_variables:
                data_rows_df['exactmatch'] = \
                        ((data_rows_df.exactmatch) &
                        (sbs_df[f'{exact_match_col}_1'] == sbs_df[f'{exact_match_col}_2']) &
                        (sbs_df[f'{exact_match_col}_1'] != '')).astype(int)
            data_rows_df['var_exact_match'] = data_rows_df.exactmatch.copy()
            for neg_var in params.negate_exact_match_variables:
                data_rows_df.loc[(sbs_df[f'{neg_var}_1'] != sbs_df[f'{neg_var}_2']) &
                                (sbs_df[f'{neg_var}_1'] != '') &
                                (sbs_df[f'{neg_var}_2'] != ''), 'var_exact_match'] = 0
        except:
            pass

        data_rows_df['label'] = generate_label(
                sbs_df,
                uid_cols,
                params.leven_thresh)

        if data_rows_df is not None and len(data_rows_df) > 0:
            data_rows_df['dr_id'] = data_rows_df.record_id_1 + '__' + data_rows_df.record_id_2
            data_rows_df.set_index('dr_id', inplace=True)

        return data_rows_df

    @log_runtime_and_memory
    @profile
    def generate_data_row_files(
                self, params, schema, an, cp_df, name_probs,
                start_ix_worker, end_ix_worker, dr_file, **kw):
        '''The get_data_row_files function is run in parallel to generate the data needed for
        the random forest; it performs the merge between candidate pairs and all-names and
        calls the function that calculates distance metrics.

        Args:
            params (Parameters object): contains matching parameters
            schema (Schema object): contains matching schema (data files and variables)
            an (pd.DatFrame): all-names table (one row per input record)

                =====================   =======================================================
                record_id               unique record identifier
                file_type               either "new" or "existing"
                <fields for matching>   both for the matching model and for constraint checking
                <raw name fields>       pre-cleaning version of first and last name
                blockstring             concatenated version of blocking columns (sep by ::)
                drop_from_nm            flag, 1 if met any "to drop" criteria 0 otherwise
                =====================   =======================================================

            cp_df (pd.DataFrame): candidate-pairs list

                ======================   =======================================================
                blockstring_1            concatenated version of blocking columns for first element in pair (sep by ::)
                blockstring_2            concatenated version of blocking columns for second element in pair (sep by ::)
                covered_pair             flag; 1 for pairs that made it through blocking, 0 otherwise; all 1s here
                ======================   =======================================================

            name_probs (nm_prob.NameProbability object): contains information about name probabilities
            start_ix_worker (int): starting index of the candidate-pairs chunk to read in this thread
            end_ix_worker (int): end index of the candidate-pairs chunk to read in this thread
            dr_file (str): path to data-rows file to write (one for each worker thread)
        '''
        try:
            cp_df = cp_df.copy()

            # sqlite replacement
            an_bs = an[['blockstring']].copy()
            an_bs['i'] = np.arange(len(an_bs))
            an_ix_map = defaultdict(lambda: [])
            for tup in an_bs.itertuples():
                an_ix_map[tup.blockstring].append(tup.i)

            # for each candidate pair, get the records associated with each name and compute
            # distances between features like date, first_name, dob, etc.

            cp_df['candidate_pair_ix'] = cp_df.index

            self.num_batches = ceil(len(cp_df) / self.params.data_rows_batch_size)
            start_ix_cp = 0
            while start_ix_cp < len(cp_df):
                if (start_ix_worker == 0) and (params.verbose is not None) and (start_ix_cp % params.verbose == 0):
                    logger.info(f"   Generated features for {start_ix_cp * params.num_workers} "
                                f"of ~{len(cp_df) * params.num_workers} pairs of blockstrings.")

                end_ix_cp = min(start_ix_cp + self.params.data_rows_batch_size, len(cp_df))

                relevant_cp = cp_df.iloc[start_ix_cp : end_ix_cp].copy()
                relevant_blockstrings = pd.concat([relevant_cp.blockstring_1,
                                                   relevant_cp.blockstring_2]).unique().tolist()
                relevant_bs_ix = [an_ix_map[bs] for bs in relevant_blockstrings]
                relevant_bs_ix = [item for sublist in relevant_bs_ix for item in sublist]

                # get relevant records for this batch
                relevant_records = an.iloc[relevant_bs_ix].copy()
                relevant_records.set_index('blockstring', inplace=True)

                # expand the candidate pairs from a blockstring level to a record level
                side_by_side_df = pd.merge(
                    relevant_cp[['blockstring_1', 'blockstring_2', 'candidate_pair_ix', 'covered_pair']],
                    relevant_records.copy(),
                    left_on='blockstring_1', right_index=True).reset_index(drop=True)
                side_by_side_df = pd.merge(
                    side_by_side_df,
                    relevant_records.copy(),
                    left_on='blockstring_2', right_index=True, suffixes=['_1', '_2'])

                data_rows_df = self.generate_actual_data_rows(
                        params, schema, side_by_side_df, name_probs, first_iter=(start_ix_cp == 0))

                if data_rows_df is None:
                    start_ix_cp += self.params.data_rows_batch_size
                    continue

                # outcome for selection model
                data_rows_df['labeled_data'] = (data_rows_df.label != '').astype(int)

                table = pa.Table.from_pandas(data_rows_df)
                if start_ix_cp == 0:
                    pqwriter = pq.ParquetWriter(dr_file, table.schema)

                pqwriter.write_table(table)

                start_ix_cp += self.params.data_rows_batch_size

            if pqwriter:
                pqwriter.close()

        except Exception as e:
            os.remove(dr_file)
            logger.error(f"{dr_file} failed.")
            raise e

