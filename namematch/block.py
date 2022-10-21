import csv
import editdistance
import logging
import os
import multiprocessing as mp

# suppress super-verbose nmslib logging
logging.getLogger('nmslib').setLevel(logging.WARNING)
import nmslib

import numpy as np
import itertools
import pandas as pd
import pickle
import string

import pyarrow as pa
import pyarrow.parquet as pq

from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer

from namematch.data_structures.schema import Schema
from namematch.data_structures.parameters import Parameters
from namematch.base import NamematchBase
from namematch.utils.utils import (
    log_runtime_and_memory,
    get_nn_string_from_blockstring,
    build_blockstring,
    get_endpoints,
    get_ed_string_from_blockstring,
)
from namematch.utils.profiler import Profiler

profile = Profiler()

logger = logging.getLogger()

class Block(NamematchBase):
    '''
    Args:
        params (Parameters object): contains matching parameter values
        schema (Schema object): contains match schema info (files to match, variables to use, etc.)
        all_names_file (str): path to the all-names file
        must_links_file (str): path to the must-links file
        blocking_index_bin_file: name of blocking index file
        og_blocking_index_file (str): path to a pre-built nmslib index (optional, if doesn't exist then None)
        candidate_pairs_file (str): path to the candidate-pairs file

    '''
    def __init__(
        self,
        params,
        schema,
        all_names_file='all_names.parquet',
        must_links_file='must_links.csv',
        candidate_pairs_file='candidate_pairs.parquet',
        blocking_index_bin_file='blocking_index.bin',
        og_blocking_index_file='None',
        *args,
        **kwargs
    ):
        super(Block, self).__init__(params, schema, *args, **kwargs)

        self.all_names_file = all_names_file
        self.must_links_file = must_links_file
        self.og_blocking_index_file = og_blocking_index_file
        self.main_index_file = blocking_index_bin_file
        self.candidate_pairs_file = candidate_pairs_file

    @property
    def output_files(self):
        output_files = [
            self.candidate_pairs_file,
            self.main_index_file,
            self.main_index_file + '.pkl',
            self.main_index_file + '.dat'
        ]
        if not self.params.incremental:
            temp_dir = os.path.dirname(self.candidate_pairs_file)
            output_files.append(os.path.join(temp_dir, 'uncovered_pairs.csv'))
        return output_files

    @log_runtime_and_memory
    @profile
    def main(self, **kw):
        '''Generate the candidate-pairs list using the blocking scheme outlined in the config.'''
        temp_dir = os.path.dirname(self.candidate_pairs_file)

        nn_cols, ed_col, absval_col = get_blocking_columns(self.params.blocking_scheme)

        # check if the absval_col is needed -- if not, set to None
        temp = pd.read_parquet(self.all_names_file, columns=['drop_from_nm', ed_col])
        if temp[temp.drop_from_nm == 0][[ed_col]].query(f"{ed_col} == '' or {ed_col}.isnull()").shape[0] == 0:
            absval_col = None

        an = read_an(self.all_names_file, nn_cols, ed_col, absval_col)

        nn_string_info, nn_string_expanded_df = \
                self.convert_all_names_to_blockstring_info(an, absval_col, self.params)

        main_index, \
        main_index_nn_strings, \
        second_index, \
        second_index_nn_strings = self.get_indices(
                self.params,
                nn_string_info.nn_string.tolist(),
                self.og_blocking_index_file)

        # only query names that appear in the new data
        nn_string_info_to_query, \
        nn_strings_to_query, \
        shingles_to_query = self.get_query_strings(nn_string_info, self.params.blocking_scheme)

        if os.path.exists(self.candidate_pairs_file):
            os.remove(self.candidate_pairs_file)

        exact_match_cp_df = self.get_exact_match_candidate_pairs(
                nn_string_info_to_query[nn_string_info_to_query.n_total > 1],
                nn_string_expanded_df,
                self.params.blocking_thresholds)
        write_some_cps(exact_match_cp_df, self.candidate_pairs_file, header=True)

        candidate_pairs_df = self.generate_candidate_pairs(
            nn_strings_to_query,
            shingles_to_query,
            nn_string_info,
            nn_string_expanded_df,
            main_index,
            main_index_nn_strings,
            second_index,
            second_index_nn_strings,
            batch_size=self.params.block_batch_size,
        )

        # evaluate blocking
        if not self.params.incremental:
            must_links_df = pd.read_csv(self.must_links_file)

            true_pairs_df = generate_true_pairs(must_links_df)
            uncovered_pairs_df = self.evaluate_blocking(
                    candidate_pairs_df, true_pairs_df, self.params.blocking_scheme)

            candidate_pairs_df = self.add_uncovered_pairs(candidate_pairs_df, uncovered_pairs_df)

            uncovered_pairs_df.to_csv( # only needed for each sanity check glance
                os.path.join(temp_dir, 'uncovered_pairs.csv'), index=False,
                quoting=csv.QUOTE_NONNUMERIC, sep=',')

        # output
        table = pa.Table.from_pandas(candidate_pairs_df)
        pq.write_table(table, self.candidate_pairs_file)

        save_main_index(main_index, main_index_nn_strings, self.main_index_file)

        if self.enable_lprof:
            self.write_line_profile_stats(profile.line_profiler)

    @log_runtime_and_memory
    @profile
    def split_last_names(self, df, last_name_column, blocking_scheme, **kw):
        '''Expand the processed all-names file to handle double last names (e.g. SAM SMITH-BROWN
        becomes SAM SMITH and SAM BROWN).

        Args:
            df: all-names table, relevant columns only (where drop_from_nm == 0)
            last_name_column (str): clean last name column
            blocking_scheme (dict): dictionary with info on how to do blocking

        Returns:
            pd.DataFrame: more rows than input all names, plus orig_last_name and orig_record columns
        '''

        from collections import Counter

        df = df.copy()

        # preliminary cleaning

        df["orig_last_name"] = df[last_name_column]

        names_to_expand = df[df[last_name_column].str.contains(' ')].copy()

        if len(names_to_expand) > 0:

            # if there are multiple spaces, split on the last one
            names_to_expand["split_names"] = (
                    names_to_expand[last_name_column].str
                    .rsplit(" ", 1))

            # unpack the list of splits, so each is in its own row
            id_vars = names_to_expand.columns.tolist()
            id_vars.remove("split_names")
            split_up = (names_to_expand["split_names"]
                    .apply(pd.Series)
                    .merge(names_to_expand, left_index=True, right_index=True)
                    .drop(columns=["split_names"], axis=1)
                    .melt(id_vars=id_vars, value_name="split_name")
                    .drop(columns=["variable"], axis=1)
                    .dropna()
                    .drop_duplicates())

            split_up[last_name_column] = split_up["split_name"]
            split_up = split_up.drop(columns=["split_name"])

            # make sure the original name is there as well
            df = pd.concat([df, split_up])

        df = df.drop_duplicates()
        df["orig_record"] = (df[last_name_column] == df["orig_last_name"]).astype(int)
        df = df.reset_index(drop=True)

        df['nn_string_full'] = df.nn_string
        df['nn_string'] = build_blockstring(df, blocking_scheme, incl_ed_string=False)

        return df

    @log_runtime_and_memory
    @profile
    def convert_all_names_to_blockstring_info(self, an, absval_col, params, **kw):
        '''Create a table with information about blockstrings. If the split_names parameter is True,
        then this function expands double last names to create two new "records" (e.g. SAM SMITH-BROWN
        becomes SAM SMITH and SAM BROWN).

        Args:
            an (pd.DataFrame): all-names table, relevant columns only (where drop_from_nm == 0)

                =======================  =======================================================
                record_id                unique record identifier
                blockstring              concatenation of all the blocking variables (sep by ::)
                file_type                either "new" or "existing"
                drop_from_nm             flag, 1 if met any "to drop" criteria 0 otherwise
                <nn-blocking column(s)>  variables for near-neighbor blocking
                <ed-blocking column>     variable for edit-distance blocking
                <av-blocking column>     (optional) variable for abs-value blocking
                nn_string                concatenated version of nn-blocking columns (sep by ::)
                ed_string                copy of ed-blocking column
                absval_string            copy of abs-value-blocking column
                =======================  =======================================================

            absval_col (str): column for absolute-value blocking
            params (Parameter object): contains matching parameters

        Returns:
                tuple: tuple containing:

                - **nn_string_info** (*pd.DataFrame*): table with one row per nn_string (or expanded nn_string)

                    ======================   ==============================================================
                    nn_string                concatenated version of nn-blocking columns (sep by ::)
                    commonness_penalty       float indicating how common the last name is
                    n_new                    number of times this nn_string appears in a "new" record
                    n_existing               number of times this nn_string appears in an "existing" record
                    n_total                  number of times this nn_string appears in any record
                    ======================   ==============================================================

                - **nn_string_expanded_df** (*pd.DataFrame*): table with one row per blockstring (or expanded blockstring)

                    ======================   ========================================================================
                    nn_string                concatenated version of nn-blocking columns (sep by ::)
                    nn_string_full           (optional) if split_names is True, this is the full (un-split) nn_string
                    ed_string                copy of ed-blocking column
                    absval_string            copy of abs-value-blocking column
                    ======================   ========================================================================

        '''


        an = an.copy()

        logger.info("Preparing blockstring data.")

        expanded_cols = ['nn_string']
        an_use = an.copy()

        if params.split_names:
            logger.trace('Splitting last names.')
            an_split_ln = self.split_last_names(an, params.last_name_column, params.blocking_scheme)
            an_use = an_split_ln.copy()
            expanded_cols.append('nn_string_full')

        expanded_cols.append('ed_string')
        if absval_col is not None:
            expanded_cols.append('absval_string')

        nn_string_expanded_df = an_use[expanded_cols].drop_duplicates().set_index('nn_string')

        # create dictionary mapping nn_strings to their record counts
        # NOTE: dict has two keys, "new" and "old"
        nn_strings_to_num_recs = get_nn_string_counts(an_use)

        # count last name frequency to establish a measure of how common a name is
        commonness_penalty_lookup = get_common_name_penalties(
                an[params.last_name_column],
                params.blocking_thresholds['common_name_max_penalty'])
        an_use['commonness_penalty'] = an_use[params.last_name_column].map(commonness_penalty_lookup)
        # NOTE: should the params.last_name_column in the above line be original last name always instead?

        nn_string_info = an_use[['nn_string', 'commonness_penalty']].drop_duplicates()

        if len(nn_string_info) != nn_string_info.nn_string.nunique():
            logger.error(f'More than one last name tied to nn_string, meaning there are '
                         f'multiple commonness_penalties per nn_string. Will cause a weird merge.')
            raise

        nn_string_info['n_new'] = nn_string_info.nn_string.map(nn_strings_to_num_recs['new'])
        nn_string_info['n_existing'] = nn_string_info.nn_string.map(nn_strings_to_num_recs['existing'])
        nn_string_info['n_total'] = nn_string_info.n_new + nn_string_info.n_existing

        return nn_string_info, nn_string_expanded_df


    def get_query_strings(self, nn_string_info, blocking_scheme):
        '''Filter to nn_strings that appear in the new data -- these are the only strings
        for which we need near-neighbors. If incremental is False, this filtering step does nothing.

        Args:
            nn_string_info (pd.DataFrame): table with one row per nn_string (or expanded nn_string)

                ======================   ==============================================================
                nn_string                concatenated version of nn-blocking columns (sep by ::)
                commonness_penalty       float indicating how common the last name is
                n_new                    number of times this nn_string appears in a "new" record
                n_existing               number of times this nn_string appears in an "existing" record
                n_total                  number of times this nn_string appears in any record
                ======================   ==============================================================

            blocking_scheme (dict): dictionary with info on how to do blocking

        Returns:
                tuple: tuple containing:

                - **nn_string_info_to_query** (*pd.DataFrame*): nn_string_info, subset to nn_strings where n_new > 0
                - **nn_strings_to_query** (*list*): nn_strings that appear at least once in a "new" record
                - **shingles_to_query** (*scipy.sparse.csr_matrix*): sparse weighted shingles matrix for the nn_strings that appear in a new record
        '''

        nn_string_info = nn_string_info.copy()

        nn_string_info_to_query = nn_string_info[nn_string_info.n_new > 0].copy()
        nn_strings_to_query = nn_string_info_to_query.nn_string.tolist()

        shingles_to_query = self.generate_shingles_matrix(
            nn_strings_to_query,
            blocking_scheme['alpha'],
            blocking_scheme['power'],
            matrix_type='query')

        return nn_string_info_to_query, nn_strings_to_query, shingles_to_query

    @log_runtime_and_memory
    @profile
    def generate_shingles_matrix(
            self,
            nn_strings,
            alpha,
            power,
            matrix_type,
            verbose=True,
            **kw):
        '''Return a weighted sparse matrix of 2-shingles

        Args:
            alpha (float): weight of LAST relative to FIRST
            power (float): parameter controlling the impact of name length on cosine distance
            matrix_type (str): description of matrix being built (for logging)
            verbose (bool): True if status messages desired

        Returns:
            scipy.sparse.csr_matrix: Weighted sparse 2-shingles matrix
        '''

        nn_strings = nn_strings[:]

        all_shingles = get_all_shingles()

        if verbose:
            logger.info(f"Creating {matrix_type} shingles matrix.")

        # shingles input strings
        vectorizer = CountVectorizer(
            analyzer='char',
            ngram_range=(2,2),
            dtype=np.double,
            decode_error='replace',
            vocabulary=all_shingles,
            lowercase=False)

        def make_matrix(i, num):
            formatted = ['*' + nns.split('::')[i] + '*' for nns in nn_strings]
            shingles_matrix = vectorizer.fit_transform(formatted)
            weights = num / np.power(shingles_matrix.sum(axis=1), power)
            shingles_matrix = shingles_matrix.multiply(weights)
            return shingles_matrix

        # sum of shingles for the first name should equal 1
        var1_shingles_matrix = make_matrix(0, 1)
        # sum of shingles for the last name should equal alpha
        var2_shingles_matrix = make_matrix(1, alpha)
        # combine weighted matrices
        shingles_matrix = var1_shingles_matrix + var2_shingles_matrix

        return shingles_matrix

    @log_runtime_and_memory
    @profile
    def load_main_index(self, index_file, **kw):
        '''Load the main index, which is reusable over time as data is added incrementally.

        Args:
            index_file (str): path to stored index

        Returns:
            nmslib.FloatIndex: nmslib index object
        '''

        ix = prep_index()
        ix.loadIndex(index_file, load_data=True)

        return ix

    @log_runtime_and_memory
    def generate_index(
            self,
            nn_strings,
            num_workers,
            M,
            efC,
            post,
            alpha,
            power,
            print_progress=True,
            **kw):
        '''Build an nmslib index based on a list of nn_strings and a set of parameters.

        Args:
            nn_strings (list): strings of the form 'FIRST::LAST' to shingle and put in matrix (rows)
            num_workers (int): number of threads nmslib should use when parallelizing
            M, efc, post: nmslib parameters
            alpha (float): weight of last-name relative to first-name
            power (float): parameter controlling the impact of name length on cosine distance
            print_progress (bool): controls verbosity of index creation

        Returns:
            nmslib.FloatIndex: nmslib index object
        '''

        index_params = {
            'M' : M,
            'indexThreadQty' : num_workers,
            'efConstruction' : efC,
            'post' : post
        }

        shingles_matrix = self.generate_shingles_matrix(nn_strings, alpha, power, 'index')
        ix = prep_index()
        # batch load data points
        temp = ix.addDataPointBatch(
            data=shingles_matrix,
            ids=np.arange(shingles_matrix.shape[0], dtype=np.int32))
        ix.createIndex(index_params, print_progress=print_progress)

        return ix

    @log_runtime_and_memory
    @profile
    def get_indices(self, params, all_nn_strings, og_blocking_index_file, **kw):
        '''Wrapper function coordinating the creation and/or loading of the nmslib indices.

        Args:
            params (Parameters object): contains matching parameter values
            all_nn_strings: list of all unique nn_strings in the data (expanded if split_names is True)
            og_blocking_index_file: path to a pre-build nmslib index (optional, if doesn't exist then None)

        Returns:
               tuple: tuple containing:

                - **main_index** (*nmslib.FloatIndex*): the main nmslib index
                - **main_index_nn_strings** (*list*): nn_strings that are in the main nmslib index
                - **second_index** (*nmslib.FloatIndex*): the secondary nmslib index for querying new nn_strings during incremental runs (often None)
                - **second_index_nn_strings** (*list*): nn_strings that are in the secondary nmslib index (often None)
        '''


        main_index = None
        second_index = None
        main_index_nn_strings = None
        second_index_nn_strings = None

        build_index_from_scratch = ((not params.incremental) or \
                                    og_blocking_index_file == 'None' or \
                                    params.index['rebuild_main_index'] == 'always')

        # if not build_index_from_scratch so far, check one more time consuming trigger
        if (not build_index_from_scratch) and params.index['rebuild_main_index'] == 'if_secondary_index_exceeds_limit':
            main_index_nn_strings = load_main_index_nn_strings(og_blocking_index_file)
            second_index_nn_strings = get_second_index_nn_strings(all_nn_strings, main_index_nn_strings)
            if len(second_index_nn_strings) >= params.index['secondary_index_limit']:
                build_index_from_scratch = True

        if build_index_from_scratch:

            logger.trace('Building index from scratch.')
            main_index = self.generate_index(
                    all_nn_strings,
                    params.num_workers, params.nmslib['M'], params.nmslib['efC'],
                    params.nmslib['post'], params.blocking_scheme['alpha'],
                    params.blocking_scheme['power'])
            main_index_nn_strings = all_nn_strings[:]

        else:

            if second_index_nn_strings is None:

                main_index_nn_strings = load_main_index_nn_strings(og_blocking_index_file)
                second_index_nn_strings = get_second_index_nn_strings(all_nn_strings, main_index_nn_strings)

            # load main index
            logger.trace('Loading main index.')
            main_index = self.load_main_index(og_blocking_index_file)

            # build second index
            if len(second_index_nn_strings) > 0:

                logger.trace('Building second index.')
                second_index = self.generate_index(
                        second_index_nn_strings,
                        params.num_workers, params.nmslib['M'], params.nmslib['efC'],
                        params.nmslib['post'], params.blocking_scheme['alpha'],
                        params.blocking_scheme['power'])

        return main_index, main_index_nn_strings, second_index, second_index_nn_strings

    @log_runtime_and_memory
    @profile
    def generate_candidate_pairs(
        self,
        nn_strings_to_query,
        shingles_to_query,
        nn_string_info,
        nn_string_expanded_df,
        main_index,
        main_index_nn_strings,
        second_index,
        second_index_nn_strings,
        batch_size,
        **kw
    ):
        '''Wrapper function for querying the nmslib index (or indices) and getting
        non-matching candidate pairs.

        Args:
            nn_strings_to_query (list): nn_strings in new data -- those that need near neighbors
            shingles_to_query (csr_matrix): shingles matrix for nn_strings_to_query
            nn_string_info (pd.DataFrame):  table with one row per nn_string (or expanded nn_string)
            nn_string_expanded_df (pd.DataFrame): maps a nn_string to a ed_string and absval_string
            main_index (nmslib index): the main nmslib index for querying
            main_index_nn_strings (list): nn_strings in main_index
            second_index (nmslib index): the secondary nmslib index, for some incremental runs
            second_index_nn_strings (list): nn_strings in second_index
            batch_size (int): batch size. Default is 10000 and can be modify in config.yaml file.
        Returns:
            pd.DataFrame: candidate-pairs list, before adding in uncovered pairs

            ======================   =======================================================
            blockstring_1            concatenated version of blocking columns for first element in pair (sep by ::)
            blockstring_2            concatenated version of blocking columns for second element in pair (sep by ::)
            cos_dist                 approximate cosine distance between two nn_strings (nmslib)
            edit_dist                number of character edits between ed-strings
            covered_pair             flag; 1 for pairs that made it through blocking, 0 otherwise; all 1s here
            ======================   =======================================================
        '''
        logger.info('Querying to get candidate pairs.')

        nn_string_info = nn_string_info.copy()

        # determine which indices to query
        indices_to_query = ['main']
        if second_index is not None:
            indices_to_query.append('secondary')

        logger.info(f'Indices to query: {len(indices_to_query)}')

        # how many names do we want to query at a time?
        # runtime/ram tradeoff; can lower if having RAM issues
        # batch_size = 2000 # NOTE can be parameterized/reduced if RAM is issue
        logger.debug(f"batch size: {batch_size}")

        start_ix_batch = 0
        while start_ix_batch < len(nn_strings_to_query):
            end_ix_batch = min(start_ix_batch + batch_size, len(nn_strings_to_query))

            cand_pair_df_list = []

            if self.params.verbose is not None and (start_ix_batch % self.params.verbose == 0):
                logger.info("  %s out of %s nn_strings queried." % (
                        min(start_ix_batch, len(nn_strings_to_query)), len(nn_strings_to_query)))

            for index_type in indices_to_query:

                if index_type == 'main':
                    index_to_query = main_index
                    nn_strings_this_index = main_index_nn_strings[:]
                elif index_type == 'secondary':
                    index_to_query = second_index
                    nn_strings_this_index = second_index_nn_strings[:]
                else:
                    logger.error('Variable "index_type" must be "main" or "secondary."')
                    raise ValueError

                # query the nmslib index:
                # pass in a group of names, and get back the k most similar names)
                near_neighbors_list = index_to_query.knnQueryBatch(
                    queries=shingles_to_query[start_ix_batch:end_ix_batch],
                    k=self.params.nmslib['k'],
                    num_threads=self.params.num_workers)

                nn_strings_queried_this_batch = nn_strings_to_query[start_ix_batch:end_ix_batch]

                near_neighbors_df = self.get_near_neighbors_df(
                        near_neighbors_list,
                        nn_string_info,
                        nn_strings_this_index,
                        nn_strings_queried_this_batch)

                # parallelize the paring down for possible candidate pairs to actual candidate pairs
                end_points = get_endpoints(len(near_neighbors_df), self.params.num_workers)
                if self.params.parallelize:

                    output = mp.Queue()
                    jobs = [
                        mp.Process(
                            target = self.get_actual_candidates,
                            args = (
                                near_neighbors_df.iloc[start_ix_worker : end_ix_worker],
                                nn_string_expanded_df,
                                nn_strings_to_query,
                                self.params.blocking_thresholds,
                                self.params.incremental,
                                output)) for start_ix_worker, end_ix_worker in end_points]
                    for job in jobs:
                        job.start()

                    cand_pairs_to_add_df = [output.get() for job in jobs]

                    [job.join() for job in jobs]
                    failure_occurred = sum([job.exitcode != 0 for job in jobs])
                    if failure_occurred:
                        logger.error("Error occurred in %s worker(s)." % failure_occurred)
                        raise Exception("Error occurred in %s worker(s)." % failure_occurred)

                else:

                    cand_pairs_to_add_df = []
                    for start_ix_worker, end_ix_worker in end_points:
                        cand_pairs_to_add_df_this_iter = self.get_actual_candidates(
                                near_neighbors_df.iloc[start_ix_worker : end_ix_worker],
                                nn_string_expanded_df,
                                nn_strings_to_query,
                                self.params.blocking_thresholds,
                                self.params.incremental)
                        cand_pairs_to_add_df.append(cand_pairs_to_add_df_this_iter)

                cand_pair_df_list.extend(cand_pairs_to_add_df)

            start_ix_batch += batch_size

            cand_pair_df = pd.concat(cand_pair_df_list)
            cand_pair_df = cand_pair_df.drop_duplicates(subset=['blockstring_1', 'blockstring_2'])
            # because same name might be in both indices

            # update total number candidate pairs generated so far; write to file
            write_some_cps(cand_pair_df, self.candidate_pairs_file)
            del cand_pair_df

        # have to load so we can drop duplicates at global level...
        # because of split names
        logger.trace('Loading candidate pairs to drop duplicates.')
        cand_pair_df = pd.read_csv(self.candidate_pairs_file)
        cand_pair_df = cand_pair_df.drop_duplicates(subset=['blockstring_1', 'blockstring_2'])
        cand_pair_df.to_csv(
                self.candidate_pairs_file,
                index=False,
                quoting=csv.QUOTE_NONNUMERIC)

        return cand_pair_df

    @log_runtime_and_memory
    def compute_cosine_sim(self, blockstrings_in_pairs, pairs_df, shingles_matrix, **kw):
        '''Fast cosine similarity computation using the shingles matrix.

        Args:
            blockstrings_in_pairs (list): used to get index of different strings in shingles_matrix
            pairs_df (pd.DataFrame): blockstrings you want cosine distance between

               ===================   =======================================================
               blockstring_1         blockstring for the first record in the pair
               blockstring_2         blockstring for the second record in the pair
               covered_pair          flag, 1 if covered 0 otherwise
               nn_strings_1          nn_string for the first record in the pair
               nn_strings_2          nn_string for the second record in the pair
               both_nn_strings       nn_string_1 + ' ' + nn_string_2
               ===================   =======================================================

        Returns:
            shingles_matrix (csr_matrix): weighted shingles matrix

        '''

        name_index_map = {}
        for i, the_name in enumerate(blockstrings_in_pairs):
            name_index_map[the_name] = i

        idx1 = []
        idx2 = []
        for i, row in pairs_df.iterrows():
            n1 = row['blockstring_1']
            n2 = row['blockstring_2']
            if n1 in name_index_map and n2 in name_index_map:
                idx1.append(name_index_map[n1])
                idx2.append(name_index_map[n2])

        mat1 = shingles_matrix[np.array(idx1)]
        mat2 = shingles_matrix[np.array(idx2)]

        numerator = mat1.multiply(mat2).sum(1)

        mat1_squared = mat1.multiply(mat1).sum(1)
        mat2_squared = mat2.multiply(mat2).sum(1)
        denom = np.multiply(np.sqrt(mat1_squared), np.sqrt(mat2_squared))

        pair_cos = 1.0 - np.divide(numerator, denom)
        logger.info('Computing cosine similarities.')

        pair_cos = pair_cos.flatten().tolist()[0]

        return pair_cos

    @log_runtime_and_memory
    @profile
    def evaluate_blocking(self, cp_df, tp_df, blocking_scheme, **kw):
        '''The evaluate_blocking function computes the pair completeness metrics to
        determine how successful blocking was at minimizing comparisons and
        maximizing true positives (i.e. generating a candidate pair between
        records that are actually matches).

        Args:
            cp_df (pd.DataFrame): candidate pairs df
            tp_df (pd.DataFrame): true pairs df (blockstring_1, blockstring_2)
            blocking_scheme (dict): blocking_scheme (dict): dictionary with info on how to do blocking

        Returns:
            pd.DataFrame: portion of candidate-pairs dataframe where covered == 0
        '''

        cp_df = cp_df.copy()

        cp_df['nn_string_1'] = np.vectorize(
                get_nn_string_from_blockstring, otypes=['object'])(cp_df.blockstring_1)
        cp_df['nn_string_2'] = np.vectorize(
                get_nn_string_from_blockstring, otypes=['object'])(cp_df.blockstring_2)
        cp_df['both_blockstrings'] = \
                cp_df.blockstring_1 + ' ' + cp_df.blockstring_2
        cp_df['both_nn_strings'] = \
                cp_df.nn_string_1 + ' ' + cp_df.nn_string_2

        # covered/uncovered pairs (get cosine distribution)
        # -----------------------------------------------

        pasted = tp_df["blockstring_1"] + " " + tp_df["blockstring_2"]
        tp_df["covered_pair"] = pasted.isin(cp_df["both_blockstrings"]).astype(int)

        tp_df['nn_string_1'] = np.vectorize(
                get_nn_string_from_blockstring, otypes=['object'])(tp_df.blockstring_1)
        tp_df['nn_string_2'] = np.vectorize(
                get_nn_string_from_blockstring, otypes=['object'])(tp_df.blockstring_2)
        tp_df['both_nn_strings'] = \
                tp_df.nn_string_1 + ' ' + tp_df.nn_string_2

        tp_df_nonmatching = \
                tp_df[tp_df.blockstring_1 != tp_df.blockstring_2].copy()

        # calculate cosine distance between true pairs (non-matching)
        blockstrings_in_true_pairs = \
                tp_df.blockstring_1.append(
                tp_df.blockstring_2, ignore_index=True).drop_duplicates().tolist()
        tp_shingles_matrix = self.generate_shingles_matrix(
                blockstrings_in_true_pairs,
                blocking_scheme['alpha'], blocking_scheme['power'],
                matrix_type='true pairs', verbose=False)
        tp_df_nonmatching['cos_dist'] = self.compute_cosine_sim(blockstrings_in_true_pairs,
                tp_df_nonmatching, tp_shingles_matrix)

        # get the distribution of cosine distances in non-matching true pairs
        tp_nonmatching_cos_distr = pd.value_counts(
                pd.cut(tp_df_nonmatching.cos_dist, np.arange(0, 1.1, .1)),
                normalize=True, sort=False)
        logger.trace(f'Cosine distribution of non-matching true pairs: \n{tp_nonmatching_cos_distr.to_string()}')
        self.stats_dict['tp_cosine_distribution'] = tp_nonmatching_cos_distr.tolist()

        # output uncovered pairs so we can see where blocking fails
        up_df = tp_df_nonmatching[tp_df_nonmatching.covered_pair == 0].copy()
        up_df['ed_string_1'] = np.vectorize(
                get_ed_string_from_blockstring, otypes=['object'])(up_df.blockstring_1)
        up_df['ed_string_2'] = np.vectorize(
                get_ed_string_from_blockstring, otypes=['object'])(up_df.blockstring_2)
        up_df['edit_dist'] = np.vectorize(editdistance.eval, otypes=['float'])(
                up_df.ed_string_1.values, up_df.ed_string_2.values)
        up_df = up_df[['blockstring_1', 'blockstring_2', 'cos_dist',
                                                 'edit_dist', 'covered_pair']].copy()

        logger.info(f"Number of uncovered pairs: {len(up_df)}")
        self.stats_dict['n_uncovered_pairs'] = len(up_df)
        logger.info(f"Number of true pairs: {len(tp_df)}")
        self.stats_dict['n_true_blockstring_pairs'] = len(tp_df)

        # get the distribution of cosine distances in uncovered pairs
        # NOTE: fine that coming from non-matching df because no uncovered pairs will ever match
        uncovered_pair_cos_distr = pd.value_counts(
                pd.cut(up_df.cos_dist, np.arange(0, 1.1, .1)),
                normalize=True, sort=False)
        logger.trace(f'Cosine distribution of uncovered pairs: \n{uncovered_pair_cos_distr.to_string()}')
        self.stats_dict['up_cosine_distribution'] = uncovered_pair_cos_distr.tolist()

        # pair completeness, aka pc (high is good, max 1)
        # -----------------------------------------------
        # this is the proportion of actual matches that make it through blocking

        logger.info("Calculating pair completeness.")

        # calculate pair completeness for all candidate pairs
        pc_numerator_nn_string_level = tp_df[tp_df.both_nn_strings.isin(
                cp_df.both_nn_strings) == False].both_nn_strings.nunique()
        pc_numerator_blockstring_level = len(up_df)
        pc_denominator_nn_string_level = tp_df.both_nn_strings.nunique()
        pc_denominator_blockstring_level = len(tp_df)
        try:
            pc_nn_string_level = \
                    1 - (pc_numerator_nn_string_level / pc_denominator_nn_string_level)
            pc_blockstring_level = \
                    1 - (pc_numerator_blockstring_level / pc_denominator_blockstring_level)
            logger.info(f"Pair completeness, including equal blockstrings (cosine level): {round(pc_nn_string_level, 3)}")
            self.stats_dict['pc_eq_cos'] = round(pc_nn_string_level, 3)
            logger.info(f"Pair completeness, including equal blockstrings (cosine + editdistance level): {round(pc_blockstring_level, 3)}")
            self.stats_dict['pc_eq_cosed'] = round(pc_blockstring_level, 3)
        except:
            logger.warning(f"Pair completeness cannot be calculated. Make sure necessary "
                           f"files are populated.")

        # calculate pair completeness for candidate pairs that are not the
        # same (e.g. NameX, NameX) since those should be perfect
        pc_denominator_nonequal_nn_string_level = \
                tp_df[tp_df.nn_string_1 != tp_df.nn_string_2].both_nn_strings.nunique()
        pc_denominator_nonequal_blockstring_level = len(tp_df_nonmatching)
        try:
            pc_nn_string_level = \
                    1 - (pc_numerator_nn_string_level / pc_denominator_nonequal_nn_string_level)
            pc_blockstring_level = \
                    1 - (pc_numerator_blockstring_level / pc_denominator_nonequal_blockstring_level)
            logger.info(f"Pair completeness, non-equal blockstrings (cosine level): {round(pc_nn_string_level, 3)}")
            self.stats_dict['pc_neq_cos'] = round(pc_nn_string_level, 3)
            logger.info(f"Pair completeness, non-equal blockstrings (cosine + editdistance level): {round(pc_blockstring_level, 3)}")
            self.stats_dict['pc_neq_cosed'] = round(pc_blockstring_level, 3)
        except:
            logger.warning(f"Pair completeness (non-equal blockstrings) cannot be calculated. "
                           f"Make sure necessary files are populated.")

        logger.info("Finished evaluating blocking.")

        return up_df

    def add_uncovered_pairs(self, candidate_pairs_df, uncovered_pairs_df):
        '''Add the uncovered pairs to the candidate pairs dataframe so that all of the
        known pairs are in the candidate pairs list.

        Args:
            candidate_pairs_df (pd.DataFrame): candidate pairs file produced by blocking
            uncovered_pairs_df (pd.DataFrame): uncovered pairs produced by evaluating blocking

        Return:
            pd.DataFrame: candidate-pairs file

            ======================   =======================================================
            blockstring_1            concatenated version of blocking columns for first element in pair (sep by ::)
            blockstring_2            concatenated version of blocking columns for second element in pair (sep by ::)
            cos_dist                 approximate cosine distance between two nn_strings (nmslib)
            edit_dist                number of character edits between ed-strings
            covered_pair             flag; 1 for pairs that made it through blocking, 0 otherwise
            ======================   =======================================================
        '''

        logger.info(f'Blocked covered pairs: {len(candidate_pairs_df)}')
        self.stats_dict['n_bcp'] = len(candidate_pairs_df)
        logger.info(f'Uncovered pairs: {len(uncovered_pairs_df)}')
        self.stats_dict['n_uncovered_pairs'] = len(uncovered_pairs_df)

        up_cols = [col for col in uncovered_pairs_df.columns.tolist() if col in candidate_pairs_df]
        candidate_pairs_df = pd.concat([
                candidate_pairs_df,
                uncovered_pairs_df[up_cols]], ignore_index=True, sort=False)
        candidate_pairs_df = candidate_pairs_df.sample(frac=1)

        self.stats_dict['n_cand_pairs'] = len(candidate_pairs_df)
        return candidate_pairs_df

    @profile
    def apply_blocking_filter(self, df, thresholds, nn_string_expanded_df, nns_match=False):
        '''Compare similarity of names and DOBs to see if a pair of records are likely to be a match.

        Args:
            df (pd.DataFrame): holds similarity and commonness info about pairs of names

                ======================   ==================================================================================
                nn_string_1              concatenated version of nn-blocking columns for first element in pair (sep by ::)
                nn_string_2              concatenated version of nn-blocking columns for second element in pair (sep by ::)
                cos_dist                 approximate cosine distance between two nn_strings (nmslib)
                commonness_penalty_1     penalty for last-name commonness for first element in pair
                commonness_penalty_2     penalty for last-name commonness for second element in pair
                ======================   ==================================================================================

            thresholds (dict): information about what blocking distances are allowed
            nn_string_expanded_df (pd.DataFrame): maps a nn_string to a ed_string and absval_string
            nns_match (bool): True if this function is called by get_exact_match_candidate_pairs

        Returns:
            pd.DataFrame: chunk of the candidate-pairs list

            ==============   ===============================================================================
            blockstring_1    concatenated version of blocking columns for first element in pair (sep by ::)
            blockstring_2    concatenated version of blocking columns for second element in pair (sep by ::)
            cos_dist         approximate cosine distance between two nn_strings (nmslib)
            edit_dist        number of character edits between ed-strings
            covered_pair     flag; 1 for pairs that made it through blocking, 0 otherwise; all 1s here
            ==============   ===============================================================================
        '''
        # add commonness penalty
        df['commonness_penalty'] = \
                df[['commonness_penalty_1', 'commonness_penalty_2']].mean(axis=1)
        df = df.drop(columns=['commonness_penalty_1', 'commonness_penalty_2'])

        # expand to full-name/dob/age(?) level
        df = pd.merge(df, nn_string_expanded_df, left_on='nn_string_1', right_index=True)
        df = pd.merge(df, nn_string_expanded_df, left_on='nn_string_2', right_index=True, suffixes=['_1', '_2'])
        df = df.reset_index(drop=True)
        if 'nn_string_full' in nn_string_expanded_df:
            df['nn_string_1'] = df.nn_string_full_1 # swap nn_string for nn_string_full
            df['nn_string_2'] = df.nn_string_full_2
            df = df.drop(columns=['nn_string_full_1', 'nn_string_full_2'])

        if nns_match:
            df = df[df.ed_string_1 <= df.ed_string_2]

        # use dob columns to calculate edit distance
        # initialize as either 0 (if equal and not null) or -1
        df['edit_dist'] = ((df.ed_string_1 != '') & (df.ed_string_1 == df.ed_string_2)) - 1
        where_calc_ed = ((df.ed_string_1 != '') & (df.ed_string_2 != '') & (df.ed_string_1 != df.ed_string_2))
        df.loc[where_calc_ed, 'edit_dist'] = \
                np.vectorize(editdistance.eval, otypes=['float'])(
                    df[where_calc_ed].ed_string_1.values, df[where_calc_ed].ed_string_2.values)

        df['absval_diff'] = 0
        if 'absval_string_1' in df.columns.tolist():
            # need to collapse to name/age level
            gb_cols = [col for col in df.columns.tolist() if 'absval' not in col]
            df['absval_diff'] = np.abs(df.absval_string_1 - df.absval_string_2)
            df = df.groupby(gb_cols).absval_diff.min().reset_index()

        # limit to the pairs that are most likely to be the same person
        pass_high_bar = (df.cos_dist <= (thresholds['high_cosine_bar'] - df.commonness_penalty)) & \
                        (df.edit_dist >= 0) & \
                        (df.edit_dist <= thresholds['low_editdist_bar'])

        pass_low_bar =  (df.cos_dist <= (thresholds['low_cosine_bar'] - df.commonness_penalty)) & \
                        (df.edit_dist >= 0) & \
                        (df.edit_dist <= thresholds['high_editdist_bar'])

        pass_nodob_bar = (df.cos_dist <= (thresholds['nodob_cosine_bar'] - df.commonness_penalty)) & \
                         (df.edit_dist == -1) & \
                         ((df.absval_diff <= thresholds['absvalue_bar']) | (df.absval_diff.isnull()))

        df = df[(pass_high_bar | pass_low_bar | pass_nodob_bar)].copy()

        # flip any strings that are not in alphabetical order
        # NOTE: necessary for incremental runs
        df['nns1_is_min'] = (df.nn_string_1.astype(str) + '::') <= (df.nn_string_2.astype(str) + '::')
        df['blockstring_1'] = df.nn_string_1.astype(str) + '::' + df.ed_string_1.astype(str)
        df['blockstring_2'] = df.nn_string_2.astype(str) + '::' + df.ed_string_2.astype(str)
        df.loc[df.nns1_is_min == False, 'blockstring_1'] = df.nn_string_2.astype(str) + '::' + df.ed_string_2.astype(str)
        df.loc[df.nns1_is_min == False, 'blockstring_2'] = df.nn_string_1.astype(str) + '::' + df.ed_string_1.astype(str)

        cand_pairs = df[['blockstring_1', 'blockstring_2', 'cos_dist', 'edit_dist']].copy()

        # once we've applied all the filters, the rows that are left will make it through blocking
        cand_pairs['covered_pair'] = 1

        return cand_pairs


    @profile
    def disallow_switched_pairs(self, df, incremental, nn_strings_to_query):
        '''Look through the columns nn_string_1 and nn_string_2 and keep only rows where
        nn_string1 <= nn_string2 to prevent duplicates in the end (i.e. ABBY->ZABBY &
        ZABBY->ABBY; only one is needed). Special case for incremental runs.

        Args:
            df (pd.DataFrame): holds similarity and commonness info about pairs of names
            incremental (bool) : Ture if current run incremental
            nn_strings_to_query (list): nn_strings that are in "to query" list

        Returns:
            pd.DataFrame: same as input df, but no AB/BA duplicates
        '''

        df = df.copy()

        # hack to help make sure things are ordered properly
        # e.g. JOHN SMITH JR > JOHN SMITH, but # e.g. JOHN SMITH JR:: < JOHN SMITH::
        # e.g. JOHN SMITH-JONES > JOHN SMITH, but # e.g. JOHN SMITH-JONES:: < JOHN SMITH::
        df['nn_string_1_'] = df.nn_string_1 + '::'
        df['nn_string_2_'] = df.nn_string_2 + '::'

        df['allowed'] = 0
        df.loc[df.nn_string_1_ < df.nn_string_2_, 'allowed'] = 1
        df.loc[(incremental) &
              (df.nn_string_1_ > df.nn_string_2_) &
              (df.nn_string_2.isin(nn_strings_to_query) == False), 'allowed'] = 1
        df = df[df.allowed == 1]
        df = df.drop(columns=['nn_string_1_', 'nn_string_2_'])

        return df

    @profile
    def get_actual_candidates(
            self,
            near_neighbors_df,
            nn_string_expanded_df,
            nn_strings_to_query,
            thresholds,
            incremental,
            output=None):
        '''Actually determines whether two names become candidates; this function is launched by
        generate_candidate_pairs() and run on individual worker threads to speed up processing.

        Args:

            near_neighbors_df (pd.DataFrame): holds similarity and commonness info about pairs of names

                ======================   =======================================================
                nn_string_ix             a string with nn_string_ix = i is the string located at nn_strings_queried_this_batch[i]
                nn_string_1              concatenated version of nn-blocking columns for first element in pair (sep by ::)
                nn_string_2              concatenated version of nn-blocking columns for second element in pair (sep by ::)
                cos_dist                 approximate cosine distance between two nn_strings (nmslib)
                commonness_penalty_1     penalty for last-name commonness for first element in pair
                commonness_penalty_2     penalty for last-name commonness for second element in pair
                ======================   =======================================================

            nn_string_expanded_df (pandas dataframe): table at nn_string/ed_string/absval_string level (expanded if split_name is True)
            nn_strings_to_query (list): nn_strings in the "to query" list (needed for incremental check)
            thresholds (dict): information about what blocking distances are allowed
            incremental (bool) : True if current run incremental
            output: None if the output should be returned, rather than written

        Returns:
            pd.DataFrame: chunk of the candidate-pairs list

            ======================   =======================================================
            blockstring_1            concatenated version of blocking columns for first element in pair (sep by ::)
            blockstring_2            concatenated version of blocking columns for second element in pair (sep by ::)
            cos_dist                 approximate cosine distance between two nn_strings (nmslib)
            edit_dist                number of character edits between ed-strings
            covered_pair             flag; 1 for pairs that made it through blocking, 0 otherwise; all 1s here
            ======================   =======================================================
        '''

        near_neighbors_df = near_neighbors_df.copy()

        # narrow down to reasaonable cosine distance
        near_neighbors_df = near_neighbors_df[near_neighbors_df.cos_dist <= thresholds['low_cosine_bar']]

        # narrow down so we don't have duplicates (AB & BA)
        near_neighbors_df = self.disallow_switched_pairs(near_neighbors_df, incremental, nn_strings_to_query)

        # filter out low edit or cosine distances
        cand_pairs_df = self.apply_blocking_filter(near_neighbors_df, thresholds, nn_string_expanded_df)

        if output is None:
            return cand_pairs_df
        else:
            output.put(cand_pairs_df)


    @profile
    def get_near_neighbors_df(
            self,
            near_neighbors_list,
            nn_string_info,
            nn_strings_this_index,
            nn_strings_queried_this_batch):
        '''For a small batch of names (nn_strings_queried_this_batch), format a dataframe that
        enumerates every pair of (name in this batch, a near neighbor), along with information about
        similarity and commonness.

        Args:
            near_neighbors_list (list): list of (list of k IDs, list of k distances) tuples, of length batch_size
            nn_string_info (pd.DataFrame): table mapping nn_string to commonness_penalty
            nn_strings_this_index (list): nn_strings in the current index
            nn_strings_queried_this_batch (list): nn_strings in the current query batch (length batch_size),
                                                  whose neighbors are stored in near_neighbors_list

        Returns:
            pd.DataFrame: holds similarity and commonness info about pairs of names

            ====================   ========================================================================================
            nn_string_ix           a string with nn_string_ix = i is the string located at nn_strings_queried_this_batch[i]
            nn_string_1            concatenated version of nn-blocking columns for first element in pair (sep by ::)
            nn_string_2            concatenated version of nn-blocking columns for second element in pair (sep by ::)
            cos_dist               approximate cosine distance between two nn_strings (nmslib)
            commonness_penalty_1   penalty for last-name commonness for first element in pair
            commonness_penalty_2   penalty for last-name commonness for second element in pair
            ====================   ========================================================================================
        '''

        nn_strings_this_index = nn_strings_this_index[:]
        nn_strings_queried_this_batch = nn_strings_queried_this_batch[:]
        nn_string_info = nn_string_info.copy()

        nn_string_info.set_index('nn_string', inplace=True)

        nn_string_info_queried_this_batch = nn_string_info.loc[nn_strings_queried_this_batch].reset_index().copy()
        nn_string_info_this_index = nn_string_info.loc[nn_strings_this_index].reset_index().copy()

        # each item in these two columns is a list of length k
        df = pd.DataFrame(near_neighbors_list, columns=['near_neighbor_nn_string_ix', 'cos_dist'])
        df['nn_string_ix'] = df.index

        # extract the k neighbor IDs, and k distances from the lists, so each is in its own cell
        # a string with nn_string_ix = i is the string located at nn_strings_queried_this_batch[i]
        # a string with nn_string_ix = i has its neighbors located at near_neighbors_list[i]
        # a string with near_neighbor_nn_string_ix = i is the string located at nn_strings_this_index[i]
        list_cols = ['near_neighbor_nn_string_ix', 'cos_dist']
        df = pd.DataFrame({
          col:np.repeat(df[col].values, df[list_cols[0]].str.len())
          for col in df.columns.drop(list_cols)}
        ).assign(**{list_cols[0]:np.concatenate(df[list_cols[0]].values),
                    list_cols[1]:np.concatenate(df[list_cols[1]].values)})[df.columns]

        # above, nn_string_ix is indices
        # match these indices to the string they correspond to in nn_strings_queried_this_batch
        # (and its commonness_penalty)
        nn_string_info_mini = nn_string_info_queried_this_batch[['nn_string', 'commonness_penalty']].copy()
        nn_string_info_mini.columns = ['nn_string_1', 'commonness_penalty_1']
        df = pd.merge(df, nn_string_info_mini, left_on='nn_string_ix', right_index=True)

        # above, near_neighbor_nn_string_ix is indices
        # match these indices to the string they correspond to in nn_strings_this_index
        # (and its commonness_penalty)
        nn_string_info_mini = nn_string_info_this_index[['nn_string', 'commonness_penalty']].copy()
        nn_string_info_mini.columns = ['nn_string_2', 'commonness_penalty_2']
        df = pd.merge(df, nn_string_info_mini, left_on='near_neighbor_nn_string_ix', right_index=True)

        df = df[['nn_string_ix', 'nn_string_1', 'nn_string_2', 'cos_dist', 'commonness_penalty_1', 'commonness_penalty_2']]

        return df

    def get_exact_match_candidate_pairs(self, nn_string_info_multi, nn_string_expanded_df, blocking_thresholds):
        '''All nn_strings that appear more than once need to have a corresponding
        nn_string, nn_string candidate pair -- we can skip the "approximation" easily
        for this type of candidate pair.

        Args:
            nn_string_info_multi (pd.DataFrame): nn_string_info, subset to nn_strings with n_new > 0 & n_total > 1
            nn_string_expanded_df (pd.DataFrame): table at nn_string/ed_string/absval_string level (expanded if split_name is True)
            blocking_thresholds (dict):  dictionary with thresholds for blocking, e.g. high and low bar

        Returns:
            pd.DataFrame: portion of the candidate pairs list (where nn_string_1 == nn_string_2)

            ==================   ==============================================================
            nn_string            concatenated version of nn-blocking columns (sep by ::)
            commonness_penalty   float indicating how common the last name is
            n_new                number of times this nn_string appears in a "new" record
            n_existing           number of times this nn_string appears in an "existing" record
            n_total              number of times this nn_string appears in any record
            ==================   ==============================================================
        '''

        logger.info('Getting identical candidate pairs.')

        # adding (nnsX, nnsX) as a candidate if there are 2+ record for bsX
        # and 1+ come from "new"; second part of condition is already met
        # given that this df has just "to query" names)

        nn_string_info_multi = nn_string_info_multi.copy()

        # identical first and last names
        exact_match_df = pd.DataFrame(data={
            'nn_string_1' : nn_string_info_multi.nn_string,
            'nn_string_2' : nn_string_info_multi.nn_string,
            'cos_dist' : 0,
            'commonness_penalty_1' : nn_string_info_multi.commonness_penalty,
            'commonness_penalty_2' : nn_string_info_multi.commonness_penalty
        })

        exact_match_cp_df = self.apply_blocking_filter(exact_match_df, blocking_thresholds,
                nn_string_expanded_df, nns_match=True)

        return exact_match_cp_df


def get_blocking_columns(blocking_scheme):
    '''Get the list of blocking variables for each type of blocking:

    Args:
        blocking_scheme (dict): dictionary with info on how to do blocking

    Returns:
        list of string list: the variable names needed for each type of blocking
    '''

    nn_cols = blocking_scheme['cosine_distance']['variables']
    ed_col = blocking_scheme['edit_distance']['variable']

    try:
        absval_col = blocking_scheme['absvalue_distance']['variable']
        if absval_col is None:
            raise
        cols_to_read = nn_cols + [ed_col, absval_col, 'file_type', 'drop_from_nm', 'record_id']
        logger.trace('Absvalue filter used as backup to edit_distance filter.')
    except:
        absval_col = None
        cols_to_read = nn_cols + [ed_col, 'file_type', 'drop_from_nm', 'record_id']
        logger.trace('No backup to edit_distance filter being used.')

    return nn_cols, ed_col, absval_col


def read_an(an_file, nn_cols, ed_col, absval_col):
    '''Read in relevant columns for blocking from the all-names file.

    Args:
        an_file (str): path to the all-names file
        nn_cols (list of strings): variables for near neighbor blocking
        ed_col (list of strings): variables for edit-distance blocking
        absval_col (list of strings): variables for absolute-value blocking

    Returns:
        pd.DataFrame: all-names dataframe, relevant columns only (where drop_from_nm == 0)

        =======================   =======================================================
        record_id                 unique record identifier
        blockstring               concatenation of all the blocking variables (sep by ::)
        file_type                 either "new" or "existing"
        drop_from_nm              flag, 1 if met any "to drop" criteria 0 otherwise
        <nn-blocking column(s)>   variables for near-neighbor blocking
        <ed-blocking column>      variable for edit-distance blocking
        <av-blocking column>      (optional) variable for abs-value blocking
        nn_string                 concatenated version of nm-blocking columns (sep by ::)
        ed_string                 copy of ed-blocking column
        absval_string             copy of ed-blocking column
        =======================   =======================================================
    '''

    cols_to_read = nn_cols + [ed_col, \
            'blockstring', 'file_type', 'drop_from_nm', 'record_id']

    if absval_col is not None:
        cols_to_read = nn_cols + [ed_col, absval_col, \
                'blockstring', 'file_type', 'drop_from_nm', 'record_id']

    table = pq.read_table(an_file)
    an =  table.to_pandas()
    if absval_col is not None and absval_col not in an.columns.tolist():
        logger.error(f'The {absval_col} variable was not found, and is needed for the chosen blocking scheme.')
        raise ValueError
    an = an[cols_to_read]

    an = an.fillna('')
    an = an[an.drop_from_nm == 0]

    an['nn_string'] = np.vectorize(get_nn_string_from_blockstring,
                                   otypes=['object'])(an.blockstring)
    an['ed_string'] = an[ed_col]
    if absval_col is not None:
        an['absval_string'] = an[absval_col]
        an.loc[an.absval_string == '', 'absval_string'] = np.NaN
        an['absval_string'] = an.absval_string.astype(float)

    return an


def get_nn_string_counts(an):
    '''Count number of records per nn_strings (per file_type).

    Args:
        an (pd.DataFrame): all-names table, relevant columns only (where drop_from_nm == 0)

            =======================  =======================================================
            record_id                unique record identifier
            blockstring              concatenation of all the blocking variables (sep by ::)
            file_type                either "new" or "existing"
            drop_from_nm             flag, 1 if met any "to drop" criteria 0 otherwise
            <nn-blocking column(s)>  variables for near-neighbor blocking
            <ed-blocking column>     variable for edit-distance blocking
            <av-blocking column>     (optional) variable for abs-value blocking
            nn_string                concatenated version of nm-blocking columns (sep by ::)
            ed_string                copy of ed-blocking column
            absval_string            copy of ed-blocking column
            =======================  =======================================================

    Return:
        dict: two keys (new and existing), mapping to a dictionary of nn_strings to n_records
    '''

    an = an.copy()

    logger.trace('Counting nn_strings.')
    temp = an.groupby(['file_type', 'nn_string']).size().reset_index().pivot(
            index='nn_string', columns='file_type')
    temp.columns = temp.columns.droplevel()
    temp = temp.fillna(0)
    nn_strings_to_num_recs = temp.to_dict('dict')

    if 'existing' not in nn_strings_to_num_recs:
        nn_strings_to_num_recs['existing'] = defaultdict(int)

    return nn_strings_to_num_recs


def get_common_name_penalties(clean_last_names, max_penalty, num_threshold_bins=1000):
    '''Create a dictionary mapping each last name to a "commonness penalty." Two SMITHs are
    less likely to be the same person than two HANDAs, since SMITH is such a common name. This
    function quantifies this penalty for use in later blocking calculations. A more common name
    recieves a higher number, topping out at max_penalty.

    Args:
        clean_last_names (pd.Series): clean (un-split) last name column (one row per record)
        max_penalty (float): the maximum penalty (for the most common names)
        num_threshold_bins (int): number of different categories of commonnness to create

    Returns:
        dict: dictionary mapping name (str) to penalty (float)
    '''

    logger.trace('Getting last names frequencies.')
    last_name_freq_df = clean_last_names.groupby(clean_last_names).size()
    last_name_freq_df.index.name = 'last_name'
    last_name_freq_df.name = 'ln_count'
    last_name_freq_df = last_name_freq_df.reset_index()

    commonness_penalty_lookup = defaultdict(lambda: 0)

    last_name_freq_df['norm_ln_count'] = \
            last_name_freq_df.ln_count / last_name_freq_df.ln_count.sum()
    last_name_freq_df = last_name_freq_df.sort_values('norm_ln_count', ascending=False)

    last_name_freq_df['bin'] = pd.qcut(
            last_name_freq_df.norm_ln_count,
            num_threshold_bins,
            labels=False, duplicates='drop')
    threshold_bins = last_name_freq_df['bin'].unique()
    threshold_levels = pd.DataFrame(data={
            'bin' : threshold_bins,
            'threshold' : np.linspace(max_penalty, 0, len(threshold_bins))})
    last_name_freq_df = pd.merge(last_name_freq_df, threshold_levels, on='bin')
    last_name_freq_df.set_index('last_name', inplace=True)
    commonness_penalty_lookup = defaultdict(
            lambda: max_penalty, last_name_freq_df.threshold.to_dict())

    return commonness_penalty_lookup


def get_all_shingles():
    '''Get all valid 2-shingles.

    Returns:
        list: valid 2-shingles
    '''

    valid_characters = string.ascii_uppercase + ' *'
    all_tuples = list(itertools.product(valid_characters,valid_characters))
    all_two_shingles = [tup[0] + tup[1] for tup in all_tuples]
    valid_two_shingles = [sh for sh in all_two_shingles if sh not in ['  ', '* ', ' *']]
    # NOTE: leave ** in case we ever allow blank nn_strings

    return valid_two_shingles


def prep_index():
    '''Initialize index data structure, which will store similarity information
    about the names, and load processed shingles into it.

    Returns:
        nnmslib.FloatIndex: nmslib index object (pre time-consuming build call)
    '''

    # intialize index
    space_type = 'cosinesimil_sparse'
    space_params = {}
    method_name = 'hnsw'

    ix = nmslib.init(
        space=space_type,
        space_params=space_params,
        method=method_name,
        data_type=nmslib.DataType.SPARSE_VECTOR,
        dtype=nmslib.DistType.FLOAT)

    return ix


def get_second_index_nn_strings(all_nn_strings, main_nn_strings):
    '''Get nn_strings that haven't already been stored in the main index.

    Args:
        all_nn_strings (list): list of all nn_strings in the data (expanded if split_names is True)
        main_nn_strings (list): list of nn_strings already in the main index

    Returns:
        list: the nn_strings that are not in main_nn_strings
    '''

    all_nn_strings = all_nn_strings[:]
    main_nn_strings = main_nn_strings[:]

    all_nn_strings_s = pd.Series(all_nn_strings)
    second_index_nn_strings = all_nn_strings_s[
            all_nn_strings_s.isin(set(main_nn_strings)) == False].tolist()

    return second_index_nn_strings


def save_main_index(main_index, main_index_nn_strings, main_index_file):
    '''Save the main nmslib index and pickle dump the associated nn_strings list.

    Args:
        main_index (nmslib.FloatIndex): the main, built nmslib index
        main_index_nn_strings (list): list of nn_strings in the main index
        main_index_file (str): path to store the main nmslib index
    '''

    main_index.saveIndex(main_index_file, save_data=True)

    with open(main_index_file + '.pkl', 'wb') as pf:
        pickle.dump(main_index_nn_strings, pf, protocol=pickle.HIGHEST_PROTOCOL)


def load_main_index_nn_strings(og_blocking_index_file):
    '''Load the nn_strings that are in an existing nmslbi index file.

    Args:
        og_blocking_index_file (str): path to original blocking index

    Returns:
        list: loaded list of nn_strings in an existing nmslib index
    '''

    with open(og_blocking_index_file + '.pkl', 'rb') as f:
        main_index_nn_strings = pickle.load(f)

    return main_index_nn_strings


def write_some_cps(cand_pairs, candidate_pairs_file, header=False):
    '''Write out a portion of the candidate-pairs.

    Args:
        cand_pairs (pd.DataFrame): chunk of the candidate-pairs file
        candidate_pairs_file (str): path to the candidate-pairs file
        header (bool): True if this is the first time calling this function
    '''

    cand_pairs.to_csv(
            candidate_pairs_file,
            mode='a',
            index=False,
            header=header,
            quoting=csv.QUOTE_NONNUMERIC)


def generate_true_pairs(must_links_df):
    """Reduce the must-link records pairs must-link blockstring pairs.

    Args:
        must_links_df (pd.DataFrame): list of must-link record pairs

            ===================   =======================================================
            record_id_1           unique identifier for the first record in the pair
            record_id_2           unique identifier for the second record in the pair
            blockstring_1         blockstring for the first record in the pair
            blockstring_2         blockstring for the second record in the pair
            drop_from_nm_1        flag, 1 if the first record in the pair was not eligible for matching
            drop_from_nm_2        flag, 1 if the second record in the pair was not eligible for matching
            existing              flag, 1 if the pair is must-link because of ExistingID
            ===================   =======================================================

    Return:
        pd.DataFrame: list of must-link blockstring pairs (where both record have drop_from_nm == 0)

        ===================   =======================================================
        blockstring_1         blockstring for the first record in the pair
        blockstring_2         blockstring for the second record in the pair
        ===================   =======================================================
    """

    must_links_df = must_links_df[(must_links_df.drop_from_nm_1 == 0) &
                                  (must_links_df.drop_from_nm_2 == 0)]

    true_pairs_df = must_links_df[['blockstring_1', 'blockstring_2']].drop_duplicates()

    return true_pairs_df

