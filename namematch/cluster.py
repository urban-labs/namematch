import editdistance
import importlib.util
import logging
import networkx as nx
import numpy as np
import pandas as pd
import pickle
import os
import time
import gc

from typing import Union
from collections import deque

import pyarrow as pa
import pyarrow.parquet as pq

from namematch.base import NamematchBase
from namematch.data_structures.parameters import Parameters
from namematch.data_structures.schema import Schema
from namematch.utils.utils import log_runtime_and_memory, load_parquet
from namematch.default_constraints import get_columns_used, is_valid_link, is_valid_cluster, apply_link_priority
from namematch.utils.profiler import Profiler


profile = Profiler()

logger = logging.getLogger()

class Constraints(object):

    @property
    def get_columns_used(self):
        try:
            return self._get_columns_used

        except AttributeError:
            logging.warning(f"Constraints.get_columns_used wasn't set. Use default function from {get_columns_used.__module__}")
            return get_columns_used

    @get_columns_used.setter
    def get_columns_used(self, func: callable):
        self._get_columns_used = func

    @property
    def is_valid_link(self):
        try:
            return self._is_valid_link

        except AttributeError:
            logging.warning(f"Constraints.is_valid_link wasn't set. Use default function from {is_valid_link.__module__}")
            return is_valid_link

    @is_valid_link.setter
    def is_valid_link(self, func: callable):
        self._is_valid_link = func

    @property
    def is_valid_cluster(self):
        try:
            return self._is_valid_cluster

        except AttributeError:
            logging.warning(f"Constraints.is_valid_cluster wasn't set. Use default function from {is_valid_cluster.__module__}")
            return is_valid_cluster

    @is_valid_cluster.setter
    def is_valid_cluster(self, func: callable):
        self._is_valid_cluster = func

    @property
    def apply_link_priority(self):
        try:
            return self._apply_link_priority

        except AttributeError:
            logging.warning(f"Constraints.apply_link_priority wasn't set. Use default function from  {apply_link_priority.__module__}")
            return apply_link_priority

    @apply_link_priority.setter
    def apply_link_priority(self, func: callable):
        self._apply_link_priority = func


class Cluster(NamematchBase):
    '''
    Args:
        params (Parameters object): contains parameter values
        schema (Schema object): contains match schema info (files to match, variables to use, etc.)
        constraints (str or Constrants object): either a path to python script defining constraint functions or a Constraints object
        must_links_file (str): path to the must-links file
        potential_edges_dir (str): path to the potential-links dir in the output/details folder
        flipped0_edges_file (str): path to the flipped-links file
        all_names_file (str): path to the all-names file
        cluster_assignments(str): path to the cluster-assignments file
    '''
    def __init__(
        self,
        params,
        schema,
        must_links_file="must_links.csv",
        potential_edges_dir="potential_links",
        flipped0_edges_file="flipped0_potential_links.csv",
        all_names_file="all_names.parquet",
        cluster_assignments="cluster_assignments.pkl",
        edges_to_cluster="edges_to_cluster.parquet",
        constraints: Union[str, Constraints]=None,
        *args,
        **kwargs
    ):
        super(Cluster, self).__init__(params, schema, *args, **kwargs)

        self.constraints = constraints
        self.must_links_file = must_links_file
        self.potential_edges_dir = potential_edges_dir
        self.flipped0_edges_file = flipped0_edges_file
        self.all_names_file = all_names_file
        self.cluster_assignments = cluster_assignments
        self.edges = edges_to_cluster

    @property
    def output_files(self):
        return [self.cluster_assignments, self.edges]

    @log_runtime_and_memory
    @profile
    def main(self, **kw):
        '''Read the record pairs with high probability of matching and connect them in a way
        that doesn't violate any logic constraints to form clusters.
        '''
        # loading cluster_logic functions
        cluster_logic = self.get_cluster_logic(self.constraints)

        # loading must links
        logger.info("Loading must-link links.")
        must_links_df = pd.read_csv(self.must_links_file, low_memory=False)
        # get uid and eid cols
        uid_cols = self.schema.variables.get_variables_where(
                attr='compare_type', attr_value='UniqueID')
        try:
            eid_col = self.schema.variables.get_variables_where(
                    attr='compare_type', attr_value='ExistingID')[0]
        except:
            eid_col = None

        # load info needed for constraint checking
        logger.info("Creating dictionary of cluster information.")
        cluster_info = self.load_cluster_info(self.all_names_file, uid_cols, eid_col, cluster_logic)

        # separate must-links if we can't initialize new 1s
        if not self.params.initialize_from_ground_truth_1s or self.params.incremental:
            gt_1s_df = must_links_df[must_links_df.existing == 0].copy()
            must_links_df = must_links_df[must_links_df.existing == 1].copy()
        else:
            gt_1s_df = None

        # create a starting point using must-links
        logger.info("Initializing initial clusters.")
        clusters, cluster_assignments, original_cluster_ids = \
                self.get_initial_clusters(must_links_df, cluster_info, eid_col)
        # potential_edges is sorted (decreasing) by phat
        logger.info("Loading potential links.")
        potential_edges_files = [os.path.join(self.potential_edges_dir, pe_file)
                                 for pe_file in os.listdir(self.potential_edges_dir)]
        self.get_potential_edges(
                potential_edges_files, self.flipped0_edges_file, gt_1s_df, cluster_logic,
                cluster_info, uid_cols, eid_col)

        logger.info("Clustering potential links.")
        cluster_assignments = self.cluster_potential_edges(
                clusters, cluster_assignments, original_cluster_ids, cluster_info,
                cluster_logic, uid_cols, eid_col)

        with open(self.cluster_assignments, "wb") as f:
            pickle.dump(cluster_assignments, f)

        if self.enable_lprof:
            self.write_line_profile_stats(profile.line_profiler)

    def get_cluster_logic(self, constraints):
        if isinstance(constraints, str):
            spec = importlib.util.spec_from_file_location("module.name", constraints)
            constraints = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(constraints)
            logger.info(f"Set up cluster constraints {constraints}")

        elif isinstance(constraints, Constraints):
            logger.info(f"Set up cluster constraints {constraints}")
            constraints = self.constraints

        else:
            if constraints is not None:
                logger.warning(f"'constraints' {constraints} is not recognized! Using the default link and cluster constraints.")
            else:
                logger.warning(f"No 'constraints' provided: using the default link constraints, cluster constraints, and edge priority.")
            import namematch.default_constraints as constraints

        cluster_logic = Constraints()
        cluster_logic.get_columns_used = constraints.get_columns_used
        cluster_logic.is_valid_link = constraints.is_valid_link
        cluster_logic.is_valid_cluster = constraints.is_valid_cluster
        cluster_logic.apply_link_priority = constraints.apply_link_priority
        cluster_logic.enable_lprof = self.enable_lprof

        return cluster_logic

    @profile
    def auto_is_valid_edge(
                self,
                edges_df,
                uid_cols, allow_clusters_w_multiple_unique_ids, leven_thresh,
                eid_col=None):
        '''Check if two records would violate a unique id or existing id constraint.

        Args:
            edges_df (pd.DataFrame): potential edges information

                ================      =======================================================
                record_id_1           unique record identifier (for first in pair)
                record_id_2           unique record identifier (for second in pair)
                phat                  predicted probability of a record pair being a match
                original_order        original ordering 1-N (useful so gt is always on top of phat=1 cases)
                ================      =======================================================

            uid_cols (list): all-names column(s) with compare_type UniqueID
            allow_clusters_w_multiple_unique_ids (bool): True if a cluster can have multiple uid values
            leven_thresh (int): n character edits to allow between uids before they're considered different
            eid_col (str): all-names column with compare_type ExistingID (None for non-incremental runs)

        Returns:
            valid_edges_df: potential edges information, but limited to rows that pass the automated validity check
        '''

        edges_df['valid'] = True

        if eid_col is not None:
            # if both existing ids are known, they cannot be different
            both_eids_known_ix = (edges_df[f"{eid_col}_1"].notnull()) & (edges_df[f"{eid_col}_2"].notnull())
            edges_df.loc[both_eids_known_ix & (edges_df[f"{eid_col}_1"] != edges_df[f"{eid_col}_2"]), 'valid'] = False

        if allow_clusters_w_multiple_unique_ids:
            return edges_df[edges_df.valid == True]

        # NOTE if you've made it here, allow_clusters_w_multiple_unique_ids is False
        # if both unique ids are known, they cannot be different
        edges_df['violations'] = 0
        edges_df['attempts'] = 0
        for uid_col in uid_cols:

            both_uids_known_ix = (edges_df[f"{uid_col}_1"].notnull()) & (edges_df[f"{uid_col}_2"].notnull())
            edges_df.loc[both_uids_known_ix, 'attempts'] = edges_df[both_uids_known_ix].attempts + 1

            uids_different_ix = edges_df[f"{uid_col}_1"] != edges_df[f"{uid_col}_2"]

            if leven_thresh is None:
                edges_df.loc[both_uids_known_ix & uids_different_ix, 'violations'] = \
                    edges_df[both_uids_known_ix & uids_different_ix].violations + 1
            else:
                edges_df.loc[both_uids_known_ix & uids_different_ix, f'{uid_col}_edit_dist'] = \
                    np.vectorize(editdistance.eval, otypes=['float'])(
                        edges_df[both_uids_known_ix & uids_different_ix][f"{uid_col}_1"].values,
                        edges_df[both_uids_known_ix & uids_different_ix][f"{uid_col}_2"].values)
                leven_violated = edges_df[f'{uid_col}_edit_dist'] > leven_thresh
                edges_df.loc[both_uids_known_ix & uids_different_ix & leven_violated, 'violations'] = \
                    edges_df[both_uids_known_ix & uids_different_ix & leven_violated].violations + 1

        edges_df.loc[(edges_df.attempts > 0) & (edges_df.attempts == edges_df.violations), 'valid'] = False

        return edges_df[edges_df.valid == True]

    @profile
    def auto_is_valid_cluster(
                self,
                cluster,
                uid_cols, allow_clusters_w_multiple_unique_ids, leven_thresh,
                eid_col=None):
        '''Check if a proposed cluster would violate a unique id or existing id constraint.

        Args:
            cluster (pd.DataFrame): all-names file (relevant columns only) records for the proposed cluster
            uid_cols (list): all-names column(s) with compare_type UniqueID
            allow_clusters_w_multiple_unique_ids (bool): True if a cluster can have multiple uid values
            leven_thresh (int): n character edits to allow between uids before they're considered different
            eid_col (str): all-names column with compare_type ExistingID (None for non-incremental runs)

        Returns:
            bool: False if an automated constraint is violated
        '''

        if eid_col is not None:

            # at most one existing id in a cluster
            n_existing_ids = cluster[eid_col].nunique()
            if n_existing_ids > 1:
                return False

        if allow_clusters_w_multiple_unique_ids:
            return True

        # NOTE if you've made it here, allow_clusters_w_multiple_unique_ids is False

        # at most one unique id in a cluster
        for uid_col in uid_cols:

            n_unique_ids = cluster[uid_col].nunique()
            if n_unique_ids > 1:

                if leven_thresh is None:
                    return False
                else:
                    # make sure that each non-NA uid has one other non-NA uid
                    # within leven_thresh edits
                    cluster.reset_index(inplace=True)
                    uid_df = cluster[~cluster[uid_col].isnull()][[uid_col]]
                    uid_df['dummy'] = 1
                    uid_df = pd.merge(uid_df, uid_df, on='dummy', suffixes = ['_1', '_2'])
                    uid_df = uid_df[uid_df[f'{uid_col}_1'] != uid_df[f'{uid_col}_2']].copy()
                    uid_df['ed'] = np.vectorize(editdistance.eval)(
                            uid_df[f'{uid_col}_1'].values, uid_df[f'{uid_col}_2'].values)
                    min_ed = uid_df.groupby(f'{uid_col}_1').ed.min()
                    return (min_ed > leven_thresh).sum() == 0

        return True

    @log_runtime_and_memory
    @profile
    def get_initial_clusters(self, must_links_df, an_df, eid_col, **kw):
        '''Use must links (ground truth and/or a previous run) to create the
        starting clusters.

        Args:
            must_links_df (pd.DataFrame): record pairs that must be linked together no matter what

                ==============  ======================================================================
                record_id_1     unique identifier for the first record in the pair
                record_id_2     unique identifier for the second record in the pair
                blockstring_1   blockstring for the first record in the pair
                blockstring_2   blockstring for the second record in the pair
                drop_from_nm_1  flag, 1 if the first record in the pair was not eligible for matching
                drop_from_nm_2  flag, 1 if the second record in the pair was not eligible for matching
                existing        flag, 1 if the pair is must-link because of ExistingID
                ==============  ======================================================================

            an_df (pd.DataFrame): all-names file, with only the columns relevant for clustering

                ===========================  =======================================================
                record_id                    unique record identifier
                <uid column(s)>              columns with compare_type UniqueID
                <eid column(s)>              columns with compare_type ExistingID
                <user-constraint column(s)>  (optional) columns mentioned in `get_columns_used()`
                ===========================  =======================================================

            eid_col (str): all-names column with compare_type ExistingID, or None

        Returns:
            dict: clusters maps a cluster id to a list of record ids
            dict: cluster_assignments maps a record_id to a cluster_id
            set: cluster ids that are already in use (only for incremental)
        '''

        # clusters maps a cluster id to a list of record ids
        # cluster_assignments maps a record_id to a cluster_id
        if eid_col is not None: # incremental

            # make dictionary mapping cluster id to a list of records in that cluster
            eid_df = an_df[an_df[eid_col].notnull() & (an_df[eid_col] != '')].copy()
            eid_df[eid_col] = eid_df[eid_col].astype(int)
            cluster_assignments = eid_df[[eid_col]].to_dict()[eid_col]
            clusters = {}
            for (record_id, cluster_id) in cluster_assignments.items():
                if (cluster_id in clusters):
                    clusters[cluster_id].append(record_id)
                else:
                    clusters[cluster_id] = [record_id]

            original_cluster_ids = set(clusters.keys())

            if len(clusters) > 0:
                # for later assign the cluster_id
                i = max(clusters.keys())

        else: # from scratch
            clusters = {}

            if len(must_links_df) > 0:
                reversed_must_links_df = pd.DataFrame({
                    "record_id_1": must_links_df["record_id_2"],
                    "record_id_2": must_links_df["record_id_1"]
                })
                must_links_df = pd.concat([must_links_df, reversed_must_links_df], join='inner')

                # find the clusters, ie connected components
                g = nx.from_pandas_edgelist(
                        must_links_df,
                        source="record_id_1",
                        target="record_id_2")
                for i, comp in enumerate(nx.connected_components(g)):
                    clusters[i] = list(comp)

            cluster_assignments = {
                record_id : cluster_id
                for cluster_id, record_ids in clusters.items()
                for record_id in record_ids
            }

            original_cluster_ids = None

        # add in the singletons
        if len(clusters) == 0:
            i = -1

        cluster_id = i + 1
        logging.debug(f"new cluster_id for singletons starts from: {cluster_id}")

        an_df = an_df.reset_index()
        an_df = an_df[an_df.record_id.isin(cluster_assignments.keys()) == False].reset_index(drop=True)
        an_df['cluster_id'] = an_df.index + cluster_id
        # NOTE: next 4 lines are for a 1000x speedup over pandas groupby
        keys, values = an_df[['cluster_id', 'record_id']].values.T
        ukeys, index = np.unique(keys, True)
        arrays = np.split(values, index[1:])
        singleton_clusters = dict(zip(ukeys, [list(a) for a in arrays]))
        clusters.update(singleton_clusters)
        cluster_assignments.update(an_df.set_index('record_id').to_dict()['cluster_id'])
        self.stats_dict['n_initial_clusters'] = len(clusters)
        del must_links_df
        gc.collect()
        return clusters, cluster_assignments, original_cluster_ids

    @log_runtime_and_memory
    def save_df_to_disk(self, df):
        table = pa.Table.from_pandas(df)
        pq.write_table(table, self.edges)

    @log_runtime_and_memory
    @profile
    def get_potential_edges(self, potential_edges_files, flipped0_edges_file, gt_1s_df, cluster_logic, cluster_info, uid_cols, eid_col, **kw):
        '''
        Use all predictions file to make a list of edges that the constrained
        clustering algorithm should try to add.

        Args:
            potential_edges_files (list): paths to the potential links files
            flipped0_edges_file (str): path to the flipped0-links file
            gt_1s_df (pd.DataFrame): for incremental runs, subset of the must-link df that are "new" 1s
            cluster_logic (module): user-defined constraint functions
            cluster_info (pd.DataFrame): all-names file, with only the columns relevant for clustering
            uid_cols (list): all-name columns with compare_type UniqueID
            eid_col (str): all-name column with compare_type ExistingID

        '''
        total_auto_invalid_edges = 0
        total_invalid_edges = 0
        for i, potential_edges_file in enumerate(potential_edges_files):

            potential_edges_df = load_parquet(potential_edges_file, conditions_dict={'potential_edge': 1})
            potential_edges_df['gt'] = 0

            if i == 0: # for first iteration only
                if (gt_1s_df is not None) and (os.path.exists(flipped0_edges_file)):
                    flipped0_edges_file = pd.read_csv(flipped0_edges_file, low_memory=False)
                    flipped0_edges_file['gt'] = 0
                    potential_edges_df = pd.concat([potential_edges_df, flipped0_edges_file], join='inner')
                    potential_edges_df = potential_edges_df.reset_index(drop=True)

            # get clustering_phat
            potential_edges_df['phat'] = -1
            for model_name in potential_edges_df.model_to_use.unique():
                potential_edges_df.loc[potential_edges_df.model_to_use == model_name, 'phat'] = \
                        potential_edges_df['%s_match_phat' % model_name]

            cols = ['record_id_1', 'record_id_2', 'gt', 'phat']
            potential_edges_df = potential_edges_df[cols]

            if i == 0: # for first iteration only
                if gt_1s_df is not None:
                    gt_1s_df['gt'] = 1
                    gt_1s_df['phat'] = 1
                    potential_edges_df = pd.concat([potential_edges_df, gt_1s_df[cols]])

            potential_edges_df = potential_edges_df.reset_index(drop=True)

            logger.trace('Dropping invalid edges')
            potential_edges_df = pd.merge(potential_edges_df, cluster_info, left_on='record_id_1', right_index=True)
            potential_edges_df = pd.merge(potential_edges_df, cluster_info, left_on='record_id_2', right_index=True, suffixes=['_1', '_2'])
            potential_edges_df = potential_edges_df.reset_index(drop=True)

            starting_cols = potential_edges_df.columns.tolist()
            n_before_drop_auto_invalid_edges = len(potential_edges_df)
            potential_edges_df = self.auto_is_valid_edge(
                    potential_edges_df, uid_cols,
                    self.params.allow_clusters_w_multiple_unique_ids, self.params.leven_thresh, eid_col)
            potential_edges_df = potential_edges_df[starting_cols]
            n_before_drop_invalid_edges = len(potential_edges_df)
            potential_edges_df['valid'] = cluster_logic.is_valid_link(potential_edges_df)
            potential_edges_df = potential_edges_df[
                (potential_edges_df.gt == 1) | (potential_edges_df.valid == 1)]
            potential_edges_df = potential_edges_df[starting_cols]
            auto_invalid_edges = n_before_drop_auto_invalid_edges - n_before_drop_invalid_edges
            invalid_edges = n_before_drop_invalid_edges - len(potential_edges_df)

            total_auto_invalid_edges += auto_invalid_edges
            total_invalid_edges += invalid_edges

            if i == 0:
                valid_potential_edges_df = potential_edges_df.copy()
            else:
                valid_potential_edges_df = pd.concat([valid_potential_edges_df, potential_edges_df])

            del potential_edges_df
            gc.collect()
        logger.info(f"Auto-invalid links: {total_auto_invalid_edges}")
        logger.info(f"Invalid links: {total_invalid_edges}")
        self.stats_dict['n_invalid_links'] = total_invalid_edges

        valid_potential_edges_df = valid_potential_edges_df.reset_index(drop=True)
        valid_potential_edges_df = valid_potential_edges_df.sort_values(
                ['gt', 'phat'], ascending=[False, False])

        logger.trace('Applying link priority.')
        valid_potential_edges_df['original_order'] = valid_potential_edges_df.index
        valid_potential_edges_df = cluster_logic.apply_link_priority(valid_potential_edges_df)

        self.stats_dict['n_potential_edges'] = len(valid_potential_edges_df)
        # NOTE: takes a little while, but worth it later to avoid iterrows

        logger.trace('Saving valid_links_df to disk as parquet file')

        self.save_df_to_disk(valid_potential_edges_df)
        del valid_potential_edges_df
        gc.collect()

    @log_runtime_and_memory
    @profile
    def load_cluster_info(self, all_names_file, uid_cols, eid_col, cluster_logic, **kw):
        '''Read in the all_names information needed for cluster constraint checking. Columns
        defined in the config as compare type UniqueID or ExistingID will automatically be loaded
        (as strings, with missing values represented as NA). Other columns you wish to be loaded
        should be defined in the user-defined `get_columns_used()` function.

        Args:
            all_names_file (str): path to the all-names file
            uid_cols (list): all-name columns with compare_type UniqueID
            eid_col (str): all-name column with compare_type ExistingID
            cluster_logic (module): user-defined constraint functions

        Returns:
            pd.DataFrame: all-names file, with only the columns relevant for clustering

            ===========================  =======================================================
            record_id                    unique record identifier
            <uid column(s)>              columns with compare_type UniqueID
            <eid column(s)>              columns with compare_type ExistingID
            <user-constraint column(s)>  (optional) columns mentioned in `get_columns_used()`
            ===========================  =======================================================
        '''

        # add things that are missing from get_columns_used

        cols_needed = cluster_logic.get_columns_used()
        if cols_needed == 'all':
            column_dtypes = {field:'object' for field in self.schema.variables.get_an_column_names()}
        else: 
            column_dtypes = cols_needed.copy()
            if ("record_id" not in column_dtypes):
                column_dtypes["record_id"] = 'object'
            if ("dataset" not in column_dtypes):
                column_dtypes["dataset"] = 'object'

        id_cols_to_load_as_obj = uid_cols
        if eid_col is not None:
            id_cols_to_load_as_obj.append(eid_col)

        for col in id_cols_to_load_as_obj:
            if col in column_dtypes:
                if column_dtypes[col] != 'object':
                    logger.warning(f"Changing dtype for {col} field to object.")
                    column_dtypes[col] = 'object'
                logger.warning(f"Note, missing values in {col} field will not be "
                               f"filled with '', since this is an ID field.")
            else:
                column_dtypes[col] = 'object'

        table = pq.read_table(all_names_file)
        cluster_info =  table.to_pandas()
        cluster_info = cluster_info[list(column_dtypes.keys())]
        del column_dtypes['record_id']
        cluster_info = cluster_info.set_index('record_id')
        # change the dtypes in all_names based on user-defined dtypes in constraints
        for col, col_dtype in column_dtypes.items():
            if col_dtype in ['int', 'float']:
                cluster_info[col] = pd.to_numeric(cluster_info[col])
            elif col_dtype in ['date']:
                cluster_info[col] = pd.to_datetime(cluster_info[col])
            elif col_dtype in ['str', 'string', 'object']:
                cluster_info[col] = cluster_info[col].fillna("")
                cluster_info[col] = cluster_info[col].astype(str)

        # replace empty string with NaN for uids and eid
        for col in id_cols_to_load_as_obj:
            cluster_info[col] = cluster_info[col].replace('', np.nan)
        return cluster_info

    @log_runtime_and_memory
    def get_ci_ix_map(self, cluster_info):
        ci_ix_map = dict(zip(cluster_info.index.tolist(), range(len(cluster_info))))
        return ci_ix_map

    @log_runtime_and_memory
    @profile
    def cluster_potential_edges(self, clusters, cluster_assignments, original_cluster_ids,
                cluster_info, cluster_logic, uid_cols, eid_col, **kw):
        '''For clusters by add potential edges to the cluster graph in order of importance, skipping those
        that cause violations.

        Args:
            clusters (dict): maps a cluster id to a list of record ids -- post initialization
            cluster_assignments (dict): maps a record_id to a cluster_id -- post initialization
            original_cluster_ids (set): set: cluster ids that are already in use (only for incremental)
            cluster_info (pd.DataFrame): all-names file, with only the columns relevant for clustering

                ===========================  =======================================================
                record_id                    unique record identifier
                <uid column(s)>              columns with compare_type UniqueID
                <eid column(s)>              columns with compare_type ExistingID
                <user-constraint column(s)>  (optional) columns mentioned in `get_columns_used()`
                ===========================  =======================================================

            potential_edges (deque): each element is a dict version of a potential edge's record
            cluster_logic (module): user-defined constraint functions
            uid_cols (list): all-name columns with compare_type UniqueID
            eid_col (str): all-name column with compare_type ExistingID

        Returns:
            dict: maps record_id to cluster_id
        '''
        # track things
        invalid_edges = 0
        invalid_clusters = 0
        merges = 0
        pf = pq.ParquetFile(self.edges)
        nrows = pf.metadata.num_rows
        logger.info(f"total number of edges: {nrows}")

        # create ix map so can use faster iloc as opposed to loc
        ci_ix_map = self.get_ci_ix_map(cluster_info)
        i = 0
        logger.debug(f"batch size: {self.params.cluster_batch_size}")
        for edge_pf in pf.iter_batches(batch_size=self.params.cluster_batch_size, use_threads=True):
            for edge in edge_pf.to_pylist():
                if (self.params.verbose is not None) and (i % self.params.verbose == 0):
                    logger.info(f"  Checked {i} of {nrows} edges: {invalid_clusters} invalid clusters, {merges} merges.")

                record_id_1 = edge["record_id_1"]
                record_id_2 = edge["record_id_2"]

                edge_is_gt = (edge["gt"] == 1)

                cluster_id_1 = cluster_assignments[record_id_1]
                cluster_id_2 = cluster_assignments[record_id_2]


                if (cluster_id_1 != cluster_id_2):
                    cluster_1 = clusters[cluster_id_1]
                    cluster_2 = clusters[cluster_id_2]
                    new_cluster = cluster_1 + cluster_2

                    if len(new_cluster) > 2:
                        new_cluster_info = cluster_info.iloc[[ci_ix_map[ncr] for ncr in new_cluster]].copy()

                        new_cluster_info['cluster'] = \
                                new_cluster_info.index.isin(cluster_1).astype(int) # differentiate

                        new_cluster_info['new_edge'] = \
                                new_cluster_info.index.isin([record_id_1, record_id_2])

                        # NOTE: new_edge is a way of checking an edge constraint within
                        #       is_valid_cluster (in case you want to enforce an edge
                        #       constraint, but only for clusters that meet some criteria)

                    # if len of new_cluster is 2 then in theory we already vetted by checking the edge
                    # (assuming that all cluster constraints were put into the edge constraint list if possible)

                    if len(new_cluster) > 2:
                        cluster_auto_valid = self.auto_is_valid_cluster(
                            new_cluster_info, uid_cols,
                            self.params.allow_clusters_w_multiple_unique_ids, self.params.leven_thresh, eid_col)

                    if  (len(new_cluster) == 2) or \
                        (cluster_auto_valid and (edge_is_gt or cluster_logic.is_valid_cluster(new_cluster_info, edge['phat']))):

                        new_cluster_id = min(cluster_id_1, cluster_id_2)

                        if original_cluster_ids is not None:
                            if cluster_id_1 in original_cluster_ids:
                                new_cluster_id = cluster_id_1
                            elif cluster_id_2 in original_cluster_ids:
                                new_cluster_id = cluster_id_2

                        clusters.pop(cluster_id_1)
                        clusters.pop(cluster_id_2)
                        clusters[new_cluster_id] = new_cluster

                        for record_id in new_cluster:
                            cluster_assignments[record_id] = new_cluster_id

                        merges += 1

                    else:
                        invalid_clusters += 1

                i += 1

        logger.info(f"Invalid clusters: {invalid_clusters}")
        self.stats_dict['n_invalid_clusters'] = invalid_clusters
        logger.info(f"n_merges: {merges}")
        n_clusters = len(clusters)
        logger.info(f"Number of clusters total: {n_clusters}")
        self.stats_dict['n_clusters'] = n_clusters
        n_singleton_clusters = len([recs for c_id, recs in clusters.items() if (len(recs) == 1)])
        logger.info(f"Number of singleton clusters: {n_singleton_clusters}")
        self.stats_dict['n_singleton_clusters'] = n_singleton_clusters

        cluster_assignments = {k: str(v) for k, v in cluster_assignments.items()}
        return cluster_assignments

