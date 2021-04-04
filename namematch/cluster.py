import argparse
import editdistance
import importlib.util
import logging
import networkx as nx
import numpy as np
import pandas as pd
import pickle
import os
import sqlite3
import time
import yaml

import pyarrow as pa
import pyarrow.parquet as pq

from namematch.base import NamematchBase
from namematch.data_structures.parameters import Parameters
from namematch.data_structures.schema import Schema
from namematch.utils.utils import equip_logger_id, log_runtime_and_memory, load_parquet_list, load_yaml
from namematch.default_cluster_constraints import get_columns_used, is_valid_edge, is_valid_cluster, apply_edge_priority


try:
    profile
except:
    from line_profiler import LineProfiler
    profile = LineProfiler()


logger = logging.getLogger()

class ClusterConstraints(object):

    @property
    @profile
    def get_columns_used(self):
        try:
            return self._get_columns_used

        except AttributeError:
            logging.warning(f"ClusterConstraints.get_columns_used wasn't set. Use default function from {get_columns_used.__module__}")
            return get_columns_used

    @get_columns_used.setter
    def get_columns_used(self, func: callable):
        self._get_columns_used = func

    @property
    @profile
    def is_valid_edge(self):
        try:
            return self._is_valid_edge

        except AttributeError:
            logging.warning(f"ClusterConstraints.is_valid_edge wasn't set. Use default function from {is_valid_edge.__module__}")
            return is_valid_edge

    @is_valid_edge.setter
    def is_valid_edge(self, func: callable):
        self._is_valid_edge = func

    @property
    @profile
    def is_valid_cluster(self):
        try:
            return self._is_valid_cluster

        except AttributeError:
            logging.warning(f"ClusterConstraints.is_valid_cluster wasn't set. Use default function from {is_valid_cluster.__module__}")
            return is_valid_cluster

    @is_valid_cluster.setter
    def is_valid_cluster(self, func: callable):
        self._is_valid_cluster = func

    @property
    @profile
    def apply_edge_priority(self):
        try:
            logging.warning(f"ClusterConstraints.apply_edge_priority wasn't set. Use default function from  {apply_edge_priority.__module__}")
            return self._apply_edge_priority

        except AttributeError:
            return apply_edge_priority

    @apply_edge_priority.setter
    def apply_edge_priority(self, func: callable):
        self._apply_edge_priority = func


class Cluster(NamematchBase):
    def __init__(
        self,
        params,
        schema,
        must_links_file,
        potential_edges_dir,
        flipped0_edges_file,
        all_names_file,
        output_file,
        constraints_file=None,
        logger_id=None,
        *args,
        **kwargs
    ):
        super(Cluster, self).__init__(params, schema, output_file, logger_id, *args, **kwargs)

        self.constraints_file = constraints_file
        self.must_links_file = must_links_file
        self.potential_edges_dir = potential_edges_dir
        self.flipped0_edges_file = flipped0_edges_file
        self.all_names_file = all_names_file

    @equip_logger_id
    @log_runtime_and_memory
    def main__cluster(self, **kw):
        '''Read the record pairs with high probability of matching and connect them in a way
        that doesn't violate any logic constraints to form clusters.

        Args:
            params (Parameters object): contains parameter values
            schema (Schema object): contains match schema info (files to match, variables to use, etc.)
            constraints_file (str): path to python file with the three required cluster constraint functions
            must_links_file (str): path to output_temp's must-links file
            potential_edges_dir (str): path to output_temp's potential-edges file
            flipped0_edges_file (str): path to output_temp's flipped-edges file
            all_names_file (str): path to output_temp's all-names file
            output_file (str): path to output_temp's cluster-assignments file
        '''

        global logger

        logger_id = kw.get('logger_id')
        if logger_id:
            logger = logging.getLogger(f'namematch_{str(logger_id)}')

        else:
            logger = self.logger

        # loading cluster_logic functions
        if isinstance(self.constraints_file, str):
            spec = importlib.util.spec_from_file_location("module.name", self.constraints_file)
            cluster_logic = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cluster_logic)
            logger.info(f"Set up cluster constraints {self.constraints_file}")

        elif isinstance(self.constraints_file, ClusterConstraints):
            logger.info(f"Set up cluster constraints {self.constraints_file}")
            cluster_logic = self.constraints_file

        else:
            logger.warning(f"'constraints_file' {self.constraints_file} is not recognized! Use default cluster constraints")
            import namematch.default_cluster_constraints as cluster_logic

        # loading must links
        logger.info("Loading must-link edges.")
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

        # Replace empty string with NaN
        for uid in uid_cols:
            cluster_info[uid] = cluster_info[uid].replace('', np.nan)

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
        logger.info("Loading potential edges.")
        potential_edges_files = [os.path.join(self.potential_edges_dir, pe_file)
                                 for pe_file in os.listdir(self.potential_edges_dir)]
        potential_edges = self.get_potential_edges(
                potential_edges_files, self.flipped0_edges_file, gt_1s_df, cluster_logic, cluster_info)
        logger.info("Clustering potential edges.")

        cluster_assignments = self.cluster_potential_edges(
                clusters, cluster_assignments, original_cluster_ids, cluster_info,
                potential_edges, cluster_logic, self.params, uid_cols, eid_col)

        with open(self.output_file, "wb") as f:
            pickle.dump(cluster_assignments, f)

    @profile
    def auto_is_valid_edge(
                self,
                record1, record2,
                uid_cols, allow_clusters_w_multiple_unique_ids, leven_thresh,
                eid_col=None):
        '''Check if two records would violate a unique id or existing id constraint.

        Args:
            record1 (pd.Series): one row of the all-names file (relevant columns only)
            record2 (pd.Series): one row of the all-names file (relevant columns only)
            uid_cols (list): all-names column(s) with compare_type UniqueID
            allow_clusters_w_multiple_unique_ids (bool): True if a cluster can have multiple uid values
            leven_thresh (int): n character edits to allow between uids before they're considered different
            eid_col (str): all-names column with compare_type ExistingID (None for non-incremental runs)

        Returns:
            bool: False if an automated constraint is violated
        '''

        # NOTE: isinstance(v, float) is a hacky way of checking if null here, since
        #       we force the input of these cols to be strings

        if eid_col is not None:
            # if both existing ids are known, they cannot be different
            if ((not isinstance(record1[eid_col], float)) and \
                (not isinstance(record2[eid_col], float)) and \
                (record1[eid_col] != record2[eid_col])):
                return False

        if allow_clusters_w_multiple_unique_ids:
            return True

        # NOTE if you've made it here, allow_clusters_w_multiple_unique_ids is False

        # if both unique ids are known, they cannot be different
        violations = 0
        attempts = 0
        for uid_col in uid_cols:

            if ((not isinstance(record1[uid_col], float)) and \
                (not isinstance(record2[uid_col], float))):

                attempts += 1

                if (record1[uid_col] != record2[uid_col]):

                    if leven_thresh is None:
                        violations += 1
                        continue
                    else:
                        uid_editdist = editdistance.eval(record1[uid_col], record2[uid_col])
                        if uid_editdist > leven_thresh:
                            violations += 1
                            continue

        if (attempts > 0) and  (violations == attempts):
            return False

        return True


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

    @profile
    @equip_logger_id
    @log_runtime_and_memory
    def get_initial_clusters(self, must_links_df, an_df, eid_col, **kw):
        '''Use must links (ground truth and/or a previous run) to create the
        starting clusters.

        Args:
            must_links_df (pd.DataFrame): record pairs that must be linked together no matter what
                ===================   =======================================================
                record_id_1           unique identifier for the first record in the pair
                record_id_2           unique identifier for the second record in the pair
                blockstring_1         blockstring for the first record in the pair
                blockstring_2         blockstring for the second record in the pair
                drop_from_nm_1        flag, 1 if the first record in the pair was not eligible for matching
                drop_from_nm_2        flag, 1 if the second record in the pair was not eligible for matching
                existing              flag, 1 if the pair is must-link because of ExistingID
                ===================   =======================================================

            an_df (pd.DataFrame): all-names file, with only the columns relevant for clustering
                ========================     =======================================================
                record_id                    unique record identifier
                <uid column(s)>              columns with compare_type UniqueID
                <eid column(s)>              columns with compare_type ExistingID
                <user-constraint column(s)>  (optional) columns mentioned in `get_columns_used()`
                ========================     =======================================================

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
            eid_df = an_df[~an_df[eid_col].isna()].copy()
            eid_df[eid_col] = eid_df[eid_col].astype(int)
            cluster_assignments = eid_df[[eid_col]].to_dict()[eid_col]
            clusters = {}
            for (record_id, cluster_id) in cluster_assignments.items():
                if (cluster_id in clusters):
                    clusters[cluster_id].append(record_id)
                else:
                    clusters[cluster_id] = [record_id]

            original_cluster_ids = set(clusters.keys())

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
        if len(clusters) > 0:
            cluster_id = max(clusters.keys()) + 1
        else:
            cluster_id = 0

        an_df = an_df.reset_index()
        an_df = an_df[an_df.record_id.isin(cluster_assignments) == False].reset_index(drop=True)

        an_df['cluster_id'] = an_df.index + cluster_id
        # NOTE: next 4 lines are for a 1000x speedup over pandas groupby
        keys, values = an_df[['cluster_id', 'record_id']].values.T
        ukeys, index = np.unique(keys, True)
        arrays = np.split(values, index[1:])
        singleton_clusters = dict(zip(ukeys, [list(a) for a in arrays]))
        clusters.update(singleton_clusters)
        cluster_assignments.update(an_df.set_index('record_id').to_dict()['cluster_id'])
        logger.stat(f"n_initial_clusters: {len(clusters)}")
        return clusters, cluster_assignments, original_cluster_ids

    @profile
    @equip_logger_id
    @log_runtime_and_memory
    def get_potential_edges(self, potential_edges_files, flipped0_edges_file, gt_1s_df, cluster_logic, cluster_info, **kw):
        '''
        Use all predictions file to make a list of edges that the constrained
        clustering algorithm should try to add.

        Args:
            potential_edges_files (list): paths to output_temp's potential edges files
            flipped0_edges_file (str): path to output_temp's flipped0-edges file
            gt_1s_df (pd.DataFrame): for incremental runs, subset of the must-link df that are "new" 1s
            cluster_logic (module): user-defined constraint functions
            cluster_info (pd.DataFrame): all-names file, with only the columns relevant for clustering

        Returns:
            list: each element is a dict version of a potential edge's record
        '''

        potential_edges_df = load_parquet_list(potential_edges_files, conditions_dict={'potential_edge': 1})
        potential_edges_df['gt'] = 0

        if (gt_1s_df is not None):
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

        if gt_1s_df is not None:
            gt_1s_df['gt'] = 1
            gt_1s_df['phat'] = 1
            potential_edges_df = pd.concat([potential_edges_df, gt_1s_df[cols]])

        potential_edges_df = potential_edges_df.sort_values(
                ['gt', 'phat'], ascending=[False, False])

        potential_edges_df = potential_edges_df.reset_index(drop=True)

        logger.debug('Applying edge priority.')
        potential_edges_df['original_order'] = potential_edges_df.index
        potential_edges_df = cluster_logic.apply_edge_priority(potential_edges_df, cluster_info)

        logger.stat(f"n_potential_edges: {len(potential_edges_df)}")

        # NOTE: takes a little while, but worth it later to avoid iterrows
        logger.debug('Converting potential_edges_df to dict.')
        potential_edges = potential_edges_df.to_dict('records')
        del potential_edges_df

        return potential_edges

    @profile
    @equip_logger_id
    @log_runtime_and_memory
    def load_cluster_info(self, all_names_file, uid_cols, eid_col, cluster_logic, **kw):
        '''Read in the all_names information needed for cluster constraint checking.

        Args:
            all_names_file (str): path to output_temp's all-names file
            uid_cols (list): all-name columns with compare_type UniqueID
            eid_col (list): all-name columns with compare_type ExistingID
            cluster_logic (module): user-defined constraint functions

        Returns:
            pd.DataFrame: all-names file, with only the columns relevant for clustering
                ========================     =======================================================
                record_id                    unique record identifier
                <uid column(s)>              columns with compare_type UniqueID
                <eid column(s)>              columns with compare_type ExistingID
                <user-constraint column(s)>  (optional) columns mentioned in `get_columns_used()`
                ========================     =======================================================
        '''

        # add things that are missing from get_columns_used

        column_dtypes = cluster_logic.get_columns_used()
        if ("record_id" not in column_dtypes):
            column_dtypes["record_id"] = 'object'
        if ("dataset" not in column_dtypes):
            column_dtypes["dataset"] = 'object'

        cols_to_load_as_obj = uid_cols
        if eid_col is not None:
            cols_to_load_as_obj.append(eid_col)

        for col in cols_to_load_as_obj:
            if col in column_dtypes:
                if column_dtypes[col] != 'object':
                    logger.warning(f"Changing dtype for {col} field to object.")
                    column_dtypes[col] = 'object'
                logger.warning(f"Note, missing values in {col} field are not getting "
                           f"filled with '', since this is an ID field.")
            else:
                column_dtypes[col] = 'object'

        table = pq.read_table(all_names_file)
        cluster_info =  table.to_pandas()
        cluster_info = cluster_info[list(column_dtypes.keys())]
        cluster_info = cluster_info.set_index('record_id')
        del column_dtypes['record_id']
        date_cols = []
        # change the dtypes in all_names based on user-defined dtypes in cluster_constraints
        for col, col_dtype in column_dtypes.items():
            if col_dtype in ['int', 'float']:
                cluster_info[col] = pd.to_numeric(cluster_info[col])
            elif col_dtype in ['date']:
                cluster_info[col] = pd.to_datetime(cluster_info[col])
                date_cols.append(col)
            elif col_dtype in ['str', 'string', 'object']:
                cluster_info[col] = cluster_info[col].fillna("")
                cluster_info[col] = cluster_info[col].astype(str)

        # fillna with '' for columns of dtype object
        obj_cols = [col for col in
                    cluster_info.select_dtypes(include='object').columns.tolist()
                    if col not in (date_cols + cols_to_load_as_obj)]
        for col in obj_cols:
            cluster_info[col] = cluster_info[col].fillna("")

        return cluster_info

    @profile
    @equip_logger_id
    @log_runtime_and_memory
    def cluster_potential_edges(self, clusters, cluster_assignments, original_cluster_ids,
                cluster_info, potential_edges, cluster_logic, params, uid_cols, eid_col, **kw):
        '''For clusters by add potential edges to the cluster graph in order of importance, skipping those
        that cause violations.

        Args:
            clusters (dict): maps a cluster id to a list of record ids -- post initialization
            cluster_assignments (dict): maps a record_id to a cluster_id -- post initialization
            original_cluster_ids (set): set: cluster ids that are already in use (only for incremental)
            cluster_info (pd.DataFrame): all-names file, with only the columns relevant for clustering
                ========================     =======================================================
                record_id                    unique record identifier
                <uid column(s)>              columns with compare_type UniqueID
                <eid column(s)>              columns with compare_type ExistingID
                <user-constraint column(s)>  (optional) columns mentioned in `get_columns_used()`
                ========================     =======================================================

            potential_edges (list): each element is a dict version of a potential edge's record
            cluster_logic (module): user-defined constraint functions
            params (Parameters object): contains parameter values
            uid_cols (list): all-name columns with compare_type UniqueID
            eid_col (list): all-name columns with compare_type ExistingID

        Returns:
            dict: maps record_id to cluster_id
        '''

        # track things
        invalid_edges = 0
        invalid_clusters = 0
        merges = 0
        nrows = len(potential_edges)

        # create ix map so can use faster iloc as opposed to loc
        ci_ix_map = dict(zip(cluster_info.index.tolist(), range(len(cluster_info))))

        logger.info("Checking potential edges:")
        for i, edge in enumerate(potential_edges):

            if (params.verbose is not None) and (i % params.verbose == 0):
                logger.info(f"  Checked {i} of {nrows} edges: {invalid_edges} invalid edges, "
                            f"{invalid_clusters} invalid clusters, {merges} merges.")

            record_id_1 = edge["record_id_1"]
            record_1 = cluster_info.iloc[ci_ix_map[record_id_1]]
            record_id_2 = edge["record_id_2"]
            record_2 = cluster_info.iloc[ci_ix_map[record_id_2]]

            edge_is_gt = (edge["gt"] == 1)

            edge_auto_valid = self.auto_is_valid_edge(
                    record_1, record_2, uid_cols,
                    params.allow_clusters_w_multiple_unique_ids, params.leven_thresh, eid_col)

            if  (edge_auto_valid) and (edge_is_gt or (cluster_logic.is_valid_edge(record_1, record_2, edge['phat']))):

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
                                new_cluster_info.index.isin([record_id_1, record_id_2]).astype(int)
                        # NOTE: new_edge is a way of checking an edge constraint within
                        #       is_valid_cluster (in case you want to enforce an edge
                        #       constraint, but only for clusters that meet some criteria)

                    # if len of new_cluster is 2 then in theory we already vetted by checking the edge
                    # (assuming that all cluster constraints were put into the edge constraint list if possible)

                    if len(new_cluster) > 2:
                        cluster_auto_valid = self.auto_is_valid_cluster(
                            new_cluster_info, uid_cols,
                            params.allow_clusters_w_multiple_unique_ids, params.leven_thresh, eid_col)

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

            else:
                invalid_edges += 1

        logger.info(f"Invalid edges: {invalid_edges}")
        logger.stat(f"n_invalid_edges: {invalid_edges}")
        logger.info(f"Invalid clusters: {invalid_clusters}")
        logger.stat(f"n_invalid_clusters: {invalid_clusters}")
        logger.info(f"Merges: {merges}")
        logger.info(f"n_merges: {merges}")
        logger.info(f"Number of clusters total: {len(clusters)}")
        logger.stat(f"n_clusters: {len(clusters)}")
        logger.info(f"Number of singleton clusters: {len([recs for c_id, recs in clusters.items() if (len(recs) == 1)])}")
        logger.stat(f"n_singleton_clusters: {len([recs for c_id, recs in clusters.items() if (len(recs) == 1)])}")
        # make cluster_id a string
        cluster_assignments = {k: str(v) for k, v in cluster_assignments.items()}

        return cluster_assignments

