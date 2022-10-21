import csv
import logging
import numpy as np
import os
import pandas as pd
import pickle

import pyarrow as pa
import pyarrow.parquet as pq

from datetime import datetime

from namematch.base import NamematchBase
from namematch.data_structures.parameters import Parameters
from namematch.data_structures.schema import Schema
from namematch.utils.utils import *

logger = logging.getLogger(__name__)

class GenerateOutput(NamematchBase):
    '''
    Args:
        params (Parameters object): contains parameter values
        schema (Schema object): contains match schema info (files to match, variables to use, etc.)
        all_names_file (str): path to the all-names file
        cluster_assignments_file (str): path to the cluster-assignments file
        an_output_file (str): path to the all-names-with-clusterid file
        output_dir (str): path to final output directory
    '''
    def __init__(
        self,
        params,
        schema,
        all_names_file,
        cluster_assignments_file,
        an_output_file,
        output_dir,
        output_file_uuid=None,
        *args,
        **kwargs
    ):
        super(GenerateOutput, self).__init__(params, schema, *args, **kwargs)

        self.all_names_file = all_names_file
        self.cluster_assignments_file = cluster_assignments_file
        self.an_output_file = an_output_file
        self.output_dir = output_dir
        self.output_file_uuid = output_file_uuid

    @property
    def output_files(self):
        output_files = [self.an_output_file]
        for data_file in self.schema.data_files.data_files:
            if self.output_file_uuid:
                self.output_file_name = data_file.output_file_stem + f"_with_clusterid_{self.output_file_uuid}.csv"
            else:
                output_file_name = data_file.output_file_stem + "_with_clusterid.csv"

            output_files.append(os.path.join(self.output_dir, output_file_name))

        return output_files

    def main(self, **kw):
        '''Read in the cluster assignments dictionary and use it to create all-names-with-cluster-id
        and the "with-cluster-id" versions of input dataset.
        '''
        with open(self.cluster_assignments_file, "rb") as f:
            cluster_assignments = pickle.load(f)

        an_df = self.create_allnames_clusterid_file(
                self.all_names_file, cluster_assignments,
                self.params.blocking_scheme['cosine_distance']['variables'])
        an_df.to_csv(self.an_output_file, index=False)

        self.output_clusterid_files(
            self.schema.data_files.data_files,
            cluster_assignments,
            self.output_dir,
            self.output_file_uuid
        )

        self.stats_dict['end'] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

    @log_runtime_and_memory
    def create_allnames_clusterid_file(self, all_names_file, cluster_assignments, cleaned_col_names, **kw):
        '''Create all-names-with-clusterid dataframe.

        Args:
            all_names_file (str): path to the all-names file
            cluster_assignments (dict): maps record_id to cluster_id
            cleaned_col_names (list): all-name columns used in cosine blocking

        Returns:
            pd.DataFrame: all-names-with-cluster-id

            =====================   =======================================================
            record_id               unique record identifier
            file_type               either "new" or "existing"
            <fields for matching>   both for the matching model and for constraint checking
            blockstring             concatenated version of blocking columns (sep by ::)
            drop_from_nm            flag, 1 if met any "to drop" criteria 0 otherwise
            cluster_id              unique person identifier, no missing values
            =====================   =======================================================
        '''

        logger.info('Generating all_names file with cluster_id.')

        table = pq.read_table(all_names_file)
        all_names = table.to_pandas()
        all_names["cluster_id"] = all_names["record_id"].map(cluster_assignments)
        for cleaned_col in cleaned_col_names:
            all_names[cleaned_col] = all_names[f'tmp_raw__{cleaned_col}']
        all_names = all_names.drop(columns=[f'tmp_raw__{col}' for col in cleaned_col_names])

        return all_names

    @log_runtime_and_memory
    def output_clusterid_files(self, data_files, cluster_assignments, output_dir, output_file_uuid=None, **kw):
        '''For each input file, construct a matching output file that has the
        cluster_id column, and write it.

        Args:
            data_files (list of DataFile objects): contains info about each input file
            cluster_assignments (dict): maps record_id to cluster_id
            output_dir (str): the path that was supplied when the name match object was created
        '''
        if output_file_uuid is None:
            output_file_uuid = self.output_file_uuid

        # remove output files if they exist (necessary for
        # input_files that have the same output_file_stem)
        for data_file in data_files:
            if output_file_uuid:
                output_file_name = data_file.output_file_stem + f"_with_clusterid_{output_file_uuid}.csv"
            else:
                output_file_name = data_file.output_file_stem + "_with_clusterid.csv"
            output_file_name = os.path.join(output_dir, output_file_name)
            if os.path.exists(output_file_name):
                os.remove(output_file_name)

        # update each of the input files
        for data_file in data_files:

            if output_file_uuid:
                output_file_name = data_file.output_file_stem + f"_with_clusterid_{output_file_uuid}.csv"
            else:
                output_file_name = data_file.output_file_stem + "_with_clusterid.csv"

            output_file_path = os.path.join(output_dir, output_file_name)
            if not os.path.exists(output_file_path):
                # read input file
                df = pd.read_csv(data_file.filepath, dtype=object, encoding="ISO-8859-1")
            else:
                # read existing output file
                df = pd.read_csv(output_file_path, dtype=object, encoding="ISO-8859-1")

            logger.info(f"Writing {output_file_name} file.")

            df['temp_nm_rec_id'] = create_nm_record_id(
                    data_file.nickname, df[data_file.record_id_col])

            cluster_id_col = f"{data_file.cluster_type}_id"
            df[cluster_id_col] = df['temp_nm_rec_id'].map(cluster_assignments)

            n_missing = df[cluster_id_col].isnull().sum()
            if  n_missing > 0:
                logger.warning(f"{n_missing} rows without cluster_ids (this shouldn't happen)")
                df = df[df[cluster_id_col].isnull() == False]

            df.drop(columns=['temp_nm_rec_id']).to_csv(
                    output_file_path, index=False, quoting=csv.QUOTE_NONNUMERIC)

