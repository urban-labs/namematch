import logging
import os
from pathlib import Path
import ruamel.yaml

from typing import Union
from shutil import copyfile

from namematch.data_structures.schema import Schema
from namematch.data_structures.parameters import Parameters
from namematch.utils.utils import to_dict, setup_logging, load_logging_params, reformat_dict
from namematch import default_parameters
from namematch.process_input_data import ProcessInputData
from namematch.generate_must_links import GenerateMustLinks
from namematch.block import Block
from namematch.generate_data_rows import GenerateDataRows
from namematch.fit_model import FitModel
from namematch.predict import Predict
from namematch.cluster import Cluster, Constraints
from namematch.generate_output import GenerateOutput
from namematch.generate_report import GenerateReport

yaml = ruamel.yaml.YAML()
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO
)


class NameMatcher(object):
    """Main interface to run all the steps in namematch"""
    def __init__(
        self,
        # inputs
        config: dict=None,
        default_params: dict=None,
        og_blocking_index_file: str='None',
        trained_model_info_file: str='None',
        # what should outputs be called
        nm_info_file: str='nm_info.yaml',
        log_file_name: str=None,
        logging_params_file: str=None,
        output_dir: str='output',
        output_temp_dir: str=None,
        all_names_file: str='all_names.parquet',
        must_links: str='must_links.csv',
        blocking_index_bin_file: str='blocking_index.bin',
        candidate_pairs_file: str='candidate_pairs.parquet',
        data_rows_dir: str='data_rows',
        selection_model_name: str='basic_selection_model.pkl',
        match_model_name: str='basic_match_model.pkl',
        flipped0_file: str='flipped0_potential_links.csv',
        model_dir: str='model',
        model_info_file: str='model.yaml',
        potential_edges_dir: str='potential_links',
        cluster_assignments: str='cluster_assignments.pkl',
        edges_to_cluster: str='edges_to_cluster.parquet',
        constraints: Union[str, Constraints]=None,
        an_output_file: str='all_names_with_clusterid.csv',
        report_file: str='matching_report.html',
        # profilers
        enable_lprof: bool=False,
        logging_level: str='INFO',
        # for reload
        params=None,
        schema=None,
    ):

        self.config = config
        self.default_params = default_params if default_params else default_parameters

        # output
        self.output_dir = os.path.abspath(output_dir)

        if output_temp_dir:
            self.output_temp_dir = os.path.abspath(output_temp_dir)
        else:
            self.output_temp_dir = os.path.join(self.output_dir, 'details')

        # profilers
        self.enable_lprof = enable_lprof
        self.profile_dir = os.path.join(self.output_temp_dir, 'profilers')

        # logging files
        self.log_file_name = log_file_name if log_file_name else 'name_match.log'
        self.log_file = os.path.join(self.output_temp_dir, self.log_file_name)

        logging.info(f"The log file will be located at {self.log_file}.")
        self.logging_params_file = logging_params_file
        self.logging_params = load_logging_params(self.logging_params_file)

        # nm stats yaml file
        self.nm_info_file = nm_info_file
        self.nm_info_file_path = os.path.join(self.output_temp_dir, self.nm_info_file)

        # schema and params
        self.params = params
        self.schema = schema

        # incremental inputs
        self.og_blocking_index_file = og_blocking_index_file
        self.trained_model_info_file = trained_model_info_file

        # all the intermediates in output/details
        self._all_names_file = all_names_file
        self._must_links = must_links
        self._candidate_pairs_file = candidate_pairs_file
        self._data_rows_dir = data_rows_dir
        self._flipped0_file = flipped0_file
        self._selection_model_name = selection_model_name
        self._match_model_name = match_model_name
        self._model_dir = model_dir
        self._model_info_file = model_info_file
        self._potential_edges_dir = potential_edges_dir
        self._cluster_assignments = cluster_assignments
        self._edges_to_cluster = edges_to_cluster
        self._constraints = constraints
        self._an_output_file = an_output_file
        self._report_file = report_file

        # prepend output/details path to output files
        self.all_names_file = os.path.join(self.output_temp_dir, all_names_file)
        self.must_links = os.path.join(self.output_temp_dir, must_links)
        self.blocking_index_bin_file = os.path.join(self.output_temp_dir, blocking_index_bin_file)
        self.candidate_pairs_file = os.path.join(self.output_temp_dir, candidate_pairs_file)
        self.data_rows_dir = os.path.join(self.output_temp_dir, data_rows_dir)
        self.model_dir = os.path.join(self.output_temp_dir, self._model_dir)
        self.selection_model_name = selection_model_name
        self.match_model_name = match_model_name
        self.flipped0_file = os.path.join(self.output_temp_dir, self._flipped0_file)
        self.model_info_file = os.path.join(self.model_dir, self._model_info_file)
        self.potential_edges_dir = os.path.join(self.output_temp_dir, self._potential_edges_dir)
        self.cluster_assignments = os.path.join(self.output_temp_dir, self._cluster_assignments)
        self.edges_to_cluster = os.path.join(self.output_temp_dir, self._edges_to_cluster)
        self.constraints = constraints
        self.an_output_file = os.path.join(self.output_temp_dir, self._an_output_file)

        # prepend output path
        self.report_file = os.path.join(self.output_dir, self._report_file)

        # Create the folder structure
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if not os.path.exists(self.output_temp_dir):
            os.makedirs(self.output_temp_dir)

        if not os.path.exists(self.data_rows_dir):
            os.makedirs(self.data_rows_dir)

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if not os.path.exists(self.potential_edges_dir):
            os.makedirs(self.potential_edges_dir)

        if self.enable_lprof and (not os.path.exists(self.profile_dir)):
            os.makedirs(self.profile_dir)

        # process config
        if not self.params or not self.schema:
            self._process_config()
        else:
            logging.info("params and schema are provided. don't have to process config")

        # Create the nm_info file
        if not os.path.exists(self.nm_info_file_path):
            nm_metadata = self.nm_metadata
            nm_info = {}
            for key, value in nm_metadata.items():
                nm_metadata[key] = reformat_dict(value)

            nm_info['metadata'] = nm_metadata
            nm_info['stats'] = {}
            with open(self.nm_info_file_path, 'w') as f:
                yaml.dump(nm_info, f)

        # Set up logger
        setup_logging(
            self.logging_params,
            self.log_file,
            self.output_temp_dir,
            filter_stats=False,
            logging_level=logging_level,
        )
        self.logging_params['filters']['stat_filter']['()'] = 'StatLogFilter'


        # Instantiate each step
        self._process_input_data = ProcessInputData(
            params=self.params,
            schema=self.schema,
            all_names_file=self.all_names_file,
            nm_info_file=self.nm_info_file_path,
            nm_metadata=self.nm_metadata,
            profile_dir=self.profile_dir,
            enable_lprof=self.enable_lprof,
        )

        self._generate_must_links = GenerateMustLinks(
            params=self.params,
            schema=self.schema,
            all_names_file=self.all_names_file,
            must_links=self.must_links,
            nm_info_file=self.nm_info_file_path,
            nm_metadata=self.nm_metadata,
            profile_dir=self.profile_dir,
            enable_lprof=self.enable_lprof,
        )

        self._block = Block(
            params=self.params,
            schema=self.schema,
            all_names_file=self.all_names_file,
            must_links_file=self.must_links,
            og_blocking_index_file=self.og_blocking_index_file,
            blocking_index_bin_file=self.blocking_index_bin_file,
            candidate_pairs_file=self.candidate_pairs_file,
            nm_info_file=self.nm_info_file_path,
            nm_metadata=self.nm_metadata,
            profile_dir=self.profile_dir,
            enable_lprof=self.enable_lprof,
        )

        self._generate_data_rows = GenerateDataRows(
            params=self.params,
            schema=self.schema,
            all_names_file=self.all_names_file,
            candidate_pairs_file=self.candidate_pairs_file,
            output_dir=self.data_rows_dir,
            nm_info_file=self.nm_info_file_path,
            nm_metadata=self.nm_metadata,
            profile_dir=self.profile_dir,
            enable_lprof=self.enable_lprof,
        )

        self._fit_model = FitModel(
            params=self.params,
            all_names_file=self.all_names_file,
            data_rows_dir=self.data_rows_dir,
            trained_model_info_file=self.trained_model_info_file,
            selection_model_name=self.selection_model_name,
            match_model_name=self.match_model_name,
            flipped0_file=self.flipped0_file,
            model_info_file=self.model_info_file,
            output_dir=self.model_dir,
            nm_info_file=self.nm_info_file_path,
            nm_metadata=self.nm_metadata,
            profile_dir=self.profile_dir,
            enable_lprof=self.enable_lprof,
        )

        self._predict = Predict(
            params=self.params,
            data_rows_dir=self.data_rows_dir,
            model_info_file=self.model_info_file,
            output_dir=self.potential_edges_dir,
            nm_info_file=self.nm_info_file_path,
            nm_metadata=self.nm_metadata,
            profile_dir=self.profile_dir,
            enable_lprof=self.enable_lprof,
        )

        self._cluster = Cluster(
            params=self.params,
            schema=self.schema,
            constraints=self.constraints,
            must_links_file=self.must_links,
            potential_edges_dir=self.potential_edges_dir,
            flipped0_edges_file=self.flipped0_file,
            all_names_file=self.all_names_file,
            cluster_assignments=self.cluster_assignments,
            edges_to_cluster=self.edges_to_cluster,
            nm_info_file=self.nm_info_file_path,
            nm_metadata=self.nm_metadata,
            profile_dir=self.profile_dir,
            enable_lprof=self.enable_lprof,
        )

        self._generate_output = GenerateOutput(
            params=self.params,
            schema=self.schema,
            all_names_file=self.all_names_file,
            cluster_assignments_file=self.cluster_assignments,
            an_output_file=self.an_output_file,
            output_dir=self.output_dir,
            nm_info_file=self.nm_info_file_path,
            nm_metadata=self.nm_metadata,
            profile_dir=self.profile_dir,
            enable_lprof=self.enable_lprof,
        )

        self._generate_report = GenerateReport(
            params=self.params,
            schema=self.schema,
            report_file=self.report_file,
            nm_info_file=self.nm_info_file_path,
            nm_metadata=self.nm_metadata,
            profile_dir=self.profile_dir,
            enable_lprof=self.enable_lprof,
        )

        self._process_input_data.next = self._generate_must_links
        self._generate_must_links.next = self._block
        self._block.next = self._generate_data_rows
        self._generate_data_rows.next = self._fit_model
        self._fit_model.next = self._predict
        self._predict.next = self._cluster
        self._cluster.next = self._generate_output
        self._generate_output.next = self._generate_report

    @property
    def process_input_data(self):
        self._process_input_data.nm_metadata = self.nm_metadata

        return self._process_input_data

    @property
    def generate_must_links(self):
        self._generate_must_links.nm_metadata = self.nm_metadata
        return self._generate_must_links

    @property
    def block(self):
        self._block.nm_metadata = self.nm_metadata
        return self._block

    @property
    def generate_data_rows(self):
        self._generate_data_rows.nm_metadata = self.nm_metadata
        return self._generate_data_rows

    @property
    def fit_model(self):
        self._fit_model.nm_metadata = self.nm_metadata
        return self._fit_model

    @property
    def predict(self):
        self._predict.nm_metadata = self.nm_metadata
        return self._predict

    @property
    def cluster(self):
        self._cluster.nm_metadata = self.nm_metadata
        return self._cluster

    @property
    def generate_output(self):
        self._generate_output.nm_metadata = self.nm_metadata
        return self._generate_output

    @property
    def generate_report(self):
        self._generate_report.nm_metadata = self.nm_metadata
        return self._generate_report

    @property
    def all_tasks(self):
        tasks = [
            "process_input_data",
            "generate_must_links",
            "block",
            "generate_data_rows",
            "fit_model",
            "predict",
            "cluster",
            "generate_output",
            "generate_report",
        ]
        for task in tasks:
            yield getattr(self, task)

    @property
    def nm_metadata(self):
        '''Namematch state including all the necessary attributes to recreate the NameMatcher object'''
        metadata = {}
        # metadata['metadata'] = {}

        params = to_dict(self.params)
        schema = to_dict(self.schema)

        metadata['params'] = params
        metadata['schema'] = schema
        metadata['filepaths'] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (str, dict, bool, int, float)) or value is None:
                if key not in params or key not in schema:
                    if key not in ["config", "default_params"]:
                        metadata['filepaths'][key] = value

        return metadata

    def _process_config(self):
        """Process config file to create parameter and schema"""
        self.params = Parameters.init(self.config, self.default_params)
        self.schema = Schema.init(self.config, self.params)
        self.params.validate(self.schema.variables)

    def run(self, force=False, write_params_schema_file=True, write_stats_file=True):
        """Main method to kick off the namematch process

        Args:
            force (bool): Force to run all the tasks
            write_params_schema_file (bool): whether to write params and schema to yaml
            write_stats_file (bool): whether to write the nm_info file
        """
        for task in self.all_tasks:
            task.run(force=force, write_stats_file=write_stats_file)

    @classmethod
    def load_namematcher(cls, nm_info_file_path, new_nm_info_file=None, **kwargs):
        '''To load NameMatcher instance given the nm_info_file.
        This classmethod help us pick up where it left last time based on the information in the
        nm_info_file. It will create a NameMatcher instance and recover all the attributes as
        well as stats_dict for tasks that was already run.

        Args:
            nm_info_file (str): nm_info.yaml file path

        Returns:
            :mod:`namematch.namematcher.NameMatcher`: NameMatcher instance

        '''
        with open(nm_info_file_path) as f:
            nm_info = yaml.load(f)
            stats = nm_info['stats']
            metadata = nm_info['metadata']
            filepaths = metadata['filepaths']

        params = Parameters.load_from_dict(metadata['params'])
        schema = Schema.load_from_dict(metadata['schema'])

        if new_nm_info_file:
            copyfile(nm_info_file_path, os.path.join(os.path.dirname(nm_info_file_path), new_nm_info_file))

        for key, value in kwargs.items():
            if key in filepaths:
                filepaths[key] = value

        nm = cls(
            nm_info_file=new_nm_info_file if new_nm_info_file else Path(nm_info_file_path).name,
            log_file_name=filepaths['log_file_name'],
            logging_params_file=None,
            output_dir=filepaths['output_dir'],
            output_temp_dir=filepaths['output_temp_dir'],
            all_names_file=filepaths['_all_names_file'],
            must_links=filepaths['_must_links'],
            og_blocking_index_file=filepaths['og_blocking_index_file'],
            candidate_pairs_file=filepaths['_candidate_pairs_file'],
            data_rows_dir=filepaths['_data_rows_dir'],
            trained_model_info_file=filepaths['trained_model_info_file'],
            selection_model_name=filepaths['selection_model_name'],
            match_model_name=filepaths['match_model_name'],
            flipped0_file=filepaths['_flipped0_file'],
            model_dir=filepaths['_model_dir'],
            model_info_file=filepaths['_model_info_file'],
            potential_edges_dir=filepaths['_potential_edges_dir'],
            cluster_assignments=filepaths['_cluster_assignments'],
            constraints=filepaths['_constraints'],
            an_output_file=filepaths['_an_output_file'],
            report_file=filepaths['_report_file'],
            params=params,
            schema=schema,
            logging_level='DEBUG',
        )
        for task in nm.all_tasks:
            task_name = task.__class__.__name__
            if task_name in stats:
                task.stats_dict = stats[task_name]

        return nm
