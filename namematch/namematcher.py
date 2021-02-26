import logging
import yaml
import os

from typing import Union
from datetime import datetime

from namematch.data_structures.schema import Schema
from namematch.data_structures.parameters import Parameters
from namematch.utils.utils import setup_logging, load_logging_params
from namematch.utils import default_parameter
from namematch.process_input_data import ProcessInputData
from namematch.generate_must_links import GenerateMustLinks
from namematch.block import Block
from namematch.generate_data_rows import GenerateDataRows
from namematch.fit_model import FitModel
from namematch.predict import Predict
from namematch.cluster import Cluster, ClusterConstraints
from namematch.generate_output import GenerateOutput

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO
)

class NameMatcher(object):
    """Name Matcher python interface to run all the steps in namematch"""
    def __init__(self,
        config: dict=None,
        private_config: dict=None,
        input_data_batch_size: int=50000,
        data_row_batch_size: int=500,
        log_file: str=None,
        logging_params_file: str=None,
        output_dir: str='output',
        output_temp_dir: str='output_temp',
        all_name_file: str='all_names.parquet',
        must_links: str='must_links.csv',
        og_blocking_index_file: str='None', # We should open an issue for this
        candidate_pairs_file: str='candidate_pairs.parquet',
        data_rows_dir: str='data_rows',
        trained_model_info_file: str='None',
        selection_model_path: str='basic_selection_model.pkl',
        match_model_path: str='basic_match_model.pkl',
        flipped0_file: str='flipped0_potential_edges.csv',
        model_dir: str='model',
        model_info_file: str='model.yaml',
        potential_edges_dir: str='potential_edges',
        cluster_assignments: str='cluster_assignments.pkl',
        cluster_constraints: Union[str, ClusterConstraints]=None,
        an_output_file: str='all_names_with_clusterid.csv',
        write_schema_params: bool=True,
    ):

        self.config = config
        self.private_config = private_config if private_config else default_parameter

        self.write_schema_params = write_schema_params

        # batch size
        self.data_row_batch_size = data_row_batch_size
        self.input_data_batch_size = input_data_batch_size

        # output
        self.output_dir = output_dir
        self.output_temp_dir = output_temp_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if not os.path.exists(self.output_temp_dir):
            os.makedirs(self.output_temp_dir)

        # logging files
        if log_file:
            self.log_file = log_file
        else:
            self.log_file = os.path.join(self.output_temp_dir, 'name_match.log')
            logging.info(f"The log file will be loacted at {self.log_file}.")
        self.logging_params_file = logging_params_file
        self.logging_params = load_logging_params(self.logging_params_file)

        # schema and params
        self.params = None
        self.schem = None

        self.schema_path = os.path.join(self.output_temp_dir, 'schema.yaml')
        if os.path.exists(self.schema_path):
            logging.info(f"{self.schema_path} was found! Schema is loaded.")
            self.schema = Schema.load(self.schema_path)

        self.params_path = os.path.join(self.output_temp_dir, 'parameters.yaml')
        if os.path.exists(self.params_path):
            logging.info(f"{self.params_path} was found! Parameters is loaded")
            self.params = Parameters.load(self.params_path)

        # all the intermediates in output_temp
        self._all_names_file = os.path.join(self.output_temp_dir, all_name_file)
        self._must_links = os.path.join(self.output_temp_dir, must_links)
        self._og_blocking_index_file = os.path.join(self.output_temp_dir, og_blocking_index_file) if og_blocking_index_file != 'None' else og_blocking_index_file
        self._candidate_pairs_file = os.path.join(self.output_temp_dir, candidate_pairs_file)
        self._data_rows_dir = os.path.join(self.output_temp_dir, data_rows_dir)
        self._trained_model_info_file = os.path.join(self.output_temp_dir, trained_model_info_file) if trained_model_info_file != 'None' else trained_model_info_file
        self._selection_model_path = selection_model_path
        self._match_model_path = match_model_path
        self._flipped0_file = flipped0_file
        self._model_dir = os.path.join(self.output_temp_dir, model_dir)
        self._model_info_file = os.path.join(self._model_dir, model_info_file)
        self._potential_edges_dir = os.path.join(self.output_temp_dir, potential_edges_dir)
        self._cluster_assignments = os.path.join(self.output_temp_dir, cluster_assignments)
        self._cluster_constraints = cluster_constraints
        self._an_output_file = os.path.join(self.output_temp_dir, an_output_file)

        # Instantiate each step
        self._process_config(self.write_schema_params)

        self.process_input_data = ProcessInputData(
            self.params,
            self.schema,
            input_data_batch_size=self.input_data_batch_size,
            output_file=self._all_names_file
        )

        self.generate_must_links = GenerateMustLinks(
            self.params,
            self.schema,
            all_names_file=self._all_names_file,
            output_file=self._must_links
        )

        self.block = Block(
            self.params,
            self.schema,
            all_names_file=self._all_names_file,
            must_links_file=self._must_links,
            og_blocking_index_file=self._og_blocking_index_file,
            output_file=self._candidate_pairs_file
        )

        self.generate_data_rows = GenerateDataRows(
            self.params,
            self.schema,
            all_names_file=self._all_names_file,
            candidate_pairs_file=self._candidate_pairs_file,
            batch_size=self.data_row_batch_size,
            output_dir=self._data_rows_dir,
        )

        self.fit_model = FitModel(
            self.params,
            all_names_file=self._all_names_file,
            data_rows_dir=self._data_rows_dir,
            trained_model_info_file=self._trained_model_info_file,
            selection_model_path=self._selection_model_path,
            match_model_path=self._match_model_path,
            flipped0_file=self._flipped0_file,
            output_file=self._model_info_file,
            output_dir=self._model_dir,
        )

        self.predict = Predict(
            self.params,
            data_rows_dir=self._data_rows_dir,
            model_info_file=self._model_info_file,
            output_dir=self._potential_edges_dir,
        )

        self.cluster = Cluster(
            self.params,
            self.schema,
            constraints_file=self._cluster_constraints,
            must_links_file=self._must_links,
            potential_edges_dir=self._potential_edges_dir,
            flipped0_edges_file=self._flipped0_file,
            all_names_file=self._all_names_file,
            output_file=self._cluster_assignments,
        )

        self.generate_output = GenerateOutput(
            self.params,
            self.schema,
            all_names_file=self._all_names_file,
            cluster_assignments_file=self._cluster_assignments,
            an_output_file=self._an_output_file,
            output_dir=self.output_dir,
        )

    def _process_config(self, write_to_file=True):
        """Process config file to create parameter and schema"""
        self.params = Parameters.init(self.config, self.private_config)
        self.schema = Schema.init(self.config, self.params)
        self.params.validate(self.schema.variables)

        if write_to_file:
            self.schema.write(self.schema_path)
            self.params.write(self.params_path)

    def _process_input_data(self):
        """Process input data"""
        if self.process_input_data.output_exists:
            logging.info(f"Output {self.process_input_data.output_file} exists! ")
        else:
            self.process_input_data.logger_init(self.logging_params, self.log_file)
            self.process_input_data.main__process_input_data()
            self.process_input_data.release_log_handlers()

    def _generate_must_links(self):
        """Generate must links"""
        if self.generate_must_links.output_exists:
            logging.info(f"Output {self.generate_must_links.output_file} exists!")
        else:
            self.generate_must_links.logger_init(self.logging_params, self.log_file)
            self.generate_must_links.main__generate_must_links()
            self.generate_must_links.release_log_handlers()

    def _block(self):
        """Block"""
        if self.block.output_exists:
            logging.info(f"Output {self.block.output_file} exists!")
        else:
            self.block.logger_init(self.logging_params, self.log_file)
            self.block.main__block()
            self.block.release_log_handlers()

    def _generate_data_rows(self):
        """Generate data rows"""
        if self.generate_data_rows.output_exists:
            logging.info(f"Output {self.generate_data_rows.output_dir} exists!")
        else:
            self.generate_data_rows.logger_init(self.logging_params, self.log_file)
            self.generate_data_rows.main__generate_data_rows()
            self.generate_data_rows.release_log_handlers()

    def _fit_model(self):
        """Fit model"""
        if self.fit_model.output_exists:
            logging.info(f"Output {self.fit_model.output_dir} exists!")
        else:
            self.fit_model.logger_init(self.logging_params, self.log_file)
            self.fit_model.main__fit_model()
            self.fit_model.release_log_handlers()

    def _predict(self):
        """Predict"""
        if self.predict.output_exists:
            logging.info(f"Output {self.predict.output_dir} exists!")
        else:
            self.predict.logger_init(self.logging_params, self.log_file)
            self.predict.main__predict()
            self.predict.release_log_handlers()

    def _cluster(self):
        """Cluster"""
        if self.cluster.output_exists:
            logging.info(f"Output {self.cluster.output_file} exists!")
        else:
            self.cluster.logger_init(self.logging_params, self.log_file)
            self.cluster.main__cluster()
            self.cluster.release_log_handlers()

    def _generate_output(self):
        """Generate output"""
        if self.generate_output.output_exists:
            logging.info(f"Output {self.generate_output.output_dir} exists!")
        else:
            self.generate_output.logger_init(self.logging_params, self.log_file)
            self.generate_output.main__generate_output()
            self.generate_output.release_log_handlers()

    def run(self):
        self._process_config()
        self._process_input_data()
        self._generate_must_links()
        self._block()
        self._generate_data_rows()
        self._fit_model()
        self._predict()
        self._cluster()
        self._generate_output()

