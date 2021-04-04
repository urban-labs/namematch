import logging
import os

from abc import ABC, abstractmethod

from namematch.utils.utils import setup_logging
from namematch.data_structures.parameters import Parameters
from namematch.data_structures.schema import Schema

class NamematchBase(ABC):
    """The base class for namematch steps

    All the inherited classes should have other input/output file paths for each step.

    Args:
        params (object): namematch's Parameter object
        schema (object): namematch's Schema object
        logger_id (str): logging for using specific logger
        output_file (str): output file path
        logger(object): logging.logger object

    """
    def __init__(
        self,
        params: Parameters,
        schema: Schema,
        output_file: str=None,
        logger_id: str=None,
    ):

        self.params = params
        self.schema = schema
        self.logger_id = logger_id
        self.output_file = output_file
        self.logger = None

    def logger_init(self, logging_params, log_file, output_temp_dir):
        setup_logging(logging_params, log_file, output_temp_dir)
        logging_params['filters']['stat_filter']['()'] = 'StatLogFilter'
        self.logger = logging.getLogger()

    @property
    def output_exists(self):
        try:
            return os.path.exists(self.output_file)

        except TypeError:
            if os.path.exists(self.output_dir) and len(os.listdir(self.output_dir)) == self.params.num_threads:
                return True
            else:
                return False

    def remove_output(self):
        try:
            if os.path.exists(self.output_file):
                os.remove(self.output_file)
        except:
            for output_file in self.output_file:
                os.remove(output_file)

    def release_log_handlers(self):
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)
