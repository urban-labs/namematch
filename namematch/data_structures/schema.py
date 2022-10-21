
import argparse
import errno
import logging
import os
import pickle
import shutil
import sys
import yaml

from datetime import datetime

from namematch.data_structures.parameters import *
from namematch.data_structures.data_file import *
from namematch.data_structures.variable import *
from namematch.utils import utils


class Schema():
    '''Class that houses the most essential instructions for how to complete the match: what
    data files to match, and which variables to use to do so.'''

    def __init__(self, data_files, variables):

        self.data_files = data_files
        self.variables = variables


    @classmethod
    def init(cls, config, params):
        '''Create and validate a DataFileList instance and a VariableList instance.

        Args:
            config (dict): dictionary with match parameter values
            params (dict): dictionary with processed match parameter values

        Returns:
            :mod:`namematch.data_structures.schema.Schema`: instance of the Schema class
        '''

        data_files = DataFileList.build(
                config['data_files'],
                config.get('existing_data_files', {}))
        data_files.validate()

        variables = VariableList.build(config['variables'], params)
        variables.validate(data_files)

        return cls(data_files, variables)


    @classmethod
    def load(cls, filepath):
        '''Load a Schema instance.

        Args:
            filepath (str): path to a yaml version of a Schema instance

        Returns:
            :mod:`namematch.data_structures.schema.Schema`: instance of the Schema class
        '''

        schema_dict = utils.load_yaml(filepath)

        data_files = DataFileList.load(schema_dict['data_files'])
        variables = VariableList.load(schema_dict['variables'])

        return cls(data_files, variables)

    @classmethod
    def load_from_dict(cls, schema_dict):

        data_files = DataFileList.load(schema_dict['data_files'])
        variables = VariableList.load(schema_dict['variables'])

        return cls(data_files, variables)

    def write(self, output_file):
        '''Write the Schema to a yaml file.

        Args:
            output_file (str): path to write schema dictionary
        '''

        utils.dump_yaml(utils.to_dict(self), output_file)


    # def copy(self):
    #     '''Create a deep copy of a Schema object.'''

    #     return Schema(
    #         self.data_files.copy(),
    #         self.variables.copy())
