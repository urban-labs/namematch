
import logging
from copy import deepcopy

from namematch.utils.utils import *

logger = logging.getLogger()


class DataFile():
    '''Parent class for NewDataFile and ExistingDateFile, which house details about the
    data files input for matching.'''

    def __init__(self, validated_data_file_dict):

        for attribute, attribute_value in validated_data_file_dict.items():
            setattr(self, attribute, attribute_value)

    @classmethod
    def load(cls, data_file_dict):
        '''Load a DataFile instance (either a NewDataFile or an ExistingDataFile).

        Args:
            data_file_dict (dict): dictionary-version of a DataFile object

        Returns:
            instance of the DataFile class
        '''

        return cls(data_file_dict)

    def validate_existance(self):
        '''Validate that an input file path exists.'''

        if not os.path.exists(self.filepath):
            logger.error(f"The {self.nickname} file does not exists: "
                         f"{self.filepath}")
            raise

    def validate_record_id_col(self):
        '''Validate that the record_id column exists and meets uniqueness criteria.'''

        df = pd.read_csv(self.filepath, nrows=1)
        if self.record_id_col not in df.columns.tolist():
            logger.error(f"The record_id_col specified for the {self.nickname} "
                         f"file does not exist: {self.record_id_col}")
            raise

        df = pd.read_csv(self.filepath, usecols=[self.record_id_col])
        if df[self.record_id_col].isnull().sum() > 0:
            logger.error(f"The record_id_col specified for the {self.nickname} "
                         f"file contains nulls: {self.record_id_col}.")
            raise
        if len(df) != df[self.record_id_col].nunique():
            logger.warning(f"The record_id_col specified for the {self.nickname} "
                           f"file is not unique: {self.record_id_col}.")
        if (self.file_type == 'existing') and self.use_record_id_as_is:
            split_recids = df[self.record_id_col].str.split("__", n=1, expand=True)
            if (split_recids.shape[1] == 1) or split_recids[1].isnull().any():
                logger.error(f"The use_record_id_as_is parameter is set to True for the {self.nickname} file, "
                             f"but at least one of the underlying record_id values contains no prefix "
                             f"(e.g. 'arrests__'). Consider changing use_record_id_as_is to False.")
                raise
            prefixes = split_recids[0].unique().tolist()
            logger.info(f"The use_record_id_as_is parameter is set to True for the {self.nickname} file. "
                        f"The ids are prefixed by the following nicknames: {', '.join(prefixes)}.")


    def copy(self):
        '''Create a deep copy of a DataFile object.'''
        return deepcopy(self)


class NewDataFile(DataFile):

    @classmethod
    def build(cls, nickname, info):
        '''Create a NewDataFile instance.

        Args:
            nickname (str): the data file's nickname
            info (dict): info about a data file definition from user-input config

        Returns:
           :mod:`namematch.data_structures.data_file.NewDataFile`: instance of the NewDataFile class
        '''

        params_required = ['filepath', 'record_id_col']
        for param_required in params_required:
            if param_required not in info:
                logger.error(f"Error in data_file definition. Please ensure "
                             f"{param_required} is defined.")
                raise ValueError

        data_file_dict = {}
        data_file_dict['nickname'] = nickname
        data_file_dict['filepath'] = info['filepath']
        data_file_dict['record_id_col'] = info['record_id_col']
        data_file_dict['delim'] = info.get('delim', ',')
        data_file_dict['file_type'] = 'new'
        data_file_dict['use_record_id_as_is'] = False
        data_file_dict['cluster_type'] = info.get('cluster_type', 'cluster')
        data_file_dict['output_file_stem'] = info.get('output_file_stem', nickname)

        return cls(data_file_dict)


class ExistingDataFile(DataFile):

    @classmethod
    def build(cls, nickname, info):
        '''Create a ExistingDataFile instance.

        Args:
            nickname (str): the data file's nickname
            info (dict): info about a data file definition from user-input config

        Returns:
            :mod:`namematch.data_structures.data_file.ExistingDataFile`: instance of the ExistingDataFile class
        '''

        params_required = ['filepath', 'record_id_col']
        for param_required in params_required:
            if param_required not in info:
                logger.error('Error in existing_data_file definition. Please ensure %s is defined.' % param_required)
                raise ValueError

        data_file_dict = {}
        data_file_dict['nickname'] = nickname
        data_file_dict['filepath'] = info['filepath']
        data_file_dict['record_id_col'] = info['record_id_col']
        data_file_dict['delim'] = info.get('delim', ',')
        data_file_dict['file_type'] = 'existing'
        data_file_dict['use_record_id_as_is'] = info.get('use_record_id_as_is', False)

        return cls(data_file_dict)


class DataFileList():
    '''Class that houses a list of DataFile objects (either NewDataFiles or ExistingDateFiles.'''

    def __init__(self, data_files_dict):

        self.data_files = data_files_dict['data_files']
        self.existing_data_files = data_files_dict['existing_data_files']
        self.validate_names()


    @classmethod
    def build(cls, data_files_dict, existing_data_files_dict):
        '''Create a DataFileList instance.

        Args:
            data_files_dict (dict): dictionary with "new data file" info from user-input config
            existing_data_files_dict (dict): dictionary with "existing data file" info from user-input config

        Returns:
            :mod:`namematch.data_structures.data_file.DataFileList`: instance of the DataFileList class
        '''

        validated_data_files_dict = {
            'data_files' : [],
            'existing_data_files' : []
        }
        for data_file_name, data_file_info in data_files_dict.items():
            validated_data_file = NewDataFile.build(data_file_name, data_file_info)
            validated_data_files_dict['data_files'].append(validated_data_file)
        for data_file_name, data_file_info in existing_data_files_dict.items():
            validated_data_file = ExistingDataFile.build(data_file_name, data_file_info)
            validated_data_files_dict['existing_data_files'].append(validated_data_file)

        return cls(validated_data_files_dict)


    @classmethod
    def load(cls, data_files_list_dict):
        '''Load a DataFileList instance.

        Args:
            data_files_list_dict (dict): dictionary-version of a DataFileList object

        Returns:
            :mod:`namematch.data_structures.data_file.DataFileList`: instance of the DataFileList class
        '''

        data_files = [
                DataFile.load(data_file_dict)
                for data_file_dict in data_files_list_dict['data_files']]
        existing_data_files = [
                DataFile.load(data_file_dict)
                for data_file_dict in data_files_list_dict['existing_data_files']]

        data_files_dict = {
            'data_files' : data_files,
            'existing_data_files' : existing_data_files
        }

        # NOTE: no need for validation because simple loading
        # an object that was validated when initialized

        return cls(data_files_dict)


    def get_all_nicknames(self):
        '''Return a list of all of the DataFile nicknames in the DataFileList.

        Return:
            list of strings
        '''

        data_file_name_list = [df.nickname for df in self.data_files]
        existing_data_file_name_list = [df.nickname for df in self.existing_data_files]
        if len(existing_data_file_name_list) > 0:
            data_file_name_list.extend(existing_data_file_name_list)

        return data_file_name_list


    def validate(self):
        '''Validate the DataFileList by validating the list overall and then validating
        each individual DataFile.'''

        self.validate_names()
        for data_file in (self.data_files + self.existing_data_files):
            data_file.validate_existance()
            data_file.validate_record_id_col()


    def validate_names(self):
        '''Validate that the DataFiles in the DataFileList all have unique nicknames
        and that the output file stems have unique cluster types.'''

        # check for unique nicknames
        data_file_name_list = self.get_all_nicknames()
        if len(data_file_name_list) != len(set(data_file_name_list)):
            logger.error('Input data files must have unique nicknames.')
            raise ValueError

        # check for unique cluster_types per output_file_stem
        cluster_types = [df.cluster_type for df in self.data_files]
        output_stems = [df.output_file_stem for df in self.data_files]
        name_df = pd.DataFrame(data={
                'cluster_types':cluster_types,
                'output_stems':output_stems})
        cluster_types_per_file = name_df.groupby('output_stems').cluster_types.nunique()
        too_many = cluster_types_per_file[cluster_types_per_file > 1].index.tolist()
        if len(too_many) > 0:
            logger.error(f"Within an output_file_stem, the cluster_types must be "
                         f"unique. Revisit the config specification for the "
                         f"following output_file_stem(s): {data_file.output_file_stem}")
            raise ValueError()


    def write(self, output_file):
        '''Write the DataFileList to a yaml file.

        Args:
            output_file (str): path to write data file list dictionary
        '''

        dump_yaml(to_dict(self), output_file)


    def copy(self):
        '''Create a deep copy of a DataFileList object.'''

        return deepcopy(self)


    def get_all_data_files(self):
        '''Retrieve list of all DataFile objects, regardless of New or Existing.

        Return:
            list of DataFile objects
        '''

        all_data_files = []
        for file_type, data_file_list in self.__dict__.items():
            for data_file in data_file_list:
                all_data_files.append(data_file.copy())

        return all_data_files

