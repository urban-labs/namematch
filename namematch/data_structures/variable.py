
import logging
import re
from copy import deepcopy

from namematch.utils.utils import *

logger = logging.getLogger()


class Variable():

    def __init__(self, validated_variable_dict):

        for attribute, attribute_value in validated_variable_dict.items():
            setattr(self, attribute, attribute_value)


    @classmethod
    def build(cls, variable_dict, params):
        '''Create a Variable instance.

        Args:
            variable_dict (dict): info about a variable definition from user-input config
            params (dict): dictionary with processed match parameter values

        Returns:
            :mod:`namematch.data_structures.variable.Variable`: instance of the Variable class
        '''

        validated_variable_dict = dict()
        validated_variable_dict['check'] = ''
        validated_variable_dict['drop_list'] = []
        validated_variable_dict['colname_dict'] = {}
        validated_variable_dict['set_missing_list'] = []

        if 'name' not in variable_dict or 'compare_type' not in variable_dict:
            logger.error(f"Error in variable definition. Please ensure "
                         f"name and compare_type are specified.")
            raise

        validated_variable_dict['name'] = variable_dict['name']
        validated_variable_dict['compare_type'] = variable_dict['compare_type']
        is_last_name = (variable_dict["name"] == params.last_name_column)
        if is_last_name and params.split_names:
            validated_variable_dict["compare_type"] = "LastName"

        col_var_list = [k for k in variable_dict.keys() if '_col' in k]
        for col_var in col_var_list:
            file_nickname = re.sub('_col$', '', col_var)
            validated_variable_dict['colname_dict'][file_nickname] = variable_dict[col_var]
            # NOTE: will validate in NameMatch.build next step

        # issue warning if user specifies param not understood by code
        params_user_specified = [v for v in list(variable_dict.keys()) if '_col' not in v]
        params_acceptable = ['name', 'compare_type', 'check', 'drop', 'set_missing']
        params_unrecognized = [
                param for param in params_user_specified
                if param not in params_acceptable]
        if len(params_unrecognized) > 0:
            params_unrecognized = ', '.join(params_unrecognized)
            logger.warning(f"The following parameters of {validated_variable_dict['name']} "
                           f"are not recognized (will be ignored): {params_unrecognized}.")

        if 'check' in variable_dict:
            validated_variable_dict['check'] = variable_dict['check']

        if 'drop' in variable_dict:
            if not isinstance(variable_dict['drop'], list):
                logger.error(f"The drop parameters of {validated_variable_dict['name']}"
                             f"must be a list. It's currently a {type(variable_dict['drop'])}.")
                raise Exception
            validated_variable_dict['drop_list'] = variable_dict['drop']

        if 'set_missing' in variable_dict:
            if not isinstance(variable_dict['set_missing'], list):
                logger.error(f"The set_missing parameters of {validated_variable_dict['name']}"
                             f"must be a list. It's currently a {type(variable_dict['set_missing'])}.")
                raise Exception
            validated_variable_dict['set_missing_list'] = variable_dict['set_missing']

        return cls(validated_variable_dict)


    def validate_col_parameters(self, data_files):
        '''Validate that each data file has a corresponding "_col" parameter in each
        variable defintion.

        Args:
            data_files (:mod:`namematch.data_structures.data_file.DataFileList`): info about what input files are being matched
        '''

        col_var_name_list = list(self.colname_dict.keys())

        data_file_name_list = data_files.get_all_nicknames()

        missing_col_var_list = []
        for data_file_name in data_file_name_list:
            if data_file_name not in col_var_name_list:
                missing_col_var_list.append(data_file_name)

        if len(missing_col_var_list) > 0:
            missing_col_var_list = ', '.join(missing_col_var_list)
            logger.error(f'The following data files do not have a column definition '
                         f'in the {self.name} variable: {missing_col_var_list}')
            raise ValueError

        if len(col_var_name_list) > len(data_file_name_list):
            logger.warning(f'There are more column definitions in the {self.name} variable ',
                            f'than data files. Column definitions that do not correspond to '
                            f'a data file will be ignored.')


    def get_columns_to_read(self, file_nickname):
        '''Get the name(s) of the column(s) from the input file that need to be read in order to
        create the current all-names column.

        Args:
            file_nickname (str): nickname of input file being searched

        Return:
            list of column names
        '''

        column_name_list = self.colname_dict[file_nickname].split(',')

        all_column_names = []
        for column_name in column_name_list:
            if ',' in column_name:  # geography
                all_column_names.extend(column_name.split(','))
            else:
                if column_name != '':
                    all_column_names.append(column_name)

        return all_column_names


    def get_an_columns(self):
        '''Get the name(s) of the current all-names column(s).

        Return:
            list of column names
        '''

        if self.compare_type == 'Address':
            return ['address_street_number', 'address_street_name', 'address_street_type']
        else:
            return self.name


    def copy(self):
        '''Create a deep copy of a Variable object.'''
        return deepcopy(self)



class VariableList():
    '''Class that houses a list of Variable objects.'''

    def __init__(self, variable_list):

        self.varlist = variable_list

    @classmethod
    def build(cls, variable_dict_list, params):
        '''Create a VariableList instance.

        Args:
            variable_dict_list (dict): dictionary with variable info from user-input config
            params (dict): dictionary with processed match parameter values

        Returns:
            :mod:`namematch.data_structures.variable.VariableList`: instance of the VariableList class
        '''

        validated_variable_list = []
        for variable_dict in variable_dict_list:
            validated_variable = Variable.build(variable_dict, params)
            validated_variable_list.append(validated_variable)

        return cls(validated_variable_list)

    @classmethod
    def load(cls, variables_list_dict):
        '''Load a VariableList instance.

        Args:
            variables_list_dict (dict): dictionary version of a VariableList instance

        Returns:
            :mod:`namematch.data_structures.variable.VariableList`: instance of the VariableList class
        '''

        variable_list = [
                Variable(variable_dict)
                for variable_dict in variables_list_dict['varlist']]

        # NOTE: no need for validation because simple loading
        # an object that was validated when initialized

        return cls(variable_list)

    def validate_col_parameters(self, data_files):
        '''Validate that the "_col" variables referenced in the config's variable definitions
        actually exist in the input datasets.

        Args:
            data_files (:mod:`namematch.data_structures.data_file.DataFileList`): info about what input files are being matched
        '''

        for variable in self.varlist:
            variable.validate_col_parameters(data_files)

        # make sure the variables actually exist
        for data_file in (data_files.data_files + data_files.existing_data_files):
            df = pd.read_csv(data_file.filepath, nrows=1)
            cols_to_read = self.get_columns_to_read(data_file)
            for col in cols_to_read:
                if col not in df.columns.tolist():
                    logger.error(f"The {col} column does not exist in the "
                                 f"{data_file.nickname} file.")
                    raise

    def validate_variable_names(self):
        '''Validate that the Variables in the VariableList all have unique nicknames.'''

        # make sure names unique
        variable_name_list = self.get_names()
        if len(variable_name_list) != len(set(variable_name_list)):
            logger.error('Variables must have unique names.')
            raise ValueError

        # if variable with compare_type Adress exists, check to see if variables
        # address_street_number, address_street_name, or address_street_type already exist
        if any([variable.compare_type == 'Address' for variable in self.varlist]):
            addr_vars = ['address_street_number', 'address_street_name', 'address_street_type']
            variable_names_to_check = addr_vars
            variable_names_invalid = []
            for variable_name in variable_names_to_check:
                if variable_name in variable_name_list:
                    variable_names_invalid.append(variable_name)
            if len(variable_names_invalid) > 0:
                variable_names_invalid = ', '.join(variable_names_invalid)
                logger.error(f'Given a variable of compare_type Address, the following '
                             f'variable names are invalid: {variable_names_invalid}')
                raise ValueError

    def validate_type_counts(self, incremental):
        '''Validate that there is exactly one variable with compare type UniqueID. If
        incremental, validate that there is exactly one variable with compare type ExistingID.

        Args:
            incremental (bool): True if the config file provides "existing" data files
        '''

        # count number of unique ids and issue warning if greater than 1
        n_uids = len(self.get_variables_where(attr='compare_type', attr_value='UniqueID'))
        if n_uids > 1:
            logger.warning(f'You have more than one variable with compare_type '
                           f'UniqueID. This is only recommended in rare cases -- '
                           f'please review the documentation.')

        # if count number of existing ids and raise error if greater than 1
        if incremental:
            n_eids = len(self.get_variables_where(attr='compare_type', attr_value='ExistingID'))
            if n_eids == 0:
                logger.error(f'You must specify a variable with compare type ExistingID for incremental '
                             f'matching, -- please review the documentation.')
                raise ValueError
            if n_eids > 1:
                logger.error(f'You have more than one variable with compare_type ExistingID, '
                             f'which is prohibited -- please review the documentation.')
                raise ValueError

    def validate(self, data_files):
        '''Validate several components of the variables defined in the config.

        Args:
            data_files (:mod:`namematch.data_structures.data_file.DataFileList`): info about what input files are being matched
        '''

        self.validate_col_parameters(data_files)
        self.validate_variable_names()
        self.validate_type_counts(incremental=(len(data_files.existing_data_files) > 0))

    def get_variables_where(self, attr, attr_value, equality_type='equals', return_type='name'):
        '''Select variables that meet a certain condition (e.g. compare_type == 'Category').

        Args:
            attr (str): variable feature to condition on
            attr_value (str): acceptable values for the variable feature
            equality_type (str): check conditions using either "equals" or "in"
            return_type (str): either "name" or "ix", determining what return type to use

        Return:
            list of variable nicknames (all-names columns) or corresponding all-names column indices
        '''

        ix_list = []
        name_list = []
        for ix, variable in enumerate(self.varlist):
            if equality_type == 'equals':
                if variable.__dict__[attr] == attr_value:
                    ix_list.append(ix)
                    name_list.append(variable.name)
            elif equality_type == 'in':
                if variable.__dict__[attr] in attr_value:
                    ix_list.append(ix)
                    name_list.append(variable.name)
            else:
                logger.error(f'Invalid equality type ({equality_type}) passed to '
                             f'get_variables_where. Only values "equals" or "in" are valid.')
                raise ValueError

        if return_type == 'ix':
            return ix_list
        elif return_type == 'name':
            return name_list
        else:
            logger.error('Unknown value passed as return_type in get_variables_where.')
            raise ValueError

    def get_names(self):
        '''Get list of variable nicknames.

        Returns:
            list of variable nicknames
        '''

        variable_names = [variable.name for variable in self.varlist]

        return variable_names

    def get_columns_to_read(self, data_file):
        '''Get the name(s) of the column(s) from the input file that need to be read in order to
        create the all-names file.

        Args:
            data_file (DataFile object): contains info about a given input file

        Return:
            list of column names
        '''

        all_column_names = [data_file.record_id_col]
        for variable in self.varlist:
            column_names = variable.get_columns_to_read(data_file.nickname)
            all_column_names.extend([column_names] if type(column_names) == str else column_names)

        return list(set(all_column_names))

    def get_an_column_names(self):
        '''Get the final list of all-names columns, including internally created columns
        like `file_type` and `drop_from_nm`.

        Return:
            list of all-names columns
        '''

        variable_list = self.copy().varlist

        column_names = ["record_id", "file_type", "drop_from_nm", "dataset"]
        for variable in variable_list:
            if variable.compare_type == 'Address':
                column_names.extend(
                        ['address_street_number',
                        'address_street_name',
                        'address_street_type'])
            else:
                column_names.append(variable.name)

        return column_names

    def write(self, output_file):
        '''Write the VariableList to a yaml file.

        Args:
            output_file (str): path to write variable list dictionary
        '''

        dump_yaml(to_dict(self), output_file)

    def copy(self):
        '''Create a deep copy of a VariableList object.'''

        return deepcopy(self)
