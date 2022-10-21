import yaml
import logging
from copy import deepcopy

from namematch.utils.utils import dump_yaml, to_dict, load_yaml

logger = logging.getLogger()

params_lookup = {
    'process_input_data': [
        'auto_drop_logic',
        'blocking_scheme',
        'last_name_column',
        'verbose',
        'incremental',
        'input_data_batch_size'
        ],
    'generate_must_links': [],
    'block': [
        'split_names',
        'last_name_column',
        'blocking_scheme',
        'blocking_thresholds',
        'incremental',
        'index',
        'num_workers',
        'nmslib',
        'verbose',
        'parallelize'
        ],
    'generate_data_rows': [
        'verbose',
        'data_rows_batch_size',
        'num_workers',
        'match_train_criteria',
        'first_name_column',
        'last_name_column',
        'parallelize',
        'incremental',
        'exact_match_variables',
        'leven_thresh',
        ],
    'fit_model': [
        'max_selection_train_eval_n',
        'missingness_model',
        'weight_using_selection_model',
        'default_threshold',
        'optimize_threshold',
        'fscore_beta',
        'match_train_criteria',
        'pct_train',
        'max_match_train_n',
        'allow_clusters_w_multiple_unique_ids',
        ],
    'predict': [
        'use_uncovered_phats',
        'parallelize',
        ],
    'cluster': [
        'initialize_from_ground_truth_1s',
        'incremental',
        'verbose',
        'allow_clusters_w_multiple_unique_ids',
        'leven_thresh',
        ],
    'generate_output': [
        'blocking_scheme',
        ],
}

class Parameters():
    '''Class that houses important matching parameters. Handles validation of the config file.'''

    def __init__(self, validate_param_dict):

        for param, param_value in validate_param_dict.items():
            setattr(self, param, param_value)

    @staticmethod
    def check_integrity(defaults, param, param_value):
        '''Ensure that parameters are of the appropriate type.

        Args:
            param (str): parameter name (key)
            param_value: value of the given key  (parameter name)
        '''

        params__start_with_alpha = []
        params__numeric = []
        params__positive_numeric = [
                'fscore_beta', 'leven_thresh', 'num_workers', 'pct_train',
                'secondary_index_limit', 'verbose', 'input_data_batch_size', 'data_rows_batch_size']
        params__boolean = [
                'use_uncovered_phats', 'allow_clusters_w_multiple_unique_ids']
        params__specific_value = {
                'rebuild_main_index' : ['always', 'never', 'if_secondary_index_exceeds_limit']}

        valid = True

        same_as_default = (param_value == defaults[param])
        if defaults[param] is None and param_value is None:
            same_as_default = True

        if not same_as_default:
            if param in params__start_with_alpha:
                if not str.isalpha(param_value[0]):
                    valid = False
            if param in params__numeric:
                if not isinstance(param_value, (int, float, complex)):
                    valid = False
            if param in params__positive_numeric:
                if not isinstance(param_value, (int, float, complex)) or param_value <= 0:
                    valid = False
            if param in params__boolean:
                if not isinstance(param_value, bool):
                    valid = False
            if param in params__specific_value:
                if param_value not in params__specific_value[param]:
                    valid = False

        if not valid:
            logger.warning(f"The {param} parameter has taken an illegal value "
                           f"({param_value}). Default value ({defaults[param]}) "
                           f"will be used.")

        return valid

    @classmethod
    def init(cls, config: dict, defaults: dict):
        '''Create a Parameters instance.

        Args:
            config (dict): dictionary with match parameter values
            defaults (dict): dictionary with default params

        Returns:
            :mod:`namematch.data_structures.parameters.Parameters`: instance of the Parameters class
        '''

        param_dict = {}

        # issue warning if user specifies param not understood by code
        params_default = list(defaults.keys())[:]
        params_user_specified = list(config.keys())
        params_acceptable_non_default = ['existing_data_files',
                                         'data_files', 'variables']
        params_acceptable = params_default + params_acceptable_non_default
        params_unrecognized = [
                param for param in params_user_specified
                if param not in params_acceptable]
        if len(params_unrecognized) > 0:
            params_unrecognized = ', '.join(params_unrecognized)
            logger.warning(f"The following parameters are not "
                           f"recognized (will be ignored): {params_unrecognized}")
        params_only_incremental = []
        for param in params_only_incremental:
            if 'existing_data_files' not in config and param in config:
                logger.warning(f"The following parameters are not recognized "
                               f"for non-incremental matching (will be "
                               f"ignored): {param}")

        # add private params based on input params
        param_dict.update(defaults)

        # overwrite defaults and private config values with values defined in config
        for param, param_value in config.items():
            if isinstance(param_value, list) and param != 'variables':
                for value in param_value:
                    if (not Parameters.check_integrity(defaults, param, value)) and (param in params_default):
                        raise Exception(f"parameter '{param}' contains value '{value}' which is an invalid type")
                param_dict[param] = param_value
            else:
                if param in params_default:
                    if Parameters.check_integrity(defaults, param, param_value):
                        param_dict[param] = param_value
                    param_dict[param] = param_value
        param_dict['incremental'] = (len(config.get('existing_data_files', {})) > 0)

        if param_dict['num_workers'] > 1:
            param_dict['parallelize'] = True

        return cls(param_dict)

    @classmethod
    def load(cls, filepath):
        '''Load a Parameters instance.

        Args:
            filepath (str): path to a yaml version of a Parameters instance

        Returns:
            :mod:`namematch.data_structures.parameters.Parameters`: instance of the Parameters class
        '''

        param_dict = load_yaml(filepath)

        # NOTE: no need for validation because simple loading
        # an object that was validated when initialized

        return cls(param_dict)

    @classmethod
    def load_from_dict(cls, param_dict):
        return cls(param_dict)

    def check_for_required_variables(self, variables):
        '''Validate that the config includes required variables.'''

        missing_required_variables = []
        for required_var in self.required_variables:
            if required_var not in variables.get_names():
                missing_required_variables.append(required_var)

        if len(missing_required_variables) > 0:
            error_str = ', '.join(missing_required_variables)
            logger.error(f'The following required variable names '
                         f'are missing: {error_str}')
            raise ValueError

    def validate_exactmatch_variables(self, variables):
        '''Validate that the exact_match_variables and negate_exact_match_variables parameters.'''

        em_vars = self.exact_match_variables + self.negate_exact_match_variables

        invalid_variables = []
        valid_variables = variables.get_names()
        for variable in em_vars:
            if variable not in valid_variables:
                invalid_variables.append(variable)

        if len(invalid_variables) > 0:
            invalid_variables = ', '.join(invalid_variables)
            logger.warning("The exactmatch flag could not be calculated because the following "
                           f"variables are not defined: {invalid_variables}. You may need to adjust "
                           "the exact_match_variables and negate_exact_match_variables parameters.")


    def validate_blocking_scheme(self, variables):
        '''Validate that the blocking scheme is in the correct format and provieds the minimum
        number of blocking variables per blocking type (cosine_distance, edit_distance, absvalue_distance).'''

        blocking_params = ['cosine_distance', 'edit_distance',
                           'absvalue_distance', 'alpha', 'power']
        if any([v not in blocking_params for v in list(self.blocking_scheme.keys())]):
            logger.error("Blocking scheme is misspecified.")
            raise ValueError

        if len(self.blocking_scheme['cosine_distance']['variables']) > 2:
            logger.warning(f"Only the first two variables in blocking_scheme"
                           f"['cosine_distance'] will be used.")
            self.blocking_scheme['cosine_distance']['variable'] = \
                    self.blocking_scheme['cosine_distance']['variable'][0:2]

        if type(self.blocking_scheme['edit_distance']['variable']) == list:
            logger.warning(f"Only one variable allowed in blocking_scheme"
                           f"['edit_distance']; using first one.")
            self.blocking_scheme['edit_distance']['variable'] = \
                    self.blocking_scheme['edit_distance']['variable'][0]

        alpha = self.blocking_scheme['alpha']
        power = self.blocking_scheme['power']
        if not isinstance(alpha, (int, float)) or not isinstance(power, float):
            logger.error(f"Incorrect parameter definitions in blocking_scheme: "
                         f"alpha (numeric) and power (float).")
            raise ValueError

        invalid_variables = []
        valid_variables = variables.get_names()
        blocking_variables = self.get_blocking_variables()
        for variable in blocking_variables:
            if variable not in valid_variables:
                invalid_variables.append(variable)

        if len(invalid_variables) > 0:
            invalid_variables = ', '.join(invalid_variables)
            logger.error(f'The following variable names, needed for the specified '
                         f'blocking scheme, are missing: {invalid_variables}')
            raise ValueError

    def get_blocking_variables(self):
        '''Get list of blocking variable nicknames.

        Return:
            list of variable nicknames (all-names columns) to use for blocking
        '''

        bv = self.blocking_scheme['cosine_distance']['variables'][:]
        bv.append(self.blocking_scheme['edit_distance']['variable'])
        #bv.append(self.blocking_scheme['absvalue_distance']['variable'])
        bv = [bv for bv in bv if bv is not None]

        return bv

    def validate(self, variables):
        '''Validate several components of the config file.'''

        self.check_for_required_variables(variables)
        self.validate_blocking_scheme(variables)
        self.validate_exactmatch_variables(variables)

    def write(self, output_file):
        '''Write the Parameters to a yaml file.

        Args:
            output_file (str): path to write parameter dictionary
        '''

        dump_yaml(to_dict(self), output_file)

    def copy(self):
        '''Create a deep copy of a Parameters object.'''
        return deepcopy(self)

    def stage_params_lookup(self):
        stage_params_lookup = {}
        for k, v in params_lookup.items():
            stage_params_lookup[k] = {p: getattr(self, p, None) for p in v}
        return stage_params_lookup

    def get_stage_params(self, stage):
        return self.stage_params_lookup()[stage]
