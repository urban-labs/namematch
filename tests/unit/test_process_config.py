from namematch.data_structures.schema import Schema
from namematch.data_structures.parameters import Parameters


def test_params_init(config_dict, default_params_dict):
        # test passing
        params = Parameters.init(
            config_dict,
            default_params_dict)

        schema = Schema.init(config_dict, params)

        # test wrong set_missing type
        config_dict['variables'][0]['drop'] = ','.join(config_dict['variables'][0]['drop'])
        params = Parameters.init(
                config_dict,
                default_params_dict)
        fine = 1
        try:
            schema = Schema.init(config_dict, params)
        except:
            fine = 0
        assert fine == 0
