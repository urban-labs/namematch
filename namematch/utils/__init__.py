import os
import yaml

default_parameter = yaml.load(open(os.path.join(os.path.dirname(__file__), 'default_parameters.yaml'), 'r'), Loader=yaml.FullLoader)
