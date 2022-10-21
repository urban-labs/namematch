__version__ = '1.2.0'
import os
import yaml

default_parameters = yaml.load(open(os.path.join(os.path.dirname(__file__), 'default_parameters.yaml'), 'r'), Loader=yaml.FullLoader)
default_parameters = dict((k,v) if v != 'None' else (k, None) for k,v in default_parameters.items())

