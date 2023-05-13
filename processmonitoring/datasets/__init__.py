__all__ = ['dataset_factory']

import os
import importlib

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('__'):
        module_name = file[:-len('.py')]
        importlib.import_module('processmonitoring.datasets.' + module_name)

__MODEL_DICT__ = {}

def register_function(name):

    def register_function_fn(cls):
        if name in __MODEL_DICT__:
            raise ValueError(f'Name {name} already registered.')
        __MODEL_DICT__[name] = cls
        return cls
    
    return register_function_fn

def dataset_factory(model_type, model_path):
    return __MODEL_DICT__[model_type](model_path)