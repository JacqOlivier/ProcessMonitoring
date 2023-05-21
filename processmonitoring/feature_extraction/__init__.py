import os
import importlib

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('__'):
        module_name = file[:-len('.py')]
        importlib.import_module('processmonitoring.feature_extraction.' + module_name)