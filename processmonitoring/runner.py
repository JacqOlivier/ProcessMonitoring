import logging
from processmonitoring.datasets.utils import *
from processmonitoring.feature_extraction.utils import *

class ExperimentRunner:

    def __init__(self,
                 config: dict, 
                 save_to_folder: str = None) -> None:
        """Constructor parses the json file and initiates the different parts of the experiment

        Args:
            config (dict): JSON config file to be parsed for experiment details
            save_to_folder (str): Path to folder generated to store plots.
        """
        self._data_config = config['dataset']
        self._feature_config = config['feature_extraction']
        self._run_config = config['mode']
        self._save_to_folder = save_to_folder
        self._logger = logging.getLogger(__name__)

        # generate dataset
        try:
            self._dataset = dataset_factory(self._data_config['name'])(self._data_config, 
                                                                       self._save_to_folder)
        except:
            self._logger.exception(f'Error trying to generate dataset: {self._data_config["name"]}.')
            raise
        
        # instantiate model
        try:
            self._model = feature_factory(self._feature_config['name'])(self._dataset,
                                                                        self._feature_config, 
                                                                        self._save_to_folder)
        except:
            self._logger.exception(f'Error trying to generate feature model: {self._feature_config["name"]}.')
            raise
        
    def run(self) -> None:

        ### Model training step 
        self._model.train(mode=self._run_config)