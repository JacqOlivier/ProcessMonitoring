import logging
from processmonitoring.datasets import utils

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

        self._logger.debug('Hi From runnier.')

        self.dataset = utils.dataset_factory(self._data_config['name'])
        #self.dataset = dataset_factory(self._data_config['name'])
        """
        if self._data_config['name'] == 'MovingSineWave':
            self.dataset = MovingSineWave.MovingSineWaveGenerator(self._data_config['simulation_length'], 
                                                                  self._data_config['transition_position'], 
                                                                  self._data_config['permutation'], 
                                                                  self._data_config['window_length'], 
                                                                  self._data_config['stride'], 
                                                                  self._save_to_folder)
        elif self._data_config['name'] == 'SineWaveToRandom':
            self.dataset = SineWaveToRandom.SineWaveToRandom(self._data_config['simulation_length'], 
                                                             self._data_config['transition_position'], 
                                                             self._data_config['permutation'], 
                                                             self._data_config['window_length'], 
                                                             self._data_config['stride'], 
                                                             self._save_to_folder)
        else:
            raise RuntimeError(f'Invalid dataset selected: {self._data_config["name"]}')
        
        if self._feature_config['name'] == 'RandomForestFeatureExtraction':
            self.model = RandomForestFeatureExtraction.RandomForestFeatures(self.dataset, 
                                                                            self._feature_config['num_trees'], 
                                                                            self._save_to_folder)
        elif self._feature_config['name'] == 'AutoEncoderFeatures':
            self.model = AutoEncoder.AutoEncoderFeatures(self.dataset, 
                                                         self._feature_config, 
                                                         self._save_to_folder)
        elif self._feature_config['name'] == 'CNNFeatures':
            self.model = CNNFeatures.CNN(self.dataset, 
                                         self._feature_config, 
                                         self._save_to_folder)
        else: 
            raise RuntimeError(f'Invalid feature extraction method selected: {self._feature_config["name"]}')
        
        ### Model training step 
        self.model.train(mode=self._run_config)

        if self._run_config['name'] == 'StatisticalProcessControl': 
            self.runner = spc.SPC(self.dataset, 
                                  self.model,
                                  self._save_to_folder)

        """