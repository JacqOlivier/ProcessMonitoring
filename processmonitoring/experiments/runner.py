import json
from processmonitoring import datasets
#from processmonitoring.feature_extraction import RandomForestFeatureExtraction, AutoEncoder, CNNFeatures
#from processmonitoring.applications import spc

class ExperimentRunner:

    def __init__(self,
                 config: json, 
                 save_to_folder: str = None) -> None:
        """Constructor parses the json file and initiates the different parts of the experiment

        Args:
            config (_type_): _description_
        """
        self._data_config = config['dataset']
        self._feature_config = config['feature_extraction']
        self._run_config = config['mode']

        self._save_to_folder = save_to_folder

        self.dataset = datasets.dataset_factory(self._data_config['name'])
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