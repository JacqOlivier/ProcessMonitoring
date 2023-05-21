import numpy as np
from sklearn.model_selection import train_test_split

from processmonitoring.permutations import permutations

class DatasetWithPermutations(object):

    def __init__(self, 
                 data_config: dict,
                 save_to_folder: str = None
                 ) -> None:
        
        self.length = data_config['simulation_length']
        self.transition_fraction = data_config['transition_position']
        self.fault_pos = int(data_config['transition_position']*self.length)
        self.permutation = data_config['permutation']
        self.window_length = data_config['window_length']
        self.stride = data_config['stride']
        self.seed = data_config['seed']
        self.save_to_folder = save_to_folder
        
        self.time_series = None
        self.labels = None

    def _generate_data(self):
        # specific data generation scheme of subclass
        # raise error if not implemented in subclasses
        raise NotImplementedError()
    
    def _segment_and_split(self):

        # segment time series into overlapping sequences 
        self.num_windows = int(((len(self.time_series) - self.window_length)) / self.stride)

        self.X = np.zeros((self.num_windows, self.window_length))
        self.y = np.zeros(self.num_windows)
        for i in range(self.num_windows):
            self.X[i,:] = self.time_series[i*self.stride : i*self.stride+self.window_length]
            y_window = self.labels[i*self.stride : i*self.stride+self.window_length]
            if 1 in y_window:
                self.y[i] = 1
            else:
                self.y[i] = 0

        # divide into train-test splits
        self.X_tr, self.X_test, self.y_tr, self.y_test = train_test_split(self.X, 
                                                                          self.y, 
                                                                          test_size=1-self.transition_fraction, 
                                                                          random_state=42)
        
        # make another validation split
        self.X_tr, self.X_val, self.y_tr, self.y_val = train_test_split(self.X_tr,
                                                                          self.y_tr, 
                                                                          test_size=0.3,
                                                                          random_state=42)
        self.y_val = np.zeros_like(self.y_val) + 2
        
    def _generate_permuted_dataset(self):

        # get unrolled length of self.X_tr to determine amount of samples required for training
        real_training_data = self.time_series[:self.X_tr.shape[0]*self.stride+self.window_length]

        permutation = permutations.Permutations(real_training_data, 123, self.save_to_folder)
        permuted_ts = permutation.get_ts(self.permutation)

        #permutations.Permutations.compare_autocorrelation(real_training_data, permuted_ts)
        #permutations.Permutations.compare_distribution(real_training_data, permuted_ts)

        # segment the permuted time series
        num_windows = int(((len(permuted_ts) - self.window_length)) / self.stride)
        self.X_tr_permuted = np.zeros((num_windows, self.window_length))
        for i in range(num_windows):
            self.X_tr_permuted[i,:] = permuted_ts[i*self.stride : i*self.stride+self.window_length]
        
        self.y_tr_permuted = np.ones((num_windows,))