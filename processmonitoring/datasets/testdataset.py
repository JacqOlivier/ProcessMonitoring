from processmonitoring.datasets import dataset
from processmonitoring.datasets.utils import register_function

import numpy as np

@register_function('Test')
class TestData(dataset.DatasetWithPermutations):

    def __init__(self, 
                 data_config: dict,
                 save_to_folder: str = None) -> None:
        
        super().__init__(data_config, 
                         save_to_folder)

    def _generate_data(self) -> None:

        dt = np.linspace(0, 1000, self.length)
        
        # individual sine wave component generation
        a = np.concatenate([np.sin(dt[:self.fault_pos]*0.1), 
                            np.sin(dt[self.fault_pos:]*0.5)])
        b = np.sin(dt)
        c = 0.1*np.sin(10*dt)
        d = np.random.normal(loc=0, scale=0.1, size=self.length)
        
        self.time_series = a+b+c+d
        self.labels = np.concatenate([np.zeros((self.fault_pos,)), np.ones(self.length-self.fault_pos,)])

if __name__ == "__main__":

    sinewave = TestData(1000, 0.5, 50)
    