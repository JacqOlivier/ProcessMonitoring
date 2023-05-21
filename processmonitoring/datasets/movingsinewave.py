from processmonitoring.datasets import dataset
from processmonitoring.datasets.utils import register_function

import numpy as np
import matplotlib.pyplot as plt

@register_function('MovingSineWave')
class MovingSineWaveGenerator(dataset.DatasetWithPermutations):

    def __init__(self, 
                 data_config: dict,
                 save_to_folder: str = None,
                 coefficients: list = [0.11, 0.12]):
        
        super().__init__(data_config, 
                         save_to_folder)
        
        self.coefficients = coefficients
        self._generate_data()
        self._segment_and_split()
        self._generate_permuted_dataset()

        if self.save_to_folder:
            self._plot_to_folder()

    def _generate_data(self) -> None:

        dt = np.linspace(0, 1000, self.length)
        
        # individual sine wave component generation
        a = np.concatenate([np.sin(dt[:self.fault_pos]*self.coefficients[0]), 
                            np.sin(dt[self.fault_pos:]*self.coefficients[1])])
        b = np.sin(dt)
        c = 0.1*np.sin(10*dt)
        d = np.random.normal(loc=0, scale=0.1, size=self.length)
        
        self.time_series = a+b+c+d
        self.labels = np.concatenate([np.zeros((self.fault_pos,)), np.ones(self.length-self.fault_pos,)])

    def _plot_to_folder(self):

        import os

        plt.figure(figsize=(16,9))
        plt.plot(self.time_series)
        plt.axvline(self.fault_pos, c='r', linestyle='--')
        plt.xlabel('Time')
        plt.ylabel('y')
        plt.grid(True)
        plt.legend(["Time series", "Fault Position"])
        plt.savefig(fname=os.path.join(self.save_to_folder, 'MovingSineWaveTimeSeries'))

if __name__ == "__main__":

    sinewave = MovingSineWaveGenerator(1000, 0.5, 50)
    