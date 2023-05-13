import numpy as np
import matplotlib.pyplot as plt

from processmonitoring.datasets import register_function, GenericDataset

@register_function('SineWaveToRandom')
class SineWaveToRandom(GenericDataset.DatasetWithPermutations):

    def __init__(self, 
                 data_config: dict, 
                 save_to_folder: str = None
                 ) -> None:
        
        super().__init__(data_config, 
                         save_to_folder)

        self._generate_data()
        self._segment_and_split()
        self._generate_permuted_dataset()

        if self.save_to_folder:
            self._plot_to_folder()

    def _generate_data(self) -> None:

        dt = np.linspace(0, 1000, self.length)
        
        # individual sine wave component generation
        a = np.sin(0.10*dt)
        b = np.sin(dt)
        c = 0.1*np.sin(10*dt)
        d = np.random.normal(loc=0, scale=0.1, size=self.length)

        self.time_series = a+b+c+d

        # generate a shuffled copy
        shuffled_ts = np.random.permutation(self.time_series)

        # linearly increase the effect of the shuffle time series after fault
        alpha = np.linspace(1, 0, self.length - self.fault_pos)
        self.time_series[self.fault_pos:] = np.array(
            [self.time_series[j]*alpha[i] + shuffled_ts[j]*(1-alpha[i]) for i, j in enumerate(range(self.fault_pos, self.length))]
            )
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
