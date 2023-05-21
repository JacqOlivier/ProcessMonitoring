import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from nolitsa import surrogates
import os

"""
Class generates a permuted dataset from input time series.
"""
class Permutations():

    def __init__(self,
                 time_series : np.ndarray, 
                 prng, 
                 save_to_folder : str = None) -> None:
        """Constructor.

        Args:
            time_series (np.ndarray): Time series, self.time_series object from DatasetWithPermutations should be passed.
            prng (np.random.default_rng()): np.default_rng() object
            save_to_folder (str, optional): Plotting directory. Defaults to None.
        """

        self.ts = time_series
        self.save_to_folder = save_to_folder
        self._PRNG = prng

    def get_ts(self, permutation: str):

        if permutation == 'random shuffle':
            return self._random_shuffle()
        elif permutation == 'white noise':
            return self._white_noise()
        elif permutation == 'iaaft':
            return self._iaaft()
        else:
            raise RuntimeError(f'Permutation method: {permutation} not implemented.')

    def _random_shuffle(self):
        res = self._PRNG.permutation(self.ts)
        if self.save_to_folder:
            self.plot(self.ts, res)
            self.compare_distribution(self.ts, res)
            self.compare_autocorrelation(self.ts, res)
        return res
    
    def _white_noise(self):
        res = self._PRNG.normal(np.mean(self.ts), np.std(self.ts), len(self.ts))
        if self.save_to_folder:
            self.plot(self.ts, res)
            self.compare_distribution(self.ts, res)
            self.compare_autocorrelation(self.ts, res)
        return res
    
    def _iaaft(self):
        res, _, _ = surrogates.iaaft(self.ts, maxiter=10000) 
        if self.save_to_folder:
            self.plot(self.ts, res)
            self.compare_distribution(self.ts, res)
            self.compare_autocorrelation(self.ts, res)
        return res

    def plot(self, real_ts, permuted_ts):
        
        plt.figure(figsize=(16,9))
        plt.plot(real_ts)
        plt.plot(permuted_ts, c='r')
        plt.xlabel('Time')
        plt.ylabel('y')
        plt.grid(True)
        plt.legend(["Real Time Series", "Permuted Time Series"])
        plt.savefig(os.path.join(self.save_to_folder, 'RealVsPermutedTS'))

    def compare_autocorrelation(self, real_ts, permuted_ts):

        ts_acorr = sm.tsa.acf(real_ts)
        permuted_ts_acorr = sm.tsa.acf(permuted_ts)

        fig, ax = plt.subplots(2,1, figsize=(16,9))
        ax[0].plot(ts_acorr[:300])
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('Real Time Series ACF')
        ax[0].grid(True)

        ax[1].plot(permuted_ts_acorr[:300])
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('Permuted Time Series ACF')
        ax[1].grid(True)

        fig.savefig(os.path.join(self.save_to_folder, 'RealAndPermutedACF'))

    def compare_distribution(self, real_ts, permuted_ts):

        fig, ax = plt.subplots(2,1, figsize=(16,9))
        ax[0].hist(real_ts, bins=25, edgecolor='k')
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('Real Time Series Distribtion')
        ax[0].grid(True)

        ax[1].hist(permuted_ts, bins=25, edgecolor='k')
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('Permuted Time Series ACF')
        ax[1].grid(True)

        fig.savefig(os.path.join(self.save_to_folder, 'RealAndPermutedDistributions'))
    
if __name__ == "__main__":

    from processmonitoring.datasets import movingsinewave

    time_series = movingsinewave.MovingSineWaveGenerator(1000, 0.5)
    permute = Permutations(time_series.ts, 123)

    shuffle_ts = permute.random_shuffle()
    white_noise_ts = permute.white_noise()
    iaaft_ts = permute.iaaft()

    Permutations.plot(time_series.ts, iaaft_ts)



__all__ = ['Permutations']