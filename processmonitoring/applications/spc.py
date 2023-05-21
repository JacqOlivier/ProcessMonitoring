import numpy as np
import matplotlib.pyplot as plt

from processmonitoring.datasets import dataset
from processmonitoring.feature_extraction import GenericFeatureExtractor

class SPC():

    def __init__(self, 
                 dataset: dataset.DatasetWithPermutations, 
                 feature_extractor: GenericFeatureExtractor.FeatureExtractor,
                 save_to_folder: str = None) -> None:
        
        self.dataset = dataset
        self.feature_extractor = feature_extractor
        self.save_to_folder = save_to_folder
        self.data = np.concatenate([self.dataset.X_tr, self.dataset.X_val, self.dataset.X_test], axis=0)

        self._generate_feature_space_statistic()
        self._generate_residual_space_statistic()

    def _generate_feature_space_statistic(self):

        self.feature_space_stats = np.zeros((self.data.shape[0],))

        for i in range(self.data.shape[0]):
            ts_segment = self.data[i,:]
            self.feature_space_stats[i] = self.feature_extractor.eval_in_feature_space(ts_segment)

        self.feature_thresholds = np.percentile(self.feature_space_stats[:self.dataset.X_tr.shape[0]], [90, 95])
        
        if self.save_to_folder:
            self._plot_T2()

    def _generate_residual_space_statistic(self):

        self.residual_space_stats = np.zeros((self.data.shape[0],))

        for i in range(self.data.shape[0]):
            print(f'Iteration {i} of {self.data.shape[0]}')
            ts_segment = self.data[i,:]
            self.residual_space_stats[i] = self.feature_extractor.eval_in_residual_space(ts_segment)

        self.residual_thresholds = np.percentile(self.residual_space_stats[:self.dataset.X_tr.shape[0]], [90, 95])
        
        if self.save_to_folder:
            self._plot_residual()

    def _plot_T2(self):

        import os

        plot_limits = {
            'train' : self.dataset.X_tr.shape[0],
            'val' : self.dataset.X_tr.shape[0] + self.dataset.X_val.shape[0],
        }

        plt.figure(figsize=(16,9))
        plt.plot(list(range(plot_limits['train'])), self.feature_space_stats[:plot_limits['train']], c='b',
                 marker='o', markerfacecolor='None')
        plt.plot(list(range(plot_limits['train'], plot_limits['val'])), 
                 self.feature_space_stats[plot_limits['train']:plot_limits['val']], c='g',
                 marker='o', markerfacecolor='None')
        plt.plot(list(range(plot_limits['val'], self.data.shape[0])), 
                 self.feature_space_stats[plot_limits['val']:], c='r',
                 marker='o', markerfacecolor='None')
        plt.axhline(self.feature_thresholds[0], c='k', linestyle='--')
        plt.axhline(self.feature_thresholds[1], c='k', linestyle='--')
        plt.xlabel('Window number')
        plt.ylabel('Feature space score')
        plt.grid(True)
        plt.title(f'OC-SVM scores with 90th and 95th percentiles.')
        plt.savefig(os.path.join(self.save_to_folder, 'FeatureSpaceSPC'))

    def _plot_residual(self):

        import os

        plot_limits = {
            'train' : self.dataset.X_tr.shape[0],
            'val' : self.dataset.X_tr.shape[0] + self.dataset.X_val.shape[0],
        }

        plt.figure(figsize=(16,9))
        plt.plot(list(range(plot_limits['train'])), self.residual_space_stats[:plot_limits['train']], c='b',
                 marker='o', markerfacecolor='None')
        plt.plot(list(range(plot_limits['train'], plot_limits['val'])), 
                 self.residual_space_stats[plot_limits['train']:plot_limits['val']], c='g',
                 marker='o', markerfacecolor='None')
        plt.plot(list(range(plot_limits['val'], self.data.shape[0])), 
                 self.residual_space_stats[plot_limits['val']:], c='r',
                 marker='o', markerfacecolor='None')
        plt.axhline(self.residual_thresholds[0], c='k', linestyle='--')
        plt.axhline(self.residual_thresholds[1], c='r', linestyle='--')
        plt.xlabel('Window number')
        plt.ylabel('Residual space score')
        plt.grid(True)
        plt.title(f'Reconstruction scores with 90th and 95th percentiles.')
        plt.savefig(os.path.join(self.save_to_folder, 'ResidualSpaceSPC'))

