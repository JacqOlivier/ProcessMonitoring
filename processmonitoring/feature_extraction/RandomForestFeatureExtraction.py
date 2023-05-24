import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils import shuffle
from sklearn.manifold import MDS
from sklearn.svm import OneClassSVM
from processmonitoring.datasets import dataset
from processmonitoring.feature_extraction import GenericFeatureExtractor
from processmonitoring.feature_extraction.utils import register_function

import matplotlib.pyplot as plt
import matplotlib
import os

"""
This class executes Random Forest Feature Extraction in an unsupervised manner
by training the model to distinguish permuted copies of the original data.
"""
@register_function('RandomForestFeatureExtraction')
class RandomForestFeatures(GenericFeatureExtractor.FeatureExtractor):

    def __init__(self, 
                 dataset: dataset.DatasetWithPermutations, 
                 feature_config: dict, 
                 save_to_folder: str = None) -> None:
        
        super().__init__(dataset=dataset, feature_config=feature_config, save_to_folder=save_to_folder)
        self._n_estimators = self._feature_config['num_trees']

    def train(self, mode: dict) -> None:
        """Call the model to train, with modes specified in config file.

        Args:
            mode (dict): mode dict must be passed from caller
        """
        # generate a data set with NOC and permuted data; labels: real = 0, permuted = 1
        # such a dataset not currently generated automatically in Dataset classes
        X = np.concatenate([self._dataset.X_tr, self._dataset.X_tr_permuted], axis=0)
        y = np.concatenate([self._dataset.y_tr, self._dataset.y_tr_permuted])
        X, y = shuffle(X, y)

        # this method does the Random Forest Feature Extraction
        self._train_base_model(X, y)
        
        # train forward mapping models here using training data, but we are still just extracting features (no reconstruction)
        self._train_forward_mapping(X)

        # now we project the original dataset (NOC, val, fault) to the feature space
        self._features = np.zeros((self._dataset.X.shape[0], self._latent_dimension))
        for i in range(self._dataset.X.shape[0]):
            # each row (window) is evaluated separately
            self._features[i,:] = self.eval_in_feature_space(self._dataset.X[i,:])

        if self._save_to_folder:
            self._plot_train_features(y)
            self._plot_features()
        
    def _train_base_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """This function generates the feature space using NOC and permuted NOC data.

        Args:
            X (np.ndarray): NOC and permuted data windows, shuffled
            y (np.ndarray): labels (0=NOC, 1=permuted)
        """
        # train a RF model to classify real vs permuted data
        mdl = RandomForestClassifier(n_estimators=self._n_estimators).fit(X, y)
        # generate a proximity matrix from the samples in X
        # NOTE: This function currently uses full X-matrix that was also used for training.
        proximity_mat = self._proximity_matrix(mdl, X, plot=True)
        # calculate dissimilarity matrix (1 - P)
        dissimilarity_mat = np.eye(proximity_mat.shape[0]) - proximity_mat
        # fit a MDS model to the dissimilarity matrix
        # first need to get crossing dimension as estimate of data dimensionality
        self._get_manifold_dimension(dissimilarity_mat)
        # project all training data to manifold with calculated dimension
        self._train_features = self._generate_manifold(dissimilarity_mat)

    def eval_in_feature_space(self, sample: np.ndarray) -> np.ndarray:
        """This method is called in both 'ExtractFeatures' and 'SPC' to project single samples to the feature space.

        Args:
            sample (np.ndarray): (1 x M) sample from segmented time series (M = window length)

        Returns:
            fs_sample (np.ndarray):  (1 x self._manifold_dimension) projection of sample to feature space
        """

        # forward mapping to feature space
        fs_sample = np.zeros((self._latent_dimension))
        for i in range(len(fs_sample)):
            # the sample is mapped to the feature space 
            # each RF model only returns coordinate in one dimension
            fs_sample[i] = self.forward_models[i].predict(sample.reshape(1,-1))

        return fs_sample
    
    def eval_in_residual_space(self, sample: np.ndarray) -> float:

        # forward map to feature space
        fs_sample = np.zeros((self._feature_set.shape[1]))
        for i in range(len(fs_sample)):
            fs_sample[i] = self.forward_models[i].predict(sample.reshape(1,-1))

        # reverse map back to reconstruction space
        reconstruction = np.zeros_like(sample)
        for i in range(len(reconstruction)):
            reconstruction[i] = self.reverse_models[i].predict(fs_sample.reshape(1,-1))

        # calculate SPE for this sample
        return np.sum(np.square(sample-reconstruction))

    def _proximity_matrix(self, 
                          model: RandomForestClassifier, 
                          X: np.ndarray, 
                          normalize: bool=True, 
                          plot: bool=True) -> np.ndarray:      
        """Generate proximity matrix.

        Args:
            model (RandomForestClassifier): RF-classifier trained on NOC and permuted data
            X (np.ndarray): Training set (N x M)
            normalize (bool, optional): Whether to scale resulting matrix by number of estimators. Defaults to True.
            plot (bool, optional): Whether to plot and save. Defaults to True.

        Returns:
            np.ndarray: N x N proximity matrix
        """
        # get terminal nodes
        terminals = model.apply(X)

        a = terminals[:,0]
        proxMat = 1*np.equal.outer(a, a)

        for i in range(1, self._n_estimators):
            a = terminals[:,i]
            proxMat += 1*np.equal.outer(a, a)

        if normalize:
            proxMat = proxMat / self._n_estimators
       
        if plot:
            plt.figure(figsize=(16,9))
            plt.imshow(proxMat, cmap='viridis', interpolation='nearest')
            plt.colorbar()
            plt.xlabel('Sample number x')
            plt.ylabel('Sample number y')
            plt.title(f'Random Forest Proximity matrix.')
            plt.savefig(fname=os.path.join(self._save_to_folder, 'RandomForestProximityMatrix'))
        
        return proxMat   
    
    def _get_manifold_dimension(self, X: np.ndarray, dimension: int = 20) -> None:
        """
        Try to get an estimate of the dimensionality of the data by by comparing the eigenvalues
        calculated in a scaled feature space (using MDS), with the eigenvalues of a permuted dataset.
        The N components with eigenvalues larger than that of the permuted eigenvalues is used as dimension.

        Args:
            X (np.ndarray): Dissimilarity matrix
            dimension (int, optional): Initial amount of dimensions for the MDS algorithm. Defaults to 20.

        Sets the manifold dimension for as class member.
        """

        manifold = MDS(n_components=dimension)
        projection = manifold.fit_transform(X)

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # now use PCA to calculate eigenvalues for each dimension in the projection
        X_sc = StandardScaler().fit_transform(projection)
        pcs = PCA().fit(X_sc)

        # shuffle for baseline
        X_sc_permuted = X_sc.copy()
        for i in range(dimension):
            X_sc_permuted[:,i] = np.random.permutation(X_sc_permuted[:,i])
        pcs_permuted = PCA().fit(X_sc_permuted)

        self._latent_dimension = np.argmax(pcs_permuted.explained_variance_ratio_ > pcs.explained_variance_ratio_)

        if self._save_to_folder:
            plt.figure(figsize=(16,9))
            plt.plot(list(range(dimension)), pcs.explained_variance_ratio_*100, marker='o')
            plt.plot(list(range(dimension)), pcs_permuted.explained_variance_ratio_*100, marker='x', c='k')
            plt.xlabel('Principal component number')
            plt.ylabel('% variance explained')
            plt.legend(['MDS Features', 'Shuffled Dataset'])
            plt.title(f'Determining number of manifold dimensions using PCA and a dummy dataset')
            plt.savefig(fname=os.path.join(self._save_to_folder, 'MDSFeaturesPCA'))

    def _generate_manifold(self, X: np.ndarray) -> np.ndarray:

        manifold = MDS(n_components=self._latent_dimension)
        return manifold.fit_transform(X)
    
    def _train_forward_mapping(self, X: np.ndarray) -> None:
        """
        Trains a model to map from sequences of time series data, to each of the dimensions in the feature space.
        The model learns to map the training data to the feature space representation of the data.
        Each model maps samples to a single dimension, results need to be concatenated to generate feature space.
        Args:
            X (np.ndarray): (N x M) training dataset.
        """

        self.forward_models = {
            i:RandomForestRegressor(self._n_estimators).fit(X, self._train_features[:,i]) for i in range(self._train_features.shape[1])
        }

    def _train_reverse_mapping(self, X: np.ndarray) -> None:
        ## TODO: this is wrong, but not using since SPC not finished
        self.reverse_models = {
            i:RandomForestRegressor(self._n_estimators).fit(self._train_features, X[:,i]) for i in range(X.shape[1])
        }

if __name__ == "__main__":
    
    from processmonitoring.datasets import movingsinewave

    dataset = movingsinewave.MovingSineWaveGenerator(1000, 0.5, 'iaaft', 100)

    model = RandomForestFeatures(dataset, 100)
    model.train()






