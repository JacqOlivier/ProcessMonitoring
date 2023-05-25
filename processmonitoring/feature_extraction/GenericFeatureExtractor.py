from processmonitoring.datasets import dataset
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

"""
Generic feature extraction class. 
Must be inherited by all feature extraction methods.
Class trains and stores models built on the generated datasets.
"""
class FeatureExtractor:

    def __init__(self, 
                 dataset: dataset.DatasetWithPermutations,
                 feature_config: dict, 
                 save_to_folder: str = None) -> None:
        """Constructor initialises some general model inputs.

        Args:
            dataset (dataset.DatasetWithPermutations): Dataset with permutations object used for model bulding.
            feature_config (dict): Model specific hyperparameters.
            save_to_folder (str, optional): Plot directory. Defaults to None.
        """
        
        self._dataset = dataset
        self._feature_config = feature_config
        self._save_to_folder = save_to_folder

        # these are the two feature sets to be produced by subclasses
        self._train_features = None            # feature set generated from X_tr and permuted data (only called if using permutatino method)
        self._features = None                  # full dataset (NOC/train, VAL/val, FAULT/test ) projected to feature space
        self._latent_dimension = None

    def train():
        """
        The train method is called by client, and triggers training of the forward and (sometimes) reverse mapping models.
        This will also trigger the generation of model-specific plots.
        The method must generate the following feature sets:
            self._train_proj_features
            self._all_proj_features
        This method executes all steps that would be involved in a 'ExtractFeatures' run configuration.
        Other configurations just have additional steps.
        """
        raise NotImplementedError()
    
    def eval_in_feature_space():
        """Evaluates a single window of data as a test case from client (SPE) code.
        """
        raise NotImplementedError()
    
    def eval_in_residual_space():
        """Evaluates a single window of data as a test case from client (SPE) code.
        """
        raise NotImplementedError()
    
    def _plot_train_features(self, y: np.ndarray) -> None:
        """
        By default, the train feature set is projected down to 2D using PCA for plotting.

        Args:
            y (np.ndarray): (N x 1) array containing labels of the training set
        """
        train_features_scaled = StandardScaler().fit_transform(self._train_features)
        pca_obj = PCA().fit(train_features_scaled)
        features = pca_obj.transform(train_features_scaled)

        plt.figure(figsize=(16, 9))
        plt.scatter(features[:,0][y==0], features[:,1][y==0], c='blue', marker='o')
        plt.scatter(features[:,0][y==1], features[:,1][y==1], c='black', marker='x')
        plt.xlabel(f'1st Principal Component: {pca_obj.explained_variance_ratio_[0]*100:.2f} % variance')
        plt.ylabel(f'2nd Principal Component: {pca_obj.explained_variance_ratio_[1]*100:.2f} % variance')
        plt.legend(['Real', 'Permuted'])
        plt.title(f'2D PCA of MDS coordinates in {self._latent_dimension}-dimensional space.')
        plt.grid(True)
        plt.savefig(fname=os.path.join(self._save_to_folder, self.__class__.__name__ + '_TrainFeatures'))

    def _plot_features(self) -> None:
        """
        By default, the feature set is projected down to 2D using PCA for plotting.

        Args:
            y (np.ndarray): (N x 1) array containing labels of the training set
        """
        features_scaled = StandardScaler().fit_transform(self._features)
        pca_obj = PCA().fit(features_scaled)
        features = pca_obj.transform(features_scaled)

        plt.figure(figsize=(16, 9))
        plt.scatter(features[:,0][self._dataset.y==0], features[:,1][self._dataset.y==0], c='blue', marker='o')
        plt.scatter(features[:,0][self._dataset.y==1], features[:,1][self._dataset.y==1], c='green', marker='x')
        plt.scatter(features[:,0][self._dataset.y==2], features[:,1][self._dataset.y==2], c='red', marker='*')
        plt.xlabel(f'1st Principal Component: {pca_obj.explained_variance_ratio_[0]*100:.2f} % variance')
        plt.ylabel(f'2nd Principal Component: {pca_obj.explained_variance_ratio_[1]*100:.2f} % variance')
        plt.legend(['Train', 'Validation', 'Fault'])
        plt.title(f'2D MDS coordinates in {self._latent_dimension}-dimensional space.')
        plt.grid(True)
        plt.savefig(fname=os.path.join(self._save_to_folder, self.__class__.__name__ + '_Features'))

        # fit an SVM on the 1st two principal components of the NOC data
        OCSVM = OneClassSVM().fit(features[:,:2][self._dataset.y==0])

        y_pred_train = OCSVM.predict(features[:, :2][self._dataset.y==0])
        y_pred_val = OCSVM.predict(features[:, :2][self._dataset.y==1])
        y_pred_fault = OCSVM.predict(features[:, :2][self._dataset.y==2])
        n_error_train = y_pred_train[y_pred_train == -1].size
        n_error_test = y_pred_val[y_pred_val == -1].size
        n_error_outliers = y_pred_fault[y_pred_fault == 1].size

        xx, yy = np.meshgrid(np.linspace(np.min(features[:,0]), np.max(features[:,0]), 500), 
                             np.linspace(np.min(features[:,1]), np.max(features[:,1]), 500))

        # plot the line, the points, and the nearest vectors to the plane
        Z = OCSVM.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(16, 9))
        plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
        a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors="darkred")
        plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors="palevioletred")

        s = 40
        b1 = plt.scatter(features[:,0][self._dataset.y==0], features[:,1][self._dataset.y==0], c="white", s=s, edgecolors="k")
        b2 = plt.scatter(features[:,0][self._dataset.y==1], features[:,1][self._dataset.y==1], c="blueviolet", s=s, edgecolors="k")
        c = plt.scatter(features[:,0][self._dataset.y==2], features[:,1][self._dataset.y==2], c="gold", s=s, edgecolors="k")
        plt.axis("tight")
        plt.xlim((np.min(features[:,0]), np.max(features[:,0])))
        plt.ylim((np.min(features[:,1]), np.max(features[:,1])))
        plt.legend(
            [a.collections[0], b1, b2, c],
            [
                "learned boundary",
                "X NOC",
                "X val",
                "Fault",
            ],
            loc="upper left",
            prop=matplotlib.font_manager.FontProperties(size=11),
        )
        plt.xlabel(
            "error train: %d/%d ; errors novel regular: %d/%d ; errors novel abnormal: %d/%d"
            % (n_error_train, len(y_pred_train), n_error_test, len(y_pred_val), n_error_outliers, len(y_pred_fault))
        )
        plt.grid(True)
        plt.savefig(fname=os.path.join(self._save_to_folder, 'OCSVMPredictionsWithFault'))

        # also use full feature space and output results to json file
        OCSVM = OneClassSVM().fit(features[:,:][self._dataset.y==0])
        y_pred_train = OCSVM.predict(features[:,:][self._dataset.y==0])
        y_pred_val = OCSVM.predict(features[:,:][self._dataset.y==2])
        y_pred_fault = OCSVM.predict(features[:,:][self._dataset.y==1])
        n_error_train = y_pred_train[y_pred_train == -1].size
        n_error_val = y_pred_val[y_pred_val == -1].size
        n_error_fault = y_pred_fault[y_pred_fault == 1].size

        res = {
            "train" : {
                "num_samples": features[:,:][self._dataset.y==0].shape[0],
                "accuracy": (len(y_pred_train) - n_error_train) / len(y_pred_train) * 100
            }, 
            "validation" : {
                "num_samples": features[:,:][self._dataset.y==1].shape[0],
                "accuracy": (len(y_pred_val) - n_error_val) / len(y_pred_val) * 100
            },
            "fault" : {
                "num_samples": features[:,:][self._dataset.y==2].shape[0],
                "accuracy": (len(y_pred_fault) - n_error_fault) / len(y_pred_fault) * 100
            }
        }

        import json
        with open(os.path.join(self._save_to_folder, 'FullFeatureSpaceResults.json'), 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=4)