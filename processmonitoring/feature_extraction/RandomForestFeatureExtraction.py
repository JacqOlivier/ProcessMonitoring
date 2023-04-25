import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.manifold import MDS
from sklearn.svm import OneClassSVM
from processmonitoring.datasets import GenericDataset
from processmonitoring.feature_extraction import GenericFeatureExtractor

import matplotlib.pyplot as plt
import matplotlib
import os

class RandomForestFeatures(GenericFeatureExtractor.FeatureExtractor):

    def __init__(self, 
                 dataset: GenericDataset.DatasetWithPermutations, 
                 num_trees: int, 
                 save_to_folder: str = None
                 ) -> None:
        
        super().__init__(dataset=dataset, save_to_folder=save_to_folder)
        self.n_estimators = num_trees

    def train(self, mode: dict) -> None:
        """Call the model to train, with modes specified in config file.

        Args:
            mode (dict): mode dict must be passed from caller
        """
        # generate train and test datasets
        X = np.concatenate([self.dataset.X_tr, self.dataset.X_tr_permuted], axis=0)
        y = np.concatenate([self.dataset.y_tr, self.dataset.y_tr_permuted])

        mdl = RandomForestClassifier(n_estimators=self.n_estimators).fit(X, y)

        proximity_mat = self._proximity_matrix(mdl, X, plot=True)
        # calculate dissimilarity matrix
        dissimilarity_mat = np.eye(proximity_mat.shape[0]) - proximity_mat
        # fit a MDS model to the dissimilarity matrix
        self._get_manifold_dimension(dissimilarity_mat)
        self._feature_set = self._generate_manifold(dissimilarity_mat)

        # train forward and reverse mapping models to map data to extracted feature set
        self._train_forward_mapping(X)

        if mode['name'] == 'StatisticalProcessControl':
            # for feature space spc
            self._OCSVM = OneClassSVM(gamma='auto').fit(self._feature_set)
            # for residual space spc
            self._train_reverse_mapping(X)
        elif mode['name'] == 'ExtractFeatures' or self.save_to_folder:
            self._plot_features_vs_permuted2D(y)
            self._project_NOCFault_features2D()
            #self.feature_space_svm()
        else:
            raise RuntimeError(f'Invalid run mode: {mode["name"]}, specified for RandomForestFeatureExtraction')
    
    def eval_in_feature_space(self, sample: np.ndarray, ocsvm=True) -> float:

        # forward mapping to feature space
        fs_sample = np.zeros((self._feature_set.shape[1]))
        for i in range(len(fs_sample)):
            fs_sample[i] = self.forward_models[i].predict(sample.reshape(1,-1))

        if ocsvm:
            # now produce sample in feature space to trained OCSVM
            return self._OCSVM.score_samples(fs_sample.reshape(1,-1))
        else:
            # for when just extracting features
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

    def _plot_features_vs_permuted2D(self, y):

        import os

        colors = ['blue','black']
        plt.figure(figsize=(16, 9))
        if self._manifold_dimension > 1:
            plt.scatter(self._feature_set[:,0][y==0], self._feature_set[:,1][y==0], c='blue', marker='o')
            plt.scatter(self._feature_set[:,0][y==1], self._feature_set[:,1][y==1], c='black', marker='x')
        else:
            plt.scatter(list(range(self._feature_set.shape[0])), self._feature_set[:,0], c=y, cmap=matplotlib.colors.ListedColormap(colors))
        plt.xlabel('1st MDS component')
        plt.ylabel('2nd MDS component')
        plt.legend(['Real', 'Permuted'])
        plt.title(f'2D MDS coordinates in {self._manifold_dimension}-dimensional space.')
        plt.grid(True)
        plt.savefig(fname=os.path.join(self.save_to_folder, 'RandomForestFeaturesNOCvsPermuted'))

    def _project_NOCFault_features2D(self):

        X = np.concatenate([self.dataset.X_tr, self.dataset.X_val, self.dataset.X_test])
        y = np.concatenate([self.dataset.y_tr, self.dataset.y_val, self.dataset.y_test])

        self._projected_features = np.zeros((X.shape[0], self._feature_set.shape[1]))
        for i in range(X.shape[0]):
            self._projected_features[i,:] = self.eval_in_feature_space(X[i,:], ocsvm=False)

        colors = ['blue','green', 'red']
        plt.figure(figsize=(16, 9))
        if self._manifold_dimension > 1:
            plt.scatter(self._projected_features[:,0][y==0], self._projected_features[:,1][y==0], c='blue', marker='o')
            plt.scatter(self._projected_features[:,0][y==2], self._projected_features[:,1][y==2], c='green', marker='x')
            plt.scatter(self._projected_features[:,0][y==1], self._projected_features[:,1][y==1], c='red', marker='*')
        else:
            plt.scatter(list(range(self._projected_features[:,0])), self._projected_features[:,0], c=y, cmap=matplotlib.colors.ListedColormap(colors))
        plt.xlabel('1st MDS component')
        plt.ylabel('2nd MDS component')
        plt.legend(['Train', 'Validation', 'Fault'])
        plt.title(f'2D MDS coordinates in {self._manifold_dimension}-dimensional space.')
        plt.grid(True)
        plt.savefig(fname=os.path.join(self.save_to_folder, 'RandomForestFeaturesWithFault'))

        OCSVM = OneClassSVM().fit(self._projected_features[:,:2][y==0])

        y_pred_train = OCSVM.predict(self._projected_features[:, :2][y==0])
        y_pred_val = OCSVM.predict(self._projected_features[:, :2][y==2])
        y_pred_fault = OCSVM.predict(self._projected_features[:, :2][y==1])
        n_error_train = y_pred_train[y_pred_train == -1].size
        n_error_test = y_pred_val[y_pred_val == -1].size
        n_error_outliers = y_pred_fault[y_pred_fault == 1].size

        xx, yy = np.meshgrid(np.linspace(np.min(self._projected_features[:,0]), np.max(self._projected_features[:,1]), 500), 
                             np.linspace(np.min(self._projected_features[:,0]), np.max(self._projected_features[:,1]), 500))

        # plot the line, the points, and the nearest vectors to the plane
        Z = OCSVM.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(16, 9))
        plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
        a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors="darkred")
        plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors="palevioletred")

        s = 40
        b1 = plt.scatter(self._projected_features[:,0][y==0], self._projected_features[:, 1][y==0], c="white", s=s, edgecolors="k")
        b2 = plt.scatter(self._projected_features[:,0][y==2], self._projected_features[:,1][y==2], c="blueviolet", s=s, edgecolors="k")
        c = plt.scatter(self._projected_features[:,0][y==1], self._projected_features[:,1][y==1], c="gold", s=s, edgecolors="k")
        plt.axis("tight")
        plt.xlim((np.min(self._projected_features[:,0]), np.max(self._projected_features[:,0])))
        plt.ylim((np.min(self._projected_features[:,1]), np.max(self._projected_features[:,1])))
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
        plt.savefig(fname=os.path.join(self.save_to_folder, 'OCSVMPredictionsWithFault'))

        # also use full feature space and output results to json file
        OCSVM = OneClassSVM().fit(self._projected_features[:,:][y==0])
        y_pred_train = OCSVM.predict(self._projected_features[:,:][y==0])
        y_pred_val = OCSVM.predict(self._projected_features[:,:][y==2])
        y_pred_fault = OCSVM.predict(self._projected_features[:,:][y==1])
        n_error_train = y_pred_train[y_pred_train == -1].size
        n_error_val = y_pred_val[y_pred_val == -1].size
        n_error_fault = y_pred_fault[y_pred_fault == 1].size

        res = {
            "train" : {
                "num_samples": self.dataset.X_tr.shape[0],
                "accuracy": (len(y_pred_train) - n_error_train) / len(y_pred_train) * 100
            }, 
            "validation" : {
                "num_samples": self.dataset.X_val.shape[0],
                "accuracy": (len(y_pred_val) - n_error_val) / len(y_pred_val) * 100
            },
            "fault" : {
                "num_samples": self.dataset.X_test.shape[0],
                "accuracy": (len(y_pred_fault) - n_error_fault) / len(y_pred_fault) * 100
            }
        }

        import json
        with open(os.path.join(self.save_to_folder, 'FullFeatureSpaceResults.json'), 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=4)

    def _proximity_matrix(self, 
                          model: RandomForestClassifier, 
                          X: np.ndarray, 
                          normalize: bool=True, 
                          plot: bool=True) -> np.ndarray:      

        terminals = model.apply(X)

        a = terminals[:,0]
        proxMat = 1*np.equal.outer(a, a)

        for i in range(1, self.n_estimators):
            a = terminals[:,i]
            proxMat += 1*np.equal.outer(a, a)

        if normalize:
            proxMat = proxMat / self.n_estimators
       
        if plot:
            plt.figure(figsize=(16,9))
            plt.imshow(proxMat, cmap='viridis', interpolation='nearest')
            plt.colorbar()
            plt.xlabel('Sample number x')
            plt.ylabel('Sample number y')
            plt.title(f'Random Forest Proximity matrix.')
            plt.savefig(fname=os.path.join(self.save_to_folder, 'RandomForestProximityMatrix'))
        
        return proxMat   
    
    def _get_manifold_dimension(self, X: np.ndarray, dimension: int = 20) -> np.ndarray:

        manifold = MDS(n_components=dimension)
        projection = manifold.fit_transform(X)

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        X_sc = StandardScaler().fit_transform(projection)
        pcs = PCA(n_components=dimension).fit(X_sc)

        # shuffle for baseline
        X_sc_permuted = X_sc.copy()
        for i in range(dimension):
            X_sc_permuted[:,i] = np.random.permutation(X_sc_permuted[:,i])
        pcs_permuted = PCA(n_components=dimension).fit(X_sc_permuted)

        self._manifold_dimension = np.argmax(pcs_permuted.explained_variance_ratio_ > pcs.explained_variance_ratio_)

        plt.figure(figsize=(16,9))
        plt.plot(list(range(dimension)), pcs.explained_variance_ratio_*100, marker='o')
        plt.plot(list(range(dimension)), pcs_permuted.explained_variance_ratio_*100, marker='x', c='k')
        plt.xlabel('Principal component number')
        plt.ylabel('% variance explained')
        plt.legend(['MDS Features', 'Shuffled Dataset'])
        plt.title(f'Determining number of manifold dimensions using PCA and a dummy dataset')
        plt.savefig(fname=os.path.join(self.save_to_folder, 'MDSFeaturesPCA'))

    def _generate_manifold(self, X: np.ndarray) -> np.ndarray:

        manifold = MDS(n_components=self._manifold_dimension)
        return manifold.fit_transform(X)
    
    def _train_forward_mapping(self, X: np.ndarray) -> None:

        self.forward_models = {
            i:RandomForestRegressor(self.n_estimators).fit(X, self._feature_set[:,i]) for i in range(self._feature_set.shape[1])
        }

    def _train_reverse_mapping(self, X: np.ndarray) -> None:

        self.reverse_models = {
            i:RandomForestRegressor(self.n_estimators).fit(self._feature_set, X[:,i]) for i in range(X.shape[1])
        }

if __name__ == "__main__":
    
    from processmonitoring.datasets import MovingSineWave

    dataset = MovingSineWave.MovingSineWaveGenerator(1000, 0.5, 'iaaft', 100)

    model = RandomForestFeatures(dataset, 100)
    model.train()






