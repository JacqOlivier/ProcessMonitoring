from processmonitoring.datasets import GenericDataset
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import json

class FeatureExtractor:

    def __init__(self, 
                 dataset: GenericDataset.DatasetWithPermutations, 
                 save_to_folder: str = None) -> None:
        
        self.dataset = dataset
        self.save_to_folder = save_to_folder

    def train():
        """For model pretraining on X_tr (NOC) data.
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
    
    def feature_space_svm(self) -> None:

        from sklearn.svm import OneClassSVM
        """
        if self._feature_set.shape[1] >= 2:
            self._OCSVM = OneClassSVM(gamma='auto').fit(self._feature_set[:,:2])
        else:
            self._OCSVM = OneClassSVM(gamma='auto').fit(self._feature_set[:,:2])
        """
        if not isinstance(self._projected_features, np.ndarray):
            X = np.concatenate([self.dataset.X_tr, self.dataset.X_val, self.dataset.X_test])
            y = np.concatenate([self.dataset.y_tr, self.dataset.y_val, self.dataset.y_test])

            self._projected_features = np.zeros((X.shape[0], self._feature_set.shape[1]))
            for i in range(X.shape[0]):
                self._projected_features[i,:] = self.eval_in_feature_space(X[i,:], ocsvm=False)

        OCSVM = OneClassSVM(gamma='auto').fit(self._projected_features[:,:2])

        # some ints
        tr = self.dataset.X_tr.shape[0]
        val = self.dataset.X_tr.shape[0] + self.dataset.X_val.shape[0]

        y_pred_train = OCSVM.predict(self._projected_features[:tr, :2])
        y_pred_val = OCSVM.predict(self._projected_features[tr:val, :2])
        y_pred_fault = OCSVM.predict(self._projected_features[val:, :2])
        n_error_train = y_pred_train[y_pred_train == -1].size
        n_error_test = y_pred_val[y_pred_val == -1].size
        n_error_outliers = y_pred_fault[y_pred_fault == 1].size

        xx, yy = np.meshgrid(np.linspace(np.min(self._projected_features), np.max(self._projected_features), 500), 
                             np.linspace(np.min(self._projected_features), np.max(self._projected_features), 500))

        # plot the line, the points, and the nearest vectors to the plane
        Z = OCSVM.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(16, 9))
        plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
        a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors="darkred")
        plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors="palevioletred")

        s = 40
        b1 = plt.scatter(self._projected_features[:tr,0], self._projected_features[:tr, 1], c="white", s=s, edgecolors="k")
        b2 = plt.scatter(self._projected_features[tr:val,0], self._projected_features[tr:val,1], c="blueviolet", s=s, edgecolors="k")
        c = plt.scatter(self._projected_features[val:,0], self._projected_features[val:,1], c="gold", s=s, edgecolors="k")
        plt.axis("tight")
        plt.xlim((np.min(self._projected_features), np.max(self._projected_features)))
        plt.ylim((np.min(self._projected_features), np.max(self._projected_features)))
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
        OCSVM = OneClassSVM(gamma='auto').fit(self._projected_features)
        y_pred_train = OCSVM.predict(self._projected_features[:tr, :])
        y_pred_val = OCSVM.predict(self._projected_features[tr:val, :])
        y_pred_fault = OCSVM.predict(self._projected_features[val:, :])
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

        with open(os.path.join(self.save_to_folder, 'FullFeatureSpaceResults.json'), 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=4)