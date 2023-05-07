from processmonitoring.datasets import GenericDataset
from processmonitoring.feature_extraction import GenericFeatureExtractor

import os
import numpy as np
from scipy.spatial.distance import squareform, pdist
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import tensorflow as tf

class CNN(GenericFeatureExtractor.FeatureExtractor):

    def __init__(self, 
                 dataset: GenericDataset.DatasetWithPermutations,
                 feature_config: dict, 
                 save_to_folder: str = None) -> None:
        
        super().__init__(dataset, save_to_folder)

        self._feature_config = feature_config

    def train(self, mode):
        """For model pretraining on X_tr (NOC) data.
        """
        # generate train datasets
        self.X_train = np.concatenate([self.dataset.X_tr, self.dataset.X_tr_permuted], axis=0)
        self.y_train = np.concatenate([self.dataset.y_tr, self.dataset.y_tr_permuted])
        
        # generate images and plot 2
        self.X_train_images = generate_distance_plots(self.X_train)
        plot_two_distance_plots(
            self.X_train[np.random.randint(0, self.dataset.X_tr.shape[0])],
            self.X_train[np.random.randint(self.dataset.X_tr.shape[0], self.X_train.shape[0])], 
            self.save_to_folder)
        
        # initiate model
        self._generate_model()
        self._pretrain()
        self._finetune()

        # extract features
        self.X = np.concatenate([self.dataset.X_tr, self.dataset.X_val, self.dataset.X_test])
        self.y = np.concatenate([self.dataset.y_tr, self.dataset.y_val, self.dataset.y_test])
        self.X_images = generate_distance_plots(self.X)
        self._extract_features()
    
    def eval_in_feature_space(self):
        """Evaluates a single window of data as a test case from client (SPE) code.
        """
        raise NotImplementedError()
    
    def eval_in_residual_space(self):
        """Evaluates a single window of data as a test case from client (SPE) code.
        """
        raise NotImplementedError()
    
    def _generate_model(self):

        base_model = tf.keras.applications.VGG19(weights='imagenet', 
                                                 include_top=False, 
                                                 input_shape=(self.dataset.window_length, self.dataset.window_length, 3))

        for layer in base_model.layers:
            layer.trainable = False

        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        output = tf.keras.layers.Dense(2, activation='softmax')(x)

        self.model = tf.keras.models.Model(inputs=base_model.inputs, outputs=output)

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    def _pretrain(self) -> None:

        one_hot = tf.keras.utils.to_categorical(self.y_train)
        
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.3)
    
        datagen.fit(self.X_train_images)
    
        self.model.fit_generator(datagen.flow(self.X_train_images, one_hot, batch_size=10, subset='training'), 
            validation_data=datagen.flow(self.X_train_images, one_hot, batch_size=10, subset='validation'), 
            steps_per_epoch = int(0.7*self.X_train_images.shape[0] / 10), epochs=self._feature_config['num_epochs_pretrain'], verbose=1, 
            validation_steps=int(0.3*self.X_train_images.shape[0] / 10))
        
    def _finetune(self):

        for layer in self.model.layers[-8:]:
            layer.trainable = True

        one_hot = tf.keras.utils.to_categorical(self.y_train)

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.3)
    
        datagen.fit(self.X_train_images)
    
        self.model.fit_generator(datagen.flow(self.X_train_images, one_hot, batch_size=10, subset='training'), 
            validation_data=datagen.flow(self.X_train_images, one_hot, batch_size=10, subset='validation'), 
            steps_per_epoch = int(0.7*self.X_train_images.shape[0] / 10), epochs=self._feature_config['num_epochs_finetune'], verbose=1, 
            validation_steps=int(0.3*self.X_train_images.shape[0] / 10))
        
    def _extract_features(self) -> None:

        # remove classification layers
        self.model = tf.keras.models.Model(inputs=self.model.input, outputs=self.model.layers[-3].output)
        self.model.summary()

        # need to generate uimages here next
        self._projected_features = self.model.predict(self.X_images)

        # get a 2D TSNE representation and plot
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE   

        # scale all according to X_tr feature range
        scaler = StandardScaler().fit(self._projected_features[:self.dataset.X_tr.shape[0],:])
        projected_features_scaled = scaler.transform(self._projected_features)

        # fit pca model to reduce dimensions to 50
        pca_model = PCA(n_components=50).fit(projected_features_scaled[:self.dataset.X_tr.shape[0],:])
        pca_features = pca_model.transform(projected_features_scaled)

        # finally, fit TSNE model
        tsne_features = TSNE().fit_transform(pca_features)

        tr = self.dataset.X_tr.shape[0]
        val = self.dataset.X_tr.shape[0] + self.dataset.X_val.shape[0]

        plt.figure(figsize=(16, 9))
        plt.scatter(tsne_features[:tr,0], tsne_features[:tr,1], c='blue', marker='o')
        plt.scatter(tsne_features[tr:val,0], tsne_features[tr:val,1], c='green', marker='x')
        plt.scatter(tsne_features[val:,0], tsne_features[val:,1], c='red', marker='*')
        plt.xlabel(f'1st Dimension')
        plt.ylabel(f'2nd Dimension')
        plt.legend(['Train', 'Validation', 'Fault'])
        plt.title(f'2D TSNE plot - First reduced to 50 dimensions using PCA')
        plt.grid(True)
        plt.savefig(fname=os.path.join(self.save_to_folder, 'CNNFeaturesWIthFault'))
    
__all__ = ['CNN']

@staticmethod
def generate_distance_plots(X):

    distance_plots = []
    for i in range(X.shape[0]):
        dist = dist_matrix(X[i,:])
        distance_plots.append(distance_plot(dist))
    
    distance_plots = np.asarray(distance_plots)

    # preprocess with model and return
    return tf.keras.applications.vgg19.preprocess_input(distance_plots)
  
@staticmethod
def dist_matrix(window):

    dist = squareform(pdist(np.stack([window, window]).T))
        
    return dist

@staticmethod
def distance_plot(distance, minimum=None, maximum=None):
    
    norm = plt.Normalize()
    cmap = cm.viridis
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    distance_plot = m.to_rgba(distance)

    # remove alpha column
    distance_plot = np.delete(distance_plot, 3, axis=2)

    return (distance_plot*255).astype(int)

@staticmethod
def plot_two_distance_plots(X0: np.ndarray, X1: np.ndarray, save_to_folder=None) -> None:

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    dist1 = dist_matrix(X0)
    dist2 = dist_matrix(X1)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))

    ax = axes.flat[0]
    im = ax.imshow(dist1, origin='lower', vmin=dist1.min(), vmax=dist1.max())
    ax.set_title('Real', weight='bold', fontsize=16, y=-0.2)
    ax.tick_params(axis='both', which='major', labelsize=14)

    ax_2 = axes.flat[1]
    im2 = ax_2.imshow(dist2, origin='lower', vmin=dist1.min(), vmax=dist1.max())
    ax_2.set_title('Permuted', weight='bold', fontsize=16, y=-0.2)
    ax_2.tick_params(axis='both', which='major', labelsize=14)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.37)
    cbar.ax.tick_params(labelsize=14) 

    if save_to_folder:
        fig.savefig(fname=os.path.join(save_to_folder, 'ModelTrainingDistancePlots'))