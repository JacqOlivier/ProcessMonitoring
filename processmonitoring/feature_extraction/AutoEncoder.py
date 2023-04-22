from processmonitoring.feature_extraction import GenericFeatureExtractor
from processmonitoring.datasets import GenericDataset

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import os

class AutoEncoderFeatures(GenericFeatureExtractor.FeatureExtractor):

    def __init__(self, 
                 dataset: GenericDataset.DatasetWithPermutations, 
                 feature_config: dict,
                 save_to_folder: str = None) -> None:
        
        super(AutoEncoderFeatures, self).__init__(dataset=dataset, 
                                                  save_to_folder=save_to_folder)
        self._config = feature_config

    def train(self, mode: dict):

        self._get_latent_size()
        self.model = AE(self._hidden_layer_size,
                        self.dataset)
        self._train_autoencoder()

        if mode['name'] == 'ExtractFeatures':
            self._extract_features()
        
    def eval_in_feature_space(self, sample: np.ndarray) -> float:
        return self.model.encode(sample)

    def eval_in_residual_space():
        pass

    def _train_autoencoder(self, 
                          num_epochs: int = 100,
                          batch_size: int = 32,
                          train_val_ratio: float = 0.8, 
                          learning_rate: float = 0.001, 
                          max_epochs_without_improvement: int = 5):
        """Trains an autoencoder using the given dataset.

        Args:
            model (nn.Module): The autoencoder model to train.
            dataset (torch.utils.data.Dataset): The dataset to train on.
            num_epochs (int, optional): The number of epochs to train for. Defaults to 10.
            batch_size (int, optional): The batch size to use for training. Defaults to 32.
            train_val_ratio (float, optional): The ratio of training data to validation data.
                Must be between 0 and 1. Defaults to 0.8.
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.001.
        """
        #Convert the numpy arrays to PyTorch tensors
        X_tensor = torch.from_numpy(self.dataset.X_tr).float()
        y_tensor = torch.from_numpy(self.dataset.X_tr).float()

        # Create a PyTorch dataset object from the tensors
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        # Set up the optimizer and loss function
        criterion = nn.MSELoss()
        # Using an Adam Optimizer with lr = 0.1
        optimizer = optim.Adam(self.model.parameters(),
                               lr = learning_rate,
                               weight_decay = 1e-8)

        # Loop over the epochs
        for epoch in range(self._config['num_epochs']):
            # Split the dataset into training and validation sets for this epoch
            train_size = int(len(dataset) * train_val_ratio)
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            # Create the dataloaders for training and validation
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

            # Set the model to training mode
            self.model.train()

            # Loop over the training data
            train_loss = 0.0
            best_val_loss = float('inf')
            epochs_without_improvement = 0
            for data in train_dataloader:
                inputs, _ = data
                # Zero the gradients
                optimizer.zero_grad()
                # Forward pass
                outputs = self.model(inputs)
                # Compute the loss
                loss = criterion(outputs, inputs)
                # Backward pass
                loss.backward()
                # Update the weights
                optimizer.step()
                # Update the training loss
                train_loss += loss.item() * inputs.size(0)

            # Calculate the average training loss for this epoch
            train_loss /= len(train_dataloader.dataset)

            # Set the model to evaluation mode
            self.model.eval()

            # Turn off gradient computation for validation
            with torch.no_grad():
                # Loop over the validation data
                val_loss = 0.0
                for data in val_dataloader:
                    inputs, _ = data
                    # Forward pass
                    outputs = self.model(inputs)
                    # Compute the loss
                    loss = criterion(outputs, inputs)
                    # Update the validation loss
                    val_loss += loss.item() * inputs.size(0)
                # Calculate the average validation loss for this epoch
                val_loss /= len(val_dataloader.dataset)

            # Print the training and validation loss for this epoch
            print(f'Epoch {epoch+1}/{num_epochs}: Average Train loss: {train_loss}: Validation loss: {val_loss}')

            # Check for improvement in validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Check if we should stop training
            if epochs_without_improvement == max_epochs_without_improvement:
                print(f'Stopping training after {epoch+1} epochs due to no improvement in validation loss.')
                break

    def _get_latent_size(self, variance_retained: float=0.95):

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        X_sc = StandardScaler().fit_transform(self.dataset.X_tr)
        pca_obj = PCA().fit(X_sc)

        self._hidden_layer_size = np.argmax(np.where(np.cumsum(pca_obj.explained_variance_ratio_) >= variance_retained, 1, 0)) + 1

        plt.figure(figsize=(16,9))
        plt.plot(list(range(pca_obj.n_components_)), np.cumsum(pca_obj.explained_variance_ratio_*100), marker='o')
        plt.xlabel('Principal component number')
        plt.ylabel('% variance explained')
        plt.grid(True)
        plt.title(f'{self._hidden_layer_size + 1} selected for hidden layer suze')
        plt.savefig(fname=os.path.join(self.save_to_folder, 'PCAForHiddenSize'))

    def _extract_features(self) -> None:

        X = np.concatenate([self.dataset.X_tr, self.dataset.X_val, self.dataset.X_test])
        y = np.concatenate([self.dataset.y_tr, self.dataset.y_val, self.dataset.y_test])

        self._projected_features = np.zeros((X.shape[0], self._hidden_layer_size))
        for i in range(X.shape[0]):
            self._projected_features[i,:] = self.eval_in_feature_space(X[i,:]).detach().numpy()

        # find two columns with highest variance
        from sklearn.preprocessing import StandardScaler
        scaled_features = StandardScaler.fit_transform(self._projected_features)

        #idx = (-arr).argsort()[:n]
        # TODO: contimue from here.

        plt.figure(figsize=(16, 9))
        plt.scatter(self._projected_features[:,0][y==0], self._projected_features[:,1][y==0], c='blue', marker='o')
        plt.scatter(self._projected_features[:,0][y==2], self._projected_features[:,1][y==2], c='green', marker='x')
        plt.scatter(self._projected_features[:,0][y==1], self._projected_features[:,1][y==1], c='red', marker='*')
        plt.xlabel('1st MDS component')
        plt.ylabel('2nd MDS component')
        plt.legend(['Train', 'Validation', 'Fault'])
        plt.title(f'2D MDS coordinates in {self._manifold_dimension}-dimensional space.')
        plt.grid(True)
        plt.savefig(fname=os.path.join(self.save_to_folder, 'RandomForestFeaturesWithFault'))

class AE(nn.Module):

    def __init__(self, 
                 hidden_layer_size: int, 
                 dataset: GenericDataset.DatasetWithPermutations,
                 mid_layer_size: int = None) -> None:
        super().__init__()
        self._hidden_layer_size = hidden_layer_size
        if mid_layer_size:
            self._mid_layer_size = mid_layer_size
        else:
            self._mid_layer_size = self._hidden_layer_size*2
        self.dataset = dataset

        # build encoder architecture
        self._encoder = nn.Sequential(
            nn.Linear(in_features=self.dataset.window_length, out_features=self._hidden_layer_size*2), 
            nn.ReLU(),
            nn.Linear(in_features=self._hidden_layer_size*2, out_features=self._hidden_layer_size)
        )

        self._decoder = nn.Sequential(
            nn.Linear(in_features=self._hidden_layer_size, out_features=self._hidden_layer_size*2), 
            nn.ReLU(),
            nn.Linear(in_features=self._hidden_layer_size*2, out_features=self.dataset.window_length)
        )

        self._loss_function = nn.MSELoss()

    def forward(self, x):
        x = self._encoder(x)
        return self._decoder(x)
    
    def encode(self, x):
        return self._encoder(torch.from_numpy(x).float())


