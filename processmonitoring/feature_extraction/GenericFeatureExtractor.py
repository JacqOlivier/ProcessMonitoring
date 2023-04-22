from processmonitoring.datasets import GenericDataset

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