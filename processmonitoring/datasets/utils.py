from processmonitoring.datasets.dataset import DatasetWithPermutations

__MODEL_DICT__ = {}

def register_function(name):

    def register_function_fn(cls):
        if name in __MODEL_DICT__:
            raise ValueError(f'Name {name} already registered.')
        if not issubclass(cls, DatasetWithPermutations):
            raise ValueError(f'Class {cls} is not a subclass of DatasetWithPermutations.')
        __MODEL_DICT__[name] = cls
        return cls
    
    return register_function_fn

def dataset_factory(model_type):
    return __MODEL_DICT__[model_type]

__all__ = ['dataset_factory']