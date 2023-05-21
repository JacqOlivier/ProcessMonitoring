from processmonitoring.feature_extraction.GenericFeatureExtractor import FeatureExtractor

__MODEL_DICT__ = {}

def register_function(name):

    def register_function_fn(cls):
        if name in __MODEL_DICT__:
            raise ValueError(f'Name {name} already registered.')
        if not issubclass(cls, FeatureExtractor):
            raise ValueError(f'Class {cls} is not a subclass of FeatureExtractor.')
        __MODEL_DICT__[name] = cls
        return cls
    
    return register_function_fn

def feature_factory(model_type):
    return __MODEL_DICT__[model_type]

__all__ = ['feature_factory']