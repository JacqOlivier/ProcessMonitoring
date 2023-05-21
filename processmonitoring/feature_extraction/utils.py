__MODEL_DICT__ = {}

def register_function(name):

    def register_function_fn(cls):
        if name in __MODEL_DICT__:
            raise ValueError(f'Name {name} already registered.')
        __MODEL_DICT__[name] = cls
        return cls
    
    return register_function_fn

def feature_factory(model_type):
    return __MODEL_DICT__[model_type]

__all__ = ['feature_factory', 'register_function']