from .vector import vector
from .table import table
from .sequence import sequence
from .wrapper import flexible_wrapper
from functools import wraps

def advanced_data(func, **hyper):

    @wraps(func)
    def wrapper(*args, **kwargs):
        args = sequence(args).map_where(lambda x: isinstance(list), lambda x: vector(x),
                lambda x: isinstance(tuple), lambda x: sequence(x),
                lambda x: isinstance(dict), lambda x: table(x), None)
        return func(*args, **kwargs)
    return wrapper
