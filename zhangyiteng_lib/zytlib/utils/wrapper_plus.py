from .vector import vector
from .table import table
from .sequence import sequence
from .wrapper import flexible_wrapper
from functools import wraps

def advanced_data(func, **hyper):

    @wraps(func)
    def wrapper(*args, **kwargs):
        args = sequence(args).map_where(lambda x: isinstance(x, list), vector,
                lambda x: isinstance(x, tuple), sequence,
                lambda x: isinstance(x, dict), table, None)
        return func(*args, **kwargs)
    return wrapper
