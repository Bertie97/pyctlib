from inspect import getargspec
from .table import table
from typing import Union, Tuple

__all__ = ["save_args"]

def save_args(values, ignore: Union[None, Tuple, str]=None):
    """
    usage:

    class A:

        def __init__(self, v1, v2, v3):

            save_args(vars())

            pass
    """
    if isinstance(ignore, str):
        ignore = set([ignore])
    elif ignore is None:
        ignore = set()
    else:
        ignore = set(ignore)

    values['self'].hyper = table()
    for i in getargspec(values['self'].__init__).args[1:]:
        if i not in ignore:
            values['self'].hyper[i] = values[i]
