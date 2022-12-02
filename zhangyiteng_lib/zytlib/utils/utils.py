from types import GeneratorType, MethodWrapperType
import types
import functools

def iterable(x) -> bool:
    """
    iterable(x) -> bool

    Returns whether an instance can be iterated. Strings are excluded. 

    Args:
        x (any): the input variable.

    Example::

        >>> iterable(x for x in range(4))
        True
        >>> iterable({2, 3})
        True
        >>> iterable("12")
        False
    """
    if isinstance(x, str):
        return False
    if isinstance(x, type):
        return False
    if callable(x):
        return False
    if isinstance(x, GeneratorType):
        return True
    return hasattr(x, '__iter__') and hasattr(x, '__len__')

def totuple(x, depth=1):
    if isinstance(x, types.GeneratorType):
        x = tuple(x)
    if not iterable(x):
        x = (x, )
    if depth == 1:
        if iterable(x) and len(x) == 1 and iterable(x[0]):
            return tuple(x[0])
        if iterable(x) and len(x) == 1 and isinstance(x[0], types.GeneratorType):
            return tuple(x[0])
        else:
            return tuple(x)
    if depth == 0:
        return tuple(x)
    temp = [totuple(t, depth=depth-1) for t in x]
    return functools.reduce(lambda x, y: x+y, temp, initial=tuple())

class constant(object):

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            super().__setattr__(key, value)

    def __setattr__(self, name, value):
        if hasattr(self, name):
            if self.__getattribute__(name) == value:
                return
            else:
                raise RuntimeError("not consitant value assignment for constant")
        super().__setattr__(name, value)

    def __str__(self):

        keys = [x for x in dir(self) if not x.startswith("_")]
        content = ["{}={}".format(key, self.__getattribute__(key)) for key in keys]
        ret = "constant({})".format(", ".join(content))
        return ret

    def __repr__(self):
        return str(self)

def str_type(x) -> str:
    return str(type(x)).split("'")[1]
