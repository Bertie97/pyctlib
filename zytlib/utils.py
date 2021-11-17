from types import GeneratorType, MethodWrapperType

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
