from inspect import getargspec

__all__ = ["save_args"]

def save_args(values, ignore=None):
    values['self'].hyper = dict()
    for i in getargspec(values['self'].__init__).args[1:]:
        if ignore is None or i not in ignore:
            values['self'].hyper[i] = values[i]
