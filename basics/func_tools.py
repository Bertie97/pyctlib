#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Package PyCTLib
##############################
import sys

__all__ = []

_mid = lambda x: x[1] if len(x) > 1 else x[0]
_rawname = lambda s: _mid(str(s).split("'"))

def raw_function(func):
    if "__func__" in dir(func):
        return func.__func__
    return func

def _get_wrapped(f):
    while '__wrapped__' in dir(f): f = f.__wrapped__
    return f

class get_environ_vars(dict):
    """
    get_environ_vars(pivot) -> dict

    Returns a list of dictionaries containing the environment variables, 
        i.e. the variables defined in the most reasonable user environments. 
    
    Note:
        It search for the all function in the function stack that is not a function in this package or a built in package. 
        Please do not use it abusively as it is currently provided for private use in package pyctlib only. 

    Example::

        >>> isatype(np.array)
        False
        >>> isatype(np.ndarray)
        True
        >>> isatype(None)
        True
        >>> isatype([int, np.ndarray])
        True
    """

    def __new__(cls, pivot):
        self = super().__new__(cls)
        frame = sys._getframe()
        self.all_vars = []
        filename = raw_function(_get_wrapped(pivot)).__globals__.get('__file__', '')
        while frame.f_back is not None:
            frame = frame.f_back
            if not filename: continue
            if _rawname(frame).startswith('<') and _rawname(frame).endswith('>'): continue
            if _rawname(frame) != filename: continue
            self.all_vars.extend([frame.f_locals, frame.f_globals])
            break
        else:
            self.all_vars.extend([frame.f_locals, frame.f_globals])
        return self

    def __init__(self, _): pass
    
    def __getitem__(self, k):
        for varset in self.all_vars:
            if k in varset: return varset[k]; break
        else: raise IndexError(f"No '{k}' found in the environment. ")

    def __setitem__(self, k, v):
        for varset in self.all_vars:
            if k in varset: varset[k] = v; break
        else: self.all_vars[0][k] = v

    def __contains__(self, x):
        for varset in self.all_vars:
            if x in varset: break
        else: return False
        return True

    def update(self, x): self.all_vars.insert(0, x)

    def simplify(self):
        collector = {}
        for varset in self.all_vars[::-1]: collector.update(varset)
        return collector