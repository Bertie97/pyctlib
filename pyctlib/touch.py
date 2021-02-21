#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package <main>
##############################
__all__ = """
    touch
    get
""".split()

import sys
from pyoverload import *

class get_environ_vars(dict):
    """
    get_environ_vars(pivot) -> dict

    Returns a list of dictionaries containing the environment variables, 
        i.e. the variables defined in the most reasonable user environments. 
    
    Note:
        It search for the environment where the pivot is defined. 
        Please do not use it abusively as it is currently provided for private use in project PyCTLib only. 

    Example::
        In file `main.py`:
            from mod import function
            def pivot(): ...
            function(pivot)
        In file `mod.py`:
            from pyoverload.utils import get_environ_vars
            def function(f): return get_environ_vars(f)
        Output:
            {
                'function': <function 'function'>,
                'pivot': <function 'pivot'>,
                '__name__': "__main__",
                ...
            }
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

@overload
def touch(f: Callable, default=None):
    try: return f()
    except: return default

@overload
def touch(s: str, default=None):
    local_vars = get_environ_vars(touch)
    local_vars.update(locals())
    locals().update(local_vars.simplify())
    try: return eval(s)
    except: return default

@overload
def get__default__(var, value):
    if var is None: return value
    else: return var

@overload
def get(f: Callable, v):
    return get(touch(f), v)
