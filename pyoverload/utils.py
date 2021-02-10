#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package pyoverload
##############################
__all__ = """
    raw_function
    return_type_wrapper
    decorator
""".split()

import sys
from functools import wraps

_mid = lambda x: x[1] if len(x) > 1 else x[0]
_rawname = lambda s: _mid(str(s).split("'"))

def raw_function(func):
    if hasattr(func, "__func__"):
        return func.__func__
    return func

def _get_wrapped(f):
    while hasattr(f, '__wrapped__'): f = f.__wrapped__
    return f

def decorator(wrapper_func):
    if not callable(wrapper_func): raise TypeError("@decorator wrapping a non-wrapper")
    def wrapper(*args, **kwargs):
        if not kwargs and len(args) == 1:
            func = args[0]
            raw_func = raw_function(func)
            if callable(raw_func):
                func_name = f"{raw_func.__name__}[{wrapper_func.__qualname__.split('.')[0]}]"
                wrapped_func = wraps(raw_func)(wrapper_func(raw_func))
                wrapped_func.__name__ = func_name
                wrapped_func.__doc__ = raw_func.__doc__
                # return wrapped_func
                if 'staticmethod' in str(type(func)): trans = staticmethod
                elif 'classmethod' in str(type(func)): trans = classmethod
                else: trans = lambda x: x
                return trans(wrapped_func)
        return decorator(wrapper_func(*args, **kwargs))
    return wraps(wrapper_func)(wrapper)

# def decorator(*wrapper_func, use_raw = True):
#     if len(wrapper_func) > 2: raise TypeError("Too many arguments for @decorator")
#     elif len(wrapper_func) == 1: wrapper_func = wrapper_func[0]
#     else: return decorator(lambda x: decorator(x, use_raw = use_raw), use_raw = use_raw)
#     if not isinstance(wrapper_func, type(decorator)): raise TypeError("@decorator wrapping a non-wrapper")
#     def wrapper(*args, **kwargs):
#         if not kwargs and len(args) == 1:
#             func = args[0]
#             raw_func = raw_function(func)
#             if isinstance(raw_func, type(decorator)):
#                 func_name = f"{raw_func.__name__}[{wrapper_func.__qualname__.split('.')[0]}]"
#                 wrapped_func = wraps(raw_func)(wrapper_func(raw_func if use_raw else func))
#                 wrapped_func.__name__ = func_name
#                 wrapped_func.__doc__ = raw_func.__doc__
#                 return wrapped_func
#                 if 'staticmethod' in str(type(func)): trans = staticmethod
#                 elif 'classmethod' in str(type(func)): trans = classmethod
#                 else: trans = lambda x: x
#                 return trans(wrapped_func)
#         return decorator(wrapper_func(*args, **kwargs))
#     return wraps(wrapper_func)(wrapper)

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

    def __new__(cls):
        self = super().__new__(cls)
        frame = sys._getframe()
        self.all_vars = []
        # filename = raw_function(_get_wrapped(pivot)).__globals__.get('__file__', '')
        prev_frame = frame
        prev_frame_file = _rawname(frame)
        while frame.f_back is not None:
            frame = frame.f_back
            frame_file = _rawname(frame)
            if frame_file.startswith('<') and frame_file.endswith('>') and frame_file != '<stdin>': continue
            if '<module>' not in str(frame):
                if frame_file != prev_frame_file:
                    prev_frame = frame
                    prev_frame_file = frame_file
                continue
            if frame_file != prev_frame_file: self.all_vars.extend([frame.f_locals])
            else: self.all_vars.extend([prev_frame.f_locals])
            break
        else: raise TypeError("Unexpected function stack, please contact the developer for further information. ")
        return self

    def __init__(self): pass
    
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
