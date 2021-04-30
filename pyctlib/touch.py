#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package <main>
##############################
__all__ = """
    get_environ_vars
    touch
    crash
    no_print
""".split()

import sys

_mid = lambda x: x[1] if len(x) > 1 else x[0]
_rawname = lambda s: _mid(str(s).split("'"))

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

def touch(v: [callable, str], default=None):
    if isinstance(v, str):
        local_vars = get_environ_vars()
        local_vars.update(locals())
        locals().update(local_vars.simplify())
        try: return eval(v)
        except: return default
    else:
        try: return v()
        except: return default

def crash(func):
    try:
        func()
    except:
        return True
    return False

class _strIO:
    def __init__(self): self._str_ = ''
    def write(self, s): self._str_ += s
    def __str__(self): return self._str_
    def split(self, c=None): return self._str_.split(c)

class NoPrint:
    def __enter__(self):
        self.io = _strIO()
        self.old_io = sys.stdout
        sys.stdout = self.io
    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.old_io

no_print = NoPrint()
