#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Package PyCTLib
##############################
__all__ = """
    touch
    get
""".split()

import sys
from pyctlib.basics.override import *

@overload
def touch(f: Callable):
    try: return f()
    except: return None

@overload
def touch(s: str):
    frame = sys._getframe()
    while "pyctlib" in str(frame.f_code): frame = frame.f_back
    local_vars = frame.f_locals
    local_vars.update(locals())
    locals().update(local_vars)
    try: return eval(s)
    except: return None

@overload
def get__default__(var, value):
    if var is None: return value
    else: return var

@overload
def get(f: Callable, v):
    return get(touch(f), v)
