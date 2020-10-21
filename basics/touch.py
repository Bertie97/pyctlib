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
from pyctlib.basics.func_tools import get_environ_vars

@overload
def touch(f: Callable):
    try: return f()
    except: return None

@overload
def touch(s: str):
    local_vars = get_environ_vars(touch)
    local_vars.update(locals())
    locals().update(local_vars.simplify())
    try: return eval(s)
    except: return None

@overload
def get__default__(var, value):
    if var is None: return value
    else: return var

@overload
def get(f: Callable, v):
    return get(touch(f), v)
