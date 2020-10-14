#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Package PyCTLib
##############################
__all__ = """
    touch
""".split()

import sys
from pyctlib.basics.override import *

@override
def touch(f: Callable):
    try: return f()
    except: return None

@touch
def _(s: str):
    frame = sys._getframe()
    while "pyctlib" in str(frame.f_code): frame = frame.f_back
    local_vars = frame.f_locals
    local_vars.update(locals())
    locals().update(local_vars)
    try: return eval(s)
    except: return None
