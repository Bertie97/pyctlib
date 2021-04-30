#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package visual
##############################

# from pyctlib.basics.touch import *
from ..touch import touch

if touch(lambda: __import__("line_profiler")):
    # from pyctlib.watch import debugger
    from .debugger import *
if touch(lambda: __import__("matplotlib")):
    # from pyctlib.watch import plot as plt
    from .plot import *
if touch(lambda: __import__("tqdm")):
    # from pyctlib.watch import progress
    from .progress import *
