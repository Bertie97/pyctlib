#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package visual
##############################

__all__ = """
""".split()

try: from matplotlib import pyplot as plt
except ImportError:
    raise ImportError("'pyctlib.watch.debugger' cannot be used without dependency 'matplotlib'. ")

