#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Package PyCTLib
##############################
import sys

def get_environ_vars():
    frame = sys._getframe()
    while "pyctlib" in str(frame.f_code) or 'file "<' in str(frame.f_code): frame = frame.f_back
    all_vars = frame.f_globals.copy()
    all_vars.update(frame.f_locals)
    frame.f_locals.update(all_vars)
    return frame.f_locals