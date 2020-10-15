#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Package PyCTLib
##############################
import sys

def get_environ_vars():
    frame = sys._getframe()
    while "pyctlib" in str(frame.f_code): frame = frame.f_back
    return frame.f_locals