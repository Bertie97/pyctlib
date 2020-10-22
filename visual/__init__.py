#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package visual
##############################

from pyctlib.basics.touch import *

if touch(lambda: __import__(line_profile)):
    from pyctlib.watch import debugger
if touch(lambda: __import__(matplotlib)):
    from pyctlib.watch import plot as plt
if touch(lambda: __import__(tqdm)):
    from pyctlib.watch import progress
