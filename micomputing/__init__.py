#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package micomputing
##############################

try:
    import torch
    import numpy as np
except ImportError:
    raise ImportError("'pyctlib.mic' cannot be used without dependency 'torch' and 'numpy'.")

