#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Package PyCTLib
##############################

try:
    import torch
    import numpy as np
except ImportError:
    raise ImportError("'pyctlib.mic' cannot be used without dependency 'torch' and 'numpy'.")

