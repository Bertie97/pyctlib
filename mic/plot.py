#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Package PyCTLib
##############################
__all__ = """
""".split()

import torch
import numpy as np
try: from matplotlib import pyplot as plt
except ImportError:
    raise ImportError("'pyctlib.mic.plot' cannot be used without dependency 'matplotlib'. ")
from matplotlib.pyplot import *
from pyctlib import *
from pyctlib import torchplus as tp

# def imshow(data):
#     "Show transverse medical image with the right hand side of the subject shown on the left and anterior shown at the bottom. "
#     data = tp.Tensor(data)
#     if data.batch_dimension: data = data.sample(data.batch_dimension, shape
#     if data.dim() > 2: data = data
#     return plt.imshow(.squeeze().numpy(), cmap=plt.cm.gray)

# def 
