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

def imshow(data, **kwargs):
    "Show transverse medical image with the right hand side of the subject shown on the left and anterior shown at the bottom. "
    if 'cmap' not in kwargs: kwargs['cmap'] = plt.cm.gray
    data = tp.Tensor(data).squeeze()
    if data.dim() > 2 and data.batch_dimension: data = data.sample(number=1, random=False)
    if data.dim() > 2: raise TypeError("'plot.imshow' takes 2D-data as input, please reduce the dimension manually or specify a batch dimension to reduce. ")
    if data.dim() <= 1: data = data.unsqueeze(0)
    if data.dim() == 0: raise TypeError("Please don't use 'plot.imshow' to demonstrate a scalar. ")
    return plt.imshow(data.numpy(), **kwargs)

def sliceshow(data, dim=-1, **kwargs):
    sample_indices = [slice(None)] * data.dim()
    sample_indices[dim] = data.shape[dim] // 2
    return imshow(data[sample_indices], **kwargs)

