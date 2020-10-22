#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package micomputing
##############################

__all__ = """
""".split()

import numpy as np
import torch

try: from matplotlib import pyplot as plt
except ImportError:
    raise ImportError("'pyctlib.mic.plot' cannot be used without dependency 'matplotlib'. ")
import torchplus as tp
from matplotlib.pyplot import *
from pyoverload import *


@params
def imshow(data: Array, **kwargs):
    "Show transverse medical image with the right hand side of the subject shown on the left and anterior shown at the bottom. "
    if 'cmap' not in kwargs: kwargs['cmap'] = plt.cm.gray
    data = tp.Tensor(data).squeeze()
    if data.dim() > 2 and data.batch_dimension: data = data.sample(number=1, random=False)
    if data.dim() > 2: raise TypeError("'plot.imshow' takes 2D-data as input, please reduce the dimension manually or specify a batch dimension to reduce. ")
    if data.dim() <= 1: data = data.unsqueeze(0)
    if data.dim() == 0: raise TypeError("Please don't use 'plot.imshow' to demonstrate a scalar. ")
    return plt.imshow(data.numpy(), **kwargs)

@params
def sliceshow(data: Array, dim=-1: int, **kwargs):
    data = tp.Tensor(data)
    sample_indices = [slice(None)] * data.dim()
    sample_indices[dim] = data.shape[dim] // 2
    return imshow(data[sample_indices], **kwargs)

@overload
def maskshow(*masks: Array, on=None: [null, Array], **kwargs):
    masks = tuple(tp.Tensor(m) for m in masks)
    if on is None: on = tp.ones_like(tp.Tensor(args[0]))
    on = tp.Tensor(on)
    if on.shape[0] == 3 and on.dim() > 2 and on.batch_dimension != 0:
        masks = tuple(m.expand(on.shape[1:]) for m in masks)
    else: masks = tuple(m.expand_as(on) for m in masks)

    colormap = kwargs.get("cmap", plt.cm.hsv)
    colors = []; step = 1/3
    for i in range(len(masks)):
        if i == 0: colors.append(0); continue
        new_color = colors[-1]
        while new_color in colors:
            new_color += step
            if new_color >= 1: step /= 2; new_color -= 1
        colors.append(new_color)
    
    
    
    return imshow(, **kwargs)

@overload
def maskshow(*args: Array, **kwargs):
    ci = plt.gci()
    if ci is None: return maskshow(*args, on=tp.ones_like(tp.Tensor(args[0])), **kwargs)
    return maskshow(*args, on=ci.get_array().data, **kwargs)
