#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package micomputing
##############################

__all__ = """
""".split()

try: from matplotlib import pyplot as plt
except ImportError:
    raise ImportError("'pyctlib.mic.plot' cannot be used without dependency 'matplotlib'. ")
import torchplus as tp
from matplotlib.pyplot import *
from pyoverload import *

canvas = None

@params
def imshow(data: Array, nslice: [int, null]=None, dim: int=-1, **kwargs):
    """
    An automatic image display function for all kinds of tensors. 
    The first image in batched images will be selected to be showed. 
    For medical images:
    Displacements with channel dimension identified will be displayed as RGB colored maps.
    If there are no dimension <=3, gray scaled images will be showed. 
    Transverse medical image with the right hand side of the subject shown on the left
        and anterior shown at the bottom will be selected for 3D volumes.
    `nslice` and `dim` are used for 3D volumes only, meaning to show the `nslice` slice of dimension `dim`. 
    """
    global canvas
    has_cmap = True
    if 'cmap' not in kwargs:
        has_cmap = False
        kwargs['cmap'] = plt.cm.gray
    data = tp.Tensor(data).squeeze()
    if data.ndim <= 1: raise TypeError("Please don't use 'plot.imshow' to demonstrate an array or a scalar. ")
    if data.nspace > 3: raise TypeError("'plot.imshow' takes 2 or 3D-data as input, please reduce the dimension manually or specify special dimensions to reduce. ")
    if data.nspace == 3:
        if data.has_batch: data = data.sample(random=False, dim=[])
        if data.has_channel: data = data.sample(random=False, dim={})
        if nslice is None:
            if data.space[-1] <= 3: canvas = data.numpy()
            elif data.space[0] <= 3: canvas = data.mvdim(0, 2).numpy()
            else: nslice = data.space[-1] // 2
        else: canvas = data.pick(dim, nslice).numpy()
    elif data.nspace == 2:
        if data.has_batch: data = data.sample(random=False, dim=[])
        if data.has_channel:
            if has_cmap: data = data.sample(random=False, dim={})
            else: data = data.sample(number=min(data.channel_size, 3), random=False, dim={}).mvdim(data.channel_dimension, -1)
        canvas = data.numpy()
    elif data.ndim == 3: canvas = data.sample(random=False, dim=[]).numpy()
    else: canvas = data.numpy()
    return plt.imshow(canvas, **kwargs)

@params
def maskshow(*masks, alpha=0.5, **kwargs):
    colors = ['red', 'green', 'blue', 'gold', 'purple', 'gray', 'pink', 'darkgreen', 'dodgerblue']
    colormap = {'R': 'red', 'G': 'green', 'B': 'blue', 'C': 'cyan', 'M': 'magenta', 'Y': 'yellow', 'K': 'black'}
    kwargs = {colormap[k]: v for k, v in kwargs.items()}
    kwargs.update(dict(zip(colors*(len(masks) // len(colors) + 1), masks)))
    

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
