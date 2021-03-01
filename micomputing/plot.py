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
from matplotlib import colors as mc

canvas = None

def to_image(data: Array, nslice: [int, null]=None, dim: int=-1):
    data = tp.Tensor(data).squeeze()
    if data.ndim <= 1: raise TypeError("Please don't use 'plot.imshow' to demonstrate an array or a scalar. ")
    if data.nspace > 3: raise TypeError("'plot.imshow' takes 2 or 3D-data as input, please reduce the dimension manually or specify special dimensions to reduce. ")
    if data.nspace == 3:
        if data.has_batch: data = data.sample(random=False, dim=[])
        if data.has_channel: data = data.sample(random=False, dim={})
        if nslice is None:
            if data.space[-1] <= 3: return data.normalize().numpy()
            elif data.space[0] <= 3: return data.mvdim(0, 2).normalize().numpy()
            else: nslice = data.space[-1] // 2
        return data.pick(dim, nslice).normalize().numpy()
    elif data.nspace == 2:
        if data.has_batch: data = data.sample(random=False, dim=[])
        if data.has_channel:
            if has_cmap: data = data.sample(random=False, dim={})
            else: data = data.sample(number=min(data.channel_size, 3), random=False, dim={}).mvdim(data.channel_dimension, -1)
        return data.normalize().numpy()
    elif data.ndim == 3: return data.sample(random=False, dim=[]).normalize().numpy()
    return data.normalize().numpy()

def to_RGB(*color):
    if len(color) = 0: return (1.,) * 3
    elif len(color) == 1:
        c = color[0]
        if isinstance(c, float) and 0 <= c <= 1: return (c,) * 3
        elif isinstance(c, (int, float)) and 0 <= c <= 255: return (c / 255,) * 3
        elif isinstance(c, tuple): return to_RGB(*c)
        elif isinstance(c, str):
            if not c.startswith('#'): c = mc.BASE_COLORS.get(c.lower(), mc.CSS4_COLORS.get(c.lower(), '#FFF'))
            if isinstance(c, tuple): return to_RGB(*c)
            return mc.hex2color(c)
        else: raise TypeError("Unaccepted color type. ")
    elif len(color) == 3:
        if all(isinstance(c, float) and 0 <= c <= 1 for c in color): return color
        elif all(isinstance(c, (int, float)) and 0 <= c <= 255 for c in color): return tuple(c / 255 for c in color)
        else: raise TypeError("Unaccepted color type. ")
    else: raise TypeError("Unaccepted color type. ")


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
    canvas = to_image(data, nslice, dim)
    return plt.imshow(canvas, **kwargs)

def background(*color):
    """
    Set a background color by RGB or a gray scale, conflict with imshow. 
    """
    global canvas
    canvas = to_RGB(*color)

@params
def maskshow(*masks, alpha=0.5, nslice=None, dim=-1, **kwargs):
    if len(masks) == 0: return
    new_masks = []
    for m in masks:
        img = to_image(m, nslice, dim)
        if img.ndim == 3: new_masks.extend(x.squeeze(-1) for x in img.split(1, dim=-1))
        else: new_masks.append(img)
    colors = ['red', 'green', 'blue', 'gold', 'purple', 'gray', 'pink', 'darkgreen', 'dodgerblue']
    assert all(c in mc.CSS4_COLORS for c in colors)
    color_mask_map = {to_RGB(k): v for k, v in kwargs.items()}
    color_mask_map.update({to_RGB(k): v for k, v in zip(colors*(len(new_masks) // len(colors) + 1), new_masks))})
    if len({m.ishape for m in color_mask_map.values()}) > 1: raise TypeError('Please')


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
