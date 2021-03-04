#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package micomputing
##############################

try: from matplotlib import pyplot as plt
except ImportError:
    raise ImportError("'pyctlib.mic.plot' cannot be used without dependency 'matplotlib'. ")
import torchplus as tp
from matplotlib.pyplot import *
from pyoverload import *
from matplotlib import colors as mc
from pyctlib import vector

def totuple(x, l=None):
    if not isinstance(x, tuple): x = (x,)
    if l is None: l = len(x)
    if l % len(x) == 0: return x * (l // len(x))
    raise TypeError(f"{x} can not be converted into a tuple of length {l}. ")

canvas = None
colors = ['red', 'green', 'blue', 'gold', 'purple', 'gray', 'pink', 'darkgreen', 'dodgerblue']
assert all(c in mc.CSS4_COLORS for c in colors)

def to_image(data: Array, nslice: [int, null]=None, dim: int=-1):
    data = tp.Tensor(data).squeeze()
    if data.ndim <= 1: raise TypeError("Please don't use 'plot.imshow' to demonstrate an array or a scalar. ")
    if data.nspace > 3: raise TypeError("'plot.imshow' takes 2 or 3D-data as input, please reduce the dimension manually or specify special dimensions to reduce. ")
    if data.nspace == 3:
        if data.has_batch: data = data.sample(random=False, dim=[])
        if data.has_channel: data = data.sample(random=False, dim={})
        if nslice is None:
            if data.space[-1] <= 3: pass
            elif data.space[0] <= 3: data = data.mvdim(0, 2)
            else:
                nslice = data.space[-1] // 2
                data = data.pick(dim, nslice)
        else: data = data.pick(dim, nslice)
    elif data.nspace == 2:
        if data.has_batch: data = data.sample(random=False, dim=[])
        if data.has_channel:
            if has_cmap: data = data.sample(random=False, dim={})
            else: data = data.sample(number=min(data.channel_size, 3), random=False, dim={}).mvdim(data.channel_dimension, -1)
    elif data.ndim == 3: data = data.sample(random=False, dim=[])
    return data.float().normalize()

def to_RGB(*color):
    if len(color) == 0: return (1.,) * 3
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
def imshow(data: [Array, null]=None, nslice: [int, null]=None, dim: int=-1, **kwargs):
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
    if data is not None: canvas = to_image(data, nslice, dim)
    if canvas is None or isinstance(canvas, tuple):
        raise TypeError("Please input data in 'imshow' or 'background' to show. ")
    return plt.imshow(canvas.numpy(), **kwargs)

def background(*color):
    """
    Set a background color by RGB or a gray scale, conflict with imshow. 
    """
    global canvas
    canvas = to_RGB(*color)

@params
def maskshow(*masks, on=None, alpha=0.5, nslice=None, dim=-1, stretch=False, **kwargs):
    global canvas
    if on is not None:
        if isinstance(on, (int, tuple)): background(*on)
        elif isarray(on): canvas = to_image(on, nslice, dim)
        elif isinstance(on, list): canvas = to_image(Tensor(on), nslice, dim)
        else: raise TypeError("Unrecognized argument 'on' for 'maskshow'. ")
    elif canvas is None:
        canvas = (1.,) * 3
    if len(masks) == 0: return imshow
    alpha = totuple(alpha, len(masks))
    new_masks = []
    new_alpha = []
    for m, a in zip(masks, alpha):
        img = to_image(m, nslice, dim)
        if img.ndim == 3:
            new_masks.extend(x.squeeze(-1) for x in img.split(1, dim=dim))
            new_alpha.extend([a] * img.size(dim))
        else:
            new_masks.append(img)
            new_alpha.append(a)
    color_mask_map = [(to_RGB(c), m, a) for c, m, a in zip(colors*(len(new_masks) // len(colors) + 1), new_masks, new_alpha)]
    color_mask_map.extend((to_RGB(c), m, alpha[0]) for c, m in kwargs.items())

    if not stretch:
        shapes = [m.ishape for _, m, _ in color_mask_map]
        target_shape = shapes[0]
        if len(set(shapes)) > 1 or not isinstance(canvas, tuple) and target_shape != canvas.shape:
            raise TypeError("Please use masks of the same size as the background image, "
                            "or use 'stretch=True' in 'maskshow' to automatically adjust the image sizes. ")
    else:
        def adjust(m, to):
            ms = tuple(m.shape)
            scaling = tuple((a // b, b // a) for a, b in zip(to, ms))
            return m.down_scale([max(v, 1) for u, v in scaling]).up_scale([max(u, 1) for u, v in scaling]).crop_as(to)
        shapes = [m.ishape for _, m, _ in color_mask_map]
        if not isinstance(canvas, tuple): shapes.append(canvas.shape[:2])
        areas = [u * v for u, v in shapes]
        target_shape = shapes[areas.index(max(areas))]
        color_mask_map = [(c, adjust(m, to=target_shape), a) for c, m, a in color_mask_map]
        canvas = adjust(canvas, to=target_shape)

    target_shape = tp.Size(*target_shape, {3})
    if isinstance(canvas, tuple): canvas = tp.Tensor(list(canvas)).expand_to(target_shape)
    elif canvas.ndim == 2: canvas = canvas.expand_to(target_shape)
    coeff = vector(1 - a * m for _, m, a in color_mask_map).prod()
    canvas *= coeff
    for i, (c, m, a) in enumerate(color_mask_map):
        coeff = vector(a * m if j == i else 1 - a * m for j, (_, m, a) in enumerate(color_mask_map)).prod()
        canvas += coeff.unsqueeze(-1) * m.unsqueeze(-1) * tp.Tensor(list(c)).unsqueeze(0, 1)

    return plt.imshow(canvas.numpy(), **kwargs)

def smooth(curve):
    """
    curve: 2xn
    """
    middle = (curve[:, :-2] + curve[:, 1:-1] + curve[:, 2:]) / 3
    if all(curve[:, 0] == curve[:, -1]):
        head = (curve[:, :1] + curve[:, 1:2] + curve[:, -2:-1]) / 3
        return tp.cat(head, middle, head, dim=1)
    return tp.cat(curve[:, :1], middle, curve[:, -1:], dim=1)

def sharpen(curve, old_curve):
    """
    sharpen towards old_curve
    curve, old_curve: 2xn
    """
    a = tp.cat(curve[:, -2:-1] - curve[:, -1:], curve[:, :-1] - curve[:, 1:], dim=1)
    b = tp.cat(curve[:, 1:] - curve[:, :-1], curve[:, 1:2] - curve[:, :1], dim=1)
    costheta = (a*b).sum(0) / tp.sqrt((a*a).sum(0)) / tp.sqrt((b*b).sum(0))
    new_curve = curve.clone()
    new_curve[:, costheta > -0.9] = old_curve[:, costheta > -0.9]
    return new_curve

def constraint(new_curve, old_curve, constraint_curve):
    """
    constraint new_curve towards constraint_curve
    curve, old_curve: 2xn
    """
    dis_sqs = tp.sum((new_curve - constraint_curve) ** 2, 0)
    percentile = 2 #min(np.sort(dis_sqs)[98 * len(dis_sqs) // 100], 1)
    return tp.where(dis_sqs <= percentile, new_curve, old_curve)

def border(mask, min_length = 10):
    grid = tp.image_grid(*mask.shape)
    mask = mask > 0.5
    idx = mask[1:, :] ^ mask[:-1, :]
    idx = idx.expand_to(2, -1, mask.size(1))
    locs1 = (grid[:, 1:, :] + grid[:, :-1, :])[idx] / 2
    idx = mask[:, 1:] ^ mask[:, :-1]
    idx = idx.expand_to(2, mask.size(0), -1)
    locs2 = (grid[:, :, 1:] + grid[:, :, :-1])[idx] / 2
    locs = tp.cat(locs1.reshape(2, -1), locs2.reshape(2, -1), dim=1)
    if locs.size == 0: return []
    curves = []
    unvisited = tp.ones(locs.shape[-1])
    while True:
        if not any(unvisited): break
        first = tp.argmax(unvisited).item()
        cloc = locs[:, first:first + 1]
        unvisited[first] = 0
        curve = cloc
        while True:
            dissq = tp.sum((locs - cloc) ** 2, 0)
            inloc = tp.argmax(tp.where((unvisited > 0) & (dissq > 0), 1/dissq.clamp(min=0.1), tp.tensor(0).float()))
            if dissq[inloc] > 2: break
            cloc = locs[:, inloc:inloc + 1]
            curve = tp.cat(curve, cloc, dim=1)
            unvisited[inloc] = 0
            if not any(unvisited): break
        sloc = locs[:, first:first + 1]
        if tp.sum((cloc - sloc) ** 2) <= 2:
            curve = tp.cat(curve, sloc, dim=1)
        if curve.shape[1] <= min_length: continue
        scurve = curve
        for _ in range(100): scurve = constraint(smooth(scurve), scurve, curve)
        ccurve = scurve
        for _ in range(100):
            scurve = constraint(sharpen(scurve, curve), scurve, ccurve)
            scurve = constraint(smooth(scurve), scurve, curve)
        curves.append(scurve)
    return curves

def bordershow(*masks, on=None, mask_alpha=0., nslice=None, dim=-1, stretch=False, min_length = 10, **kwargs):
    global canvas
    if on is not None:
        if isinstance(on, (int, tuple)): background(*on)
        elif isarray(on): canvas = to_image(on, nslice, dim)
        elif isinstance(on, list): canvas = to_image(Tensor(on), nslice, dim)
        else: raise TypeError("Unrecognized argument 'on' for 'maskshow'. ")
    elif canvas is None:
        canvas = (1.,) * 3
    if len(masks) == 0: return
    new_masks = []
    for m in masks:
        img = to_image(m, nslice, dim)
        if img.ndim == 3:
            new_masks.extend(x.squeeze(-1) for x in img.split(1, dim=dim))
        else:
            new_masks.append(img)
    color_mask_map = [(to_RGB(c), m) for c, m in zip(colors*(len(new_masks) // len(colors) + 1), new_masks)]
    color_mask_map.extend((to_RGB(c), m) for c, m in kwargs.items())

    new_masks = [m for _, m in color_mask_map]
    shapes = [m.ishape for _, m in color_mask_map]
    if not stretch:
        target_shape = shapes[0]
        if len(set(shapes)) > 1 or not isinstance(canvas, tuple) and target_shape != canvas.shape:
            raise TypeError("Please use masks of the same size as the background image, "
                            "or use 'stretch=True' in 'maskshow' to automatically adjust the image sizes. ")
    else:
        def adjust(m, to):
            ms = tuple(m.shape)
            scaling = tuple((a // b, b // a) for a, b in zip(to, ms))
            return m.down_scale([max(v, 1) for u, v in scaling]).up_scale([max(u, 1) for u, v in scaling]).crop_as(to)
        if not isinstance(canvas, tuple): shapes.append(canvas.shape[:2])
        areas = [u * v for u, v in shapes]
        target_shape = shapes[areas.index(max(areas))]
        color_mask_map = [(c, adjust(m, to=target_shape)) for c, m in color_mask_map]
        canvas = adjust(canvas, to=target_shape)

    if mask_alpha > 0: maskshow(*new_masks, alpha=mask_alpha, **kwargs)
    else: imshow(**kwargs)
    plots = []
    for color, mask in color_mask_map:
        curves = border(mask, min_length)
        for c in curves:
            plots.append(plt.plot(c[1], c[0], color = color, **kwargs))
    return plots

