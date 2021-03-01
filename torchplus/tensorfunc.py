#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package torchplus
##############################
__all__ = """
    crop_as
    up_scale
    down_scale
    image_grid
""".split()

from pyoverload import *
from pyctlib import restore_type_wrapper
from pyctlib import vector
import torch
import numpy as np
import torchplus as tp

def add_special(size, special, fill=1):
    s = special
    if len(s) == 0: pass
    elif len(s) == 1: size = size[:s[0]] + (fill,) + size[s[0]:]
    else: size = size[:s[0]] + (fill,) + size[s[0]:s[1]] + (fill,) + size[s[1]:]
    return size

@overload
@restore_type_wrapper("roi")
def crop_as(x: Array, y: tuple, center: tuple, fill: Scalar=0) -> Array:
    x = tp.Tensor(x)
    size_x = x.shape
    size_y = y

    if isinstance(size_y, tp.Size) and size_x.nspace == size_y.nspace:
        size_y = tuple(size_y.space)
    size_y = tuple(size_y)
    if len(size_y) == len(size_x): pass
    elif len(size_y) == size_x.nspace: size_y = add_special(size_y, size_x.special, -1)
    else: raise TypeError("Mismatch dimensions in 'crop_as', please use -1 if the dimension doesn't need to be cropped. ")
    assert len(size_y) == len(size_x)
    size_y = tuple(a if b == -1 else b for a, b in zip(size_x, size_y))

    if len(center) == len(size_x): pass
    elif len(center) == size_x.nspace: center = add_special(center, size_x.special, -1)
    elif len(x for x in center if x >= 0) == len(x for x in size_y if x >= 0):
        center = tuple(a if b >= 0 else -1 for a, b in zip(center, size_y))
    else: raise TypeError("Mismatch dimensions for the center in 'crop_as', please use -1 if the dimension that is centered or doesn't need cropping. ")
    assert len(center) == len(size_x)
    center = tuple(a / 2 if b == -1 else b for a, b in zip(size_x, center))

    z = fill * tp.ones(*size_y).type_as(x)
    def intersect(u, v):
        return max(u[0], v[0]), min(u[1], v[1])
    z_box = [intersect((0, ly), (- round(float(m - float(ly) / 2)), - round(float(m - float(ly) / 2)) + lx)) for m, lx, ly in zip(center, size_x, size_y)]
    x_box = [intersect((0, lx), (+ round(float(m - float(ly) / 2)), + round(float(m - float(ly) / 2)) + ly)) for m, lx, ly in zip(center, size_x, size_y)]
    # if the two boxes are seperated
    if any([r[0] >= r[1] for r in z_box]) or any([r[0] >= r[1] for r in x_box]): z.roi = None; return z
    region_z = tuple(slice(u, v) for u, v in z_box)
    region_x = tuple(slice(u, v) for u, v in x_box)
    z[region_z] = x[region_x]
    z.roi = region_x
    z.special_from_(x)
    return z

@overload
def crop_as(x: Array, y: Array, center: tuple, fill: Scalar=0) -> Array:
    return crop_as(x, y.shape, center, fill)

@overload
def crop_as(x: Array, y: [tuple, Array], fill: Scalar=0) -> Array:
    center = tuple(m/2 for m in x.shape)
    return crop_as(x, y, center, fill)

@restore_type_wrapper
def up_scale(image, *scaling:int):
    image = tp.Tensor(image)
    if len(scaling) == 0:
        scaling = (1,)
    elif len(scaling) == 1 and iterable(scaling[0]):
        scaling = scaling[0]
    if len(scaling) == 1:
        if isinstance(scaling[0], int):
            scaling *= image.nspace
            scaling = add_special(scaling, image.special, 1)
        else: raise TypeError("Unknown scaling type for 'up_scale'. ")
    elif len(scaling) < image.ndim and len(scaling) == image.nspace:
        scaling = add_special(scaling, image.special, 1)
    for i, s in enumerate(scaling):
        image = (
            image
            .transpose(i, -1)
            .unsqueeze(-1)
            .repeat((1,) * image.ndim + (int(s),))
            .flatten(-2)
            .transpose(i, -1)
        )
    return image

@restore_type_wrapper
def down_scale(image, *scaling:int):
    image = tp.Tensor(image)
    if len(scaling) == 0:
        scaling = (1,)
    elif len(scaling) == 1 and iterable(scaling[0]):
        scaling = scaling[0]
    if len(scaling) == 1:
        if isinstance(scaling[0], int):
            scaling *= image.nspace
            scaling = add_special(scaling, image.special, 1)
        else: raise TypeError("Unknown scaling type for 'down_scale'. ")
    elif len(scaling) < image.ndim and len(scaling) == image.nspace:
        scaling = add_special(scaling, image.special, 1)
    return image[tuple(slice(None, None, s) for s in scaling)]

@overload
@restore_type_wrapper
def image_grid(x: Array):
    return image_grid(x.space)

@overload
def image_grid__default__(*shape):
    if len(shape) == 1 and isinstance(shape, (list, tuple)):
        shape = shape[0]
    ret = tp.stack(tp.meshgrid(*[tp.arange(x) for x in shape]))
    return ret.channel_dimension_(0)


def linear(input, weight, bias):
    result = input @ weight.T
    if bias is not None:
        if bias.dim() == 2:
            return result + bias
        return result + bias.unsqueeze(0)
    return result

def get_shape(input):
    if isinstance(input, list):
        input = vector(input)
        l_shape = input.map(get_shape)
        if l_shape.all(lambda x: x == l_shape[0]):
            return "L{}".format(len(l_shape)) + ("[{}]".format(l_shape[0]) if not l_shape[0].startswith("[") else l_shape[0])
        else:
            return "[{}]".format(", ".join(l_shape))
    if isinstance(input, tuple):
        input = vector(input)
        l_shape = input.map(get_shape)
        if l_shape.all(lambda x: x == l_shape[0]):
            return "T{}".format(len(l_shape)) + ("[{}]".format(l_shape[0]) if not l_shape[0].startswith("[") else l_shape[0])
        else:
            return "[{}]".format(", ".join(l_shape))
    if isinstance(input, torch.Tensor):
        return str(input.shape)
    if isinstance(input, np.ndarray):
        return str(input.shape)
    return str(type(input))[8:-2]
