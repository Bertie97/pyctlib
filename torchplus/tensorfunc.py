#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package torchplus
##############################
__all__ = """
    crop_as
""".split()

from pyoverload import *
from .tensor import Tensor, ones
from pyctlib import restore_type_wrapper
from pyctlib import vector
import torch
import numpy as np

@overload
@restore_type_wrapper("roi")
def crop_as(x: Array, y: tuple, center: tuple, fill: Scalar=0) -> Array:
    x = Tensor(x)
    size_x = x.shape
    size_y = y
    if len(size_y) < len(size_x) - 1 or len(size_y) < len(size_x) and not x.batch_dimension:
        raise TypeError("Need more dimensions in size y, please use -1 if the dimension doesn't need to be cropped. ")
    if len(size_y) > len(size_x): raise TypeError("Too many dimensions in target size y, please check your input.")
    if len(size_y) == len(size_x) - 1: size_y = size_y[:x.batch_dimension] + (-1,) + size_y[x.batch_dimension:]
    assert len(size_x) == len(size_y)
    size_y = tuple(a if b == -1 else b for a, b in zip(size_x, size_y))
    if len(center) < len(size_x) - 1 or len(center) < len(size_x) and not x.batch_dimension:
        raise TypeError("Need more dimensions in center, please use -1 if the dimension that is centered or doesn't need cropping. ")
    if len(center) > len(size_x): raise TypeError("Too many dimensions in center, please check your input.")
    if len(center) == len(size_x) - 1: center = center[:x.batch_dimension] + (-1,) + center[x.batch_dimension:]
    assert len(center) == len(size_x)
    center = tuple(a / 2 if b == -1 else b for a, b in zip(size_x, center))
    z = Tensor(fill) * ones(*size_y).type_as(x)
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
    return z

@overload
def crop_as(x: Array, y: Array, center: tuple, fill: Scalar=0) -> Array:
    return crop_as(x, tuple(y.shape), center, fill)

@overload
def crop_as(x: Array, y: [tuple, Array], fill: Scalar=0) -> Array:
    center = tuple(m/2 for m in x.shape)
    return crop_as(x, y, center, fill)

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
