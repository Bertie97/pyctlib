import torch
from zytlib import vector
from functools import reduce
import numpy as np

def get_shape(input):
    if isinstance(input, vector):
        if input.shape and not isinstance(input.shape, str):
            flattened = input.flatten()
            if reduce(lambda x, y: x*y, input.shape, 1) == flattened.length:
                if isinstance(flattened[0], (int, float, bool)):
                    return "V[{}]".format(", ".join(vector(input.shape).map(str)))
                else:
                    return "V[{}][{}]".format(", ".join([str(t) for t in input.shape]), get_shape(flattened[0]))
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
