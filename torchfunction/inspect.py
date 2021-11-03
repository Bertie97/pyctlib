import torch
from pyctlib import vector
from functools import reduce

def get_shape(input):
    if isinstance(input, vector):
        if input.shape and not isinstance(input.shape, str):
            flattened = input.flatten()
            if reduce(lambda x, y: x*y, input.shape, initial=1) == flattened.length:
                return "V[{}][{}]".format(", ".join([str(t) for t in input.shape]), flattened[0].shape)
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
