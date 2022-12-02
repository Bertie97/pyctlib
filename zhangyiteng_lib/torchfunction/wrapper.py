from functools import wraps
import torch
import numpy as np
from .utils import tonumpy

def asnumpy(func):
    assert callable(func)
    @warp(func)
    def wrapper(*args, **kwargs):
        args_numpy = (tonumpy(t) for t in args)
        kwargs_numpy = {key: tonumpy(value) for key, value in kwargs.items()}
        return func(*args, **kwargs)
    return wrapper
