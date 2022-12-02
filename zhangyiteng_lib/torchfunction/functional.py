import torch
import numpy as np
from pyctlib import vector
import torch

def one_hot_max(x: torch.Tensor, axis=-1) -> torch.Tensor:
    indices = x.argmax(axis=axis, keepdim=True)
    one_hot = torch.zeros_like(x)
    one_hot.scatter_(axis, indices, 1)
    return one_hot

def softmax_sample(x: torch.Tensor, axis=-1) -> torch.Tensor:
    if axis != -1 and axis != x.dim() - 1:
        t = x.transpose(axis, -1)
    else:
        t = x
    one_hot_categorical = torch.distributions.OneHotCategorical(logits=t)
    sample = one_hot_categorical.sample()
    if axis != -1 and axis != x.dim() - 1:
        return sample.transpose(axis, -1)
    return sample

