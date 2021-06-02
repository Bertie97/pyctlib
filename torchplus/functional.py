import torch

def one_hot_max(x: torch.Tensor, axis=-1):
    indices = x.argmax(axis=axis, keepdim=True)
    one_hot = torch.zeros_like(x)
    one_hot.scatter_(axis, indices, 1)
    return one_hot
