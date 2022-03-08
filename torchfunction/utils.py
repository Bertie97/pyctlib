import torch
import numpy as np
import random
import os
import torch.nn as nn
import torch.nn.functional as F

def seed_torch(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def low_rank(self, n, r, name="J"):
    self.__setattr__("_" + name + "_r", r)
    if r == -1:
        self.__setattr__("_" + name, nn.Parameter(torch.randn(n, n) * math.sqrt(n)))
    else:
        self.__setattr__("_" + name + "_V", nn.Parameter(torch.randn(n, r)))
        self.__setattr__("_" + name + "_U", nn.Parameter(torch.randn(n, r)))
    def func(self):
        if getattr(self, f"_{name}_r") == -1:
            return getattr(self, f"_{name}")
        else:
            return getattr(self, f"_{name}_U") @ getattr(self, f"_{name}_V").T
    def load_state_dict(state: table):
        max_rank = r
        if isinstance(state, nn.Module):
            state = state.state_dict()
        state = state.copy()
        if max_rank == -1 and f"_{name}" in state:
            getattr(self, f"_{name}").data.copy_(state[f"_{name}"])
        elif max_rank != -1 and f"_{name}_U" in state and f"_{name}_V" in state:
            U = state[f"_{name}_U"]
            V = state[f"_{name}_V"]
            getattr(self, f"_{name}_V").data.copy_(V)
            getattr(self, f"_{name}_U").data.copy_(U)
        elif max_rank != -1 and f"_{name}" in state:
            U, S, V = torch.pca_lowrank(state[f"_{name}"], q=max_rank, center=False)
            getattr(self, f"_{name}_V").data.copy_(V @ torch.diag(torch.sqrt(S)))
            getattr(self, f"_{name}_U").data.copy_(U @ torch.diag(torch.sqrt(S)))
        elif max_rank == -1 and f"_{name}_U" in state and f"_{name}_V" in state:
            U = state[f"_{name}_U"]
            V = state[f"_{name}_V"]
            getattr(self, f"_{name}").data.copy_(U @ V.T)
    if not hasattr(self, name):
        setattr(self.__class__, name, property(func))
    return load_state_dict

def pad(tensors, dim=0, padded=0):

    assert len(tensors) > 1
    shape = tensors[0].shape
    for t in tensors:
        len(shape) == len(t.shape)
    shape_max = [0 for _ in range(len(shape))]
    for t in tensors:
        for index in range(len(shape)):
            raise NotImplementedError()

def pad_to(tensors, shape, value=0):
    assert len(tensors.shape) == len(shape)
import torch
import numpy as np
import random
import os
import torch.nn as nn

def seed_torch(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def low_rank(self, n, r, name="J"):
    self.__setattr__("_" + name + "_r", r)
    if r == -1:
        self.__setattr__("_" + name, nn.Parameter(torch.randn(n, n) * math.sqrt(n)))
    else:
        self.__setattr__("_" + name + "_V", nn.Parameter(torch.randn(n, r)))
        self.__setattr__("_" + name + "_U", nn.Parameter(torch.randn(n, r)))
    def func(self):
        if getattr(self, f"_{name}_r") == -1:
            return getattr(self, f"_{name}")
        else:
            return getattr(self, f"_{name}_U") @ getattr(self, f"_{name}_V").T
    def load_state_dict(state: table):
        max_rank = r
        if isinstance(state, nn.Module):
            state = state.state_dict()
        state = state.copy()
        if max_rank == -1 and f"_{name}" in state:
            getattr(self, f"_{name}").data.copy_(state[f"_{name}"])
        elif max_rank != -1 and f"_{name}_U" in state and f"_{name}_V" in state:
            U = state[f"_{name}_U"]
            V = state[f"_{name}_V"]
            getattr(self, f"_{name}_V").data.copy_(V)
            getattr(self, f"_{name}_U").data.copy_(U)
        elif max_rank != -1 and f"_{name}" in state:
            U, S, V = torch.pca_lowrank(state[f"_{name}"], q=max_rank, center=False)
            getattr(self, f"_{name}_V").data.copy_(V @ torch.diag(torch.sqrt(S)))
            getattr(self, f"_{name}_U").data.copy_(U @ torch.diag(torch.sqrt(S)))
        elif max_rank == -1 and f"_{name}_U" in state and f"_{name}_V" in state:
            U = state[f"_{name}_U"]
            V = state[f"_{name}_V"]
            getattr(self, f"_{name}").data.copy_(U @ V.T)
    if not hasattr(self, name):
        setattr(self.__class__, name, property(func))
    return load_state_dict

def pad_cat(tensors, dim=0, value=0):
    assert len(tensors) > 1
    shape = tensors[0].shape
    for t in tensors:
        len(shape) == len(t.shape)
    shape_max = [0 for _ in range(len(shape))]
    for t in tensors:
        for index in range(len(shape)):
            shape_max[index] = max(shape_max[index], t.shape[index])
    return torch.cat(pad_to(tensors, shape_max, value=value, ignore_dim=dim),dim=dim)

def pad_to(tensor, shape, value=0, ignore_dim=None):
    """
    tensor can be Tensor or List of Tensor
    """
    shape = [t for t in shape]

    if isinstance(tensor, torch.Tensor):
        assert tensor.dim() == len(shape)
        if ignore_dim is None:
            pass
        elif isinstance(ignore_dim, int):
            ignore_dim = ignore_dim % tensor.dim()
            shape[ignore_dim] = tensor.shape[ignore_dim]
        elif isinstance(ignore_dim, list):
            for id in ignore_dim:
                shape[id] = tensor.shape[id]
        pd = tuple()
        for index in range(tensor.dim()):
            assert tensor.shape[index] <= shape[index]
            pd = (0, shape[index] - tensor.shape[index], *pd)
        return F.pad(tensor, pd, mode="constant", value=value)
    elif isinstance(tensor, (list, tuple)):
        return type(tensor)(pad_to(t, shape, value=0, ignore_dim=ignore_dim) for t in tensor)
    else:
        raise ValueError()
