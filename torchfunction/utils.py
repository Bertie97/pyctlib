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
