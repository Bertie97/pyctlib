import torch
import numpy as np
import random
import os
import torch.nn as nn
import torch.nn.functional as F
import re
from zytlib.utils import totuple

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
    if len(tensors) == 1:
        return tensors[0]
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
        return type(tensor)(pad_to(t, shape, value=value, ignore_dim=ignore_dim) for t in tensor)
    else:
        raise ValueError()

equation_full_re = re.compile("([+-]([a-zA-Z0-9]+(,[a-zA-Z0-9]+)*))+->[a-zA-Z]*")
left_equation_re = re.compile("([+-])([a-zA-Z0-9]+(,[a-zA-Z0-9]+)*)")

def einsum(equation, *operands):
    assert len(operands) > 0
    if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        operands = list(operands[0])
    else:
        operands = list(operands)
    if equation[0] not in ["+", "-"]:
        equation = "+" + equation
    equation_len = len(equation)

    index = 0
    ret = 0
    left_equation_list = list()
    left_slice_list = list()
    left_equation = ""
    left_slice = tuple()
    left_flag = True
    left_operands_equation_list = list()
    left_operands_slice_list = list()
    left_operands_sgn_list = list()
    right_equation = ""
    positive_or_negative = None
    while index < equation_len:
        ec = equation[index]
        if ec == " ":
            index += 1
            continue
        if left_flag is True:
            if positive_or_negative is None:
                if ec in ["+", "-"]:
                    if index + 1 >= len(equation):
                        raise RuntimeError(f"invalid equation {equation}")
                    if equation[index + 1] == ">":
                        left_flag = False
                        index += 2
                        continue
                    positive_or_negative = ec
                else:
                    raise RuntimeError(f"invalid equation {equation}")
                index += 1
                continue
            else:
                if ec.isalpha():
                    left_equation = left_equation + ec
                    left_slice = (*left_slice, slice(None))
                    index += 1
                    continue
                elif ec.isdigit():
                    left_slice = (*left_slice, int(ec))
                    index += 1
                    continue
                elif ec == ",":
                    left_equation_list.append(left_equation)
                    left_slice_list.append(left_slice)
                    left_equation = ""
                    left_slice = tuple()
                    index += 1
                    continue
                elif ec == "+" or ec == "-":
                    left_equation_list.append(left_equation)
                    left_slice_list.append(left_slice)
                    left_operands_equation_list.append(left_equation_list)
                    left_operands_slice_list.append(left_slice_list)
                    left_operands_sgn_list.append(positive_or_negative)
                    left_equation = ""
                    left_slice = tuple()
                    left_equation_list = list()
                    left_slice_list = list()
                    positive_or_negative = None
                    continue
                else:
                    raise RuntimeError(f"invalid character [{ec}]")
        else:
            right_equation = equation[index:]
            index = len(equation)

    if left_equation != "":
        left_equation_list.append(left_equation)
        left_slice_list.append(left_slice)
        left_operands_equation_list.append(left_equation_list)
        left_operands_slice_list.append(left_slice_list)
        left_operands_sgn_list.append(positive_or_negative)
        left_equation = ""
        left_slice = tuple()
        left_equation_list = list()
        left_slice_list = list()
        positive_or_negative = None


    total_num = 0
    ret = 0
    for pindex in range(len(left_operands_equation_list)):
        num = len(left_operands_equation_list[pindex])
        left_eqn = ",".join(left_operands_equation_list[pindex])
        opers = list()
        for i in range(num):
            if total_num + i >= len(operands):
                raise RuntimeError("not enough tensor are provided")
            if (ope:= operands[total_num + i]).dim() != len(left_operands_slice_list[pindex][i]):
                def _from_slice_eqn_to_full_eqn(slc, eqn):
                    fulleqn=""
                    p = 0
                    for index in range(len(slc)):
                        if isinstance(slc[index], slice):
                            fulleqn += eqn[p]
                            p += 1
                        elif isinstance(slc[index], int):
                            fulleqn += str(slc[index])
                        else:
                            raise RuntimeError()
                    return fulleqn
                fulleqn = _from_slice_eqn_to_full_eqn(left_operands_slice_list[pindex][i], left_operands_equation_list[pindex][i])
                raise RuntimeError(f"dimension of the {total_num+i+1}-th tensor (shape: {ope.shape}) is not compatible with what is said in the equation [{fulleqn}]")
            opers.append(operands[total_num+i][left_operands_slice_list[pindex][i]])
        result = torch.einsum(f"{left_eqn}->{right_equation}", opers)
        total_num += num
        if left_operands_sgn_list[pindex] == "+":
            ret += result
        else:
            ret -= result
    if total_num + i < len(operands):
        raise RuntimeError(f"{len(operands) - total_num - i} more tensor.")
    return ret

def meshgrid(*args):
    args = totuple(args)
    if len(args) == 0:
        return torch.Tensor()
    return torch.stack(torch.meshgrid(*args), -1).view(-1, len(args))
