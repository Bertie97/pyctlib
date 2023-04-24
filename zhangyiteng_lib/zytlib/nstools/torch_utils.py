import torch
import numpy as np
from typing import Callable, Any, List, Union, Tuple
from collections import namedtuple
from ..container.vector import vector
import re

def to_LongTensor(x: Any) -> torch.LongTensor:
    if isinstance(x, torch.Tensor):
        return x.long()
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).long()
    elif isinstance(x, (list, tuple)):
        return torch.tensor(x).long()
    else:
        raise TypeError("Unsupported input type. Must be a PyTorch tensor, NumPy ndarray, list, or tuple.")


def demean(x: torch.Tensor, dim: int=-1) -> torch.Tensor:
    return x - x.mean(dim, keepdim=True)

def corrcoef(x: torch.Tensor, y: torch.Tensor, dim: int=-1, *, eps: float=1e-10) -> torch.Tensor:
    # x: [*, trial]
    # y: [*, trial]
    # correlation between x and y in the dim axis
    assert x.dim() == y.dim()
    assert x.shape == y.shape

    x = demean(x, dim)
    y = demean(y, dim)

    cov_xy = (x * y).sum(dim)
    cov_x = (x).pow(2).sum(dim)
    cov_y = (y).pow(2).sum(dim)

    return cov_xy / (eps + cov_x * cov_y).sqrt()

def einflatten(cmd: str, x: torch.Tensor) -> torch.Tensor:
    if "->" not in cmd:
        return x.flatten()
    left_cmd, right_cmd = cmd.split("->")
    if not right_cmd:
        return x.flatten()
    assert len(left_cmd) == x.dim()
    assert len(left_cmd) == len(set(left_cmd))
    new_shape: List[int] = list()
    i, j = 0, 0
    while i < len(left_cmd):
        if right_cmd[j].isalpha():
            assert left_cmd[i] == right_cmd[j]
            new_shape.append(x.shape[i])
            i += 1
            j += 1
        else:
            assert right_cmd[j] == "["
            j += 1
            t = -1
            while j < len(right_cmd):
                if right_cmd[j] == "]":
                    t = t * -1
                    j += 1
                    break
                assert right_cmd[j] == left_cmd[i]
                t *= x.shape[i]
                i += 1
                j += 1
            assert t > 0
            new_shape.append(t)
    return x.reshape(tuple(new_shape))

ConditionAverage = namedtuple("ConditionAverage", ["avg_tensor", "count"])

def condition_average(x: torch.Tensor, condition_num: Union[int, Tuple[int]], condition: torch.Tensor, *, dim: int=-1) -> ConditionAverage:
    dim = dim % x.dim()
    if isinstance(condition_num, int):
        assert condition_num > 0
        assert condition.dim() == 1
        ret_shape = list(x.shape)
        ret_shape[dim] = condition_num
        ret = ConditionAverage(torch.zeros(ret_shape).to(x.device), torch.zeros(condition_num).to(x.device).long())
        condition = to_LongTensor(condition)
        for c in range(condition_num):
            ci = condition == c
            if ci.sum() == 0:
                continue
            x_index: List[Any] = [slice(None)] * dim + [ci] +  [slice(None)] * (x.dim() - 1 - dim)
            ret_index: List[Any] = [slice(None)] * dim + [c] +  [slice(None)] * (x.dim() - 1 - dim)
            ret.avg_tensor[ret_index] = x[x_index].mean(dim)
            ret.count[c] = ci.sum()
        return ret
    elif isinstance(condition_num, tuple):
        assert condition.dim() == 2
        assert len(condition_num) == condition.shape[-1]
        new_condition = torch.zeros_like(condition[:, -1])
        basis = 1
        for i in range(condition.dim() - 1, -1, -1):
            new_condition += condition[:, i] * basis
            basis *= condition_num[i]
        ret_shape = tuple(list(x.shape[:dim]) + list(condition_num) + list(x.shape[dim + 1:]))
        temp_ret = condition_average(x, basis, new_condition, dim=dim)
        ret = ConditionAverage(temp_ret.avg_tensor.reshape(ret_shape), temp_ret.count.reshape(condition_num))
        return ret
