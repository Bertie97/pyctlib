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
    """
    Flattens the input tensor `x` according to the specified `cmd` argument.

    Args:
        cmd (str): A string specifying the desired shape of the flattened tensor.
                   It contains two parts separated by "->" sign. The left part 
                   specifies the current shape of the tensor, and the right part 
                   specifies the desired shape of the flattened tensor.
        x (torch.Tensor): The input tensor to be flattened.

    Returns:
        torch.Tensor: A flattened version of the input tensor `x`, according to 
                      the specified `cmd` argument.

    Raises:
        AssertionError: If the left command is not valid with the input tensor 
                        `x`, or if the right command is not valid.

    Example:
        # Flattening a 2x3x4 tensor into a 24-element vector
        x = torch.randn(2, 3, 4)
        flattened = einflatten("abc->[abc]", x)
        assert flattened.shape == (24,)
    """
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
    """
    Computes the average of the input tensor `x` over different conditions specified by the `condition` tensor.
    If `condition_num` is an integer, the function computes the average of `x` over `condition_num` conditions,
    where the condition of each element of `x` is specified by the corresponding element of `condition`.
    If `condition_num` is a tuple, the function computes the average of `x` over the conditions specified by
    the multi-dimensional `condition` tensor. Each dimension of `condition` specifies a different level of condition.

    Args:
        x (torch.Tensor): The input tensor to be averaged.
        condition_num (Union[int, Tuple[int]]): An integer or tuple specifying the number of conditions.
        condition (torch.Tensor): A tensor specifying the condition of each element of `x`.
                                  The shape of `condition` should match the shape of `x` except in the dimension `dim`.
                                  If `condition_num` is an integer, `condition` should have shape `(x.shape[dim],)`.
                                  If `condition_num` is a tuple, `condition` should have shape
                                  `(x.shape[0], x.shape[1], ..., x.shape[dim-1], condition_num[0], condition_num[1], ..., condition_num[-1], x.shape[dim+1], ..., x.shape[-1])`.
        dim (int): The dimension along which to average the input tensor `x`. Default is `-1` (the last dimension).

    Returns:
        ConditionAverage: A named tuple containing two elements: `avg_tensor` and `count`.
                          `avg_tensor` is a tensor containing the average of `x` over different conditions.
                          `count` is a tensor containing the number of elements in each condition.

    Raises:
        AssertionError: If the input arguments do not meet the requirements.

    Example:
        # Computing the average of a 2x3x4 tensor over 5 conditions specified by a 2x2 condition tensor
        x = torch.randn(2, 3, 4)
        condition_num = (2, 2)
        condition = torch.tensor([[[0, 0], [0, 1]], [[1, 1], [0, 1]]])
        result = condition_average(x, condition_num, condition)
        assert result.avg_tensor.shape == (2, 2, 3, 4)
        assert result.count.shape == (2, 2)
    """
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
