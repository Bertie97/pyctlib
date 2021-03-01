#! python3 -u
#  -*- coding: utf-8 -*-

##############################
## Author: Yuncheng Zhou
##############################

import os, sys
# sys.path.append("/Users/zhangyiteng/Software/Python_Lib/new_pyctlib/pyctlib")
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import copy
import torch
import numpy as np
sys.path = ["../.."] + sys.path


import torchplus as tp
from pyctlib import scope, jump

t = tp.randn(3000, 400, requires_grad=True)
print(tp.nn.functional.normalize(t, p=2, dim=1))

tp.set_autodevice(True)
tp.manual_seed(0)
with scope("test tp, gpu"):
    t = tp.randn(3000, 400, requires_grad=True)
    a = t
    LP = tp.nn.Linear(400, 400)
    for _ in range(10): a = LP(a).relu()
    a.sum().backward()

torch.manual_seed(0)
with scope("test torch, gpu"):
    t_ = torch.randn(3000, 400).to(tp.Device).requires_grad_(True)
    a_ = t_
    LP_ = torch.nn.Linear(400, 400).to(tp.Device)
    for _ in range(10): a_ = LP_(a_).relu()
    a_.sum().backward()

assert a.is_cuda is True
assert t.allclose(t_)
assert isinstance(t, tp.Tensor)
assert isinstance(a, tp.Tensor)
assert isinstance(LP.weight, tp.nn.Parameter)
assert isinstance(LP.bias, tp.nn.Parameter)
assert isinstance(tp.tensor(np.array([1., 2.])), tp.Tensor)
if torch.cuda.is_available():
    assert a.is_cuda
    assert t.is_cuda
    assert tp.tensor(np.array([1., 2.])).is_cuda

tp.set_autodevice(False)
tp.manual_seed(0)
with scope("test tp, cpu"):
    t = tp.randn(3000, 400, requires_grad=True)
    a = t
    LP = tp.nn.Linear(400, 400)
    for _ in range(10): a = LP(a).relu()
    a.sum().backward()

torch.manual_seed(0)
with scope("test torch, cpu"):
    t_ = torch.randn(3000, 400).requires_grad_()
    a_ = t_
    LP_ = torch.nn.Linear(400, 400)
    for _ in range(10): a_ = LP_(a_).relu()
    a_.sum().backward()

assert a.is_cuda is False
assert t.allclose(t_)
assert isinstance(t, tp.Tensor)
assert isinstance(a, tp.Tensor)
assert isinstance(LP.weight, tp.nn.Parameter)
assert isinstance(LP.bias, tp.nn.Parameter)
assert isinstance(tp.tensor(np.array([1., 2.])), tp.Tensor)

tp.nn.ParameterList([tp.nn.Parameter(tp.zeros(30)), tp.nn.Parameter(tp.zeros(30))])
tp.nn.ParameterList([LP.weight, LP.bias])
