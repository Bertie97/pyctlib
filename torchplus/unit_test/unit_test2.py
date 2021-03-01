#! python3 -u
#  -*- coding: utf-8 -*-

##############################
## Author: Yuncheng Zhou
##############################

import sys
# sys.path.append("/Users/zhangyiteng/Software/Python_Lib/new_pyctlib/pyctlib")
# sys.path.append("../..")
import copy
import torch
import numpy as np
sys.path = ["../.."] + sys.path


import torchplus as tp
from pyctlib import scope

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
    t_ = torch.randn(3000, 400, requires_grad=True)
    a_ = t_
    LP_ = torch.nn.Linear(400, 400)
    for _ in range(10): a_ = LP_(a_).relu()
    a_.sum().backward()

assert a.is_cuda is False
assert isinstance(t, tp.Tensor)
assert isinstance(a, tp.Tensor)
assert isinstance(LP.weight, tp.nn.Parameter)
assert isinstance(LP.bias, tp.nn.Parameter)
assert isinstance(tp.tensor(np.array([1., 2.])), tp.Tensor)
# assert t.allclose(t_)
# assert t._grad.allclose(t_._grad)

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
    t_ = torch.randn(3000, 400).to(tp.Device).requires_grad_()
    a_ = t_
    LP_ = torch.nn.Linear(400, 400).to(tp.Device)
    for _ in range(10): a_ = LP_(a_).relu()
    a_.sum().backward()

assert t.allclose(t_.to(tp.Device))
assert isinstance(t, tp.Tensor)
assert isinstance(a, tp.Tensor)
assert isinstance(LP.weight, tp.nn.Parameter)
assert isinstance(LP.bias, tp.nn.Parameter)
assert isinstance(tp.tensor(np.array([1., 2.])), tp.Tensor)
if torch.cuda.is_available():
    assert a.is_cuda
    assert t.is_cuda
    assert tp.tensor(np.array([1., 2.])).is_cuda

tp.nn.ParameterList([tp.nn.Parameter(tp.zeros(30)), tp.nn.Parameter(tp.zeros(30))])
tp.nn.ParameterList([LP.weight, LP.bias])
# assert a.allclose(a_)
# assert t._grad.allclose(t_._grad.to(tp.Device))
