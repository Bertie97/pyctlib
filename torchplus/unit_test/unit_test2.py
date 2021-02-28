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
sys.path = ["../.."] + sys.path


import torchplus as tp
from pyctlib import scope

tp.set_autodevice(True)
tp.manual_seed(0)
with scope("test tp, cpu"):
    t = tp.randn(3000, 400, requires_grad=True)
    a = t
    LP = tp.nn.Linear(400, 400)
    for _ in range(10): a = LP(a)
    a.sum().backward()

torch.manual_seed(0)
with scope("test torch, cpu"):
    t_ = torch.randn(3000, 400, requires_grad=True)
    a_ = t_
    LP_ = torch.nn.Linear(400, 400)
    for _ in range(10): a_ = LP(a_)
    a_.sum().backward()

assert t.allclose(t_)
assert t._grad.allclose(t_._grad)

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
assert a.allclose(a_)
assert t._grad.allclose(t_._grad.to(tp.Device))
