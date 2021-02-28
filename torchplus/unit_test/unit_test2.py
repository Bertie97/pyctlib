#! python3 -u
#  -*- coding: utf-8 -*-

##############################
## Author: Yuncheng Zhou
##############################

import sys
# sys.path.append("/Users/zhangyiteng/Software/Python_Lib/new_pyctlib/pyctlib")
# sys.path.append("../..")
sys.path = ["../.."] + sys.path


import torch
import torchplus as tp
from pyctlib import scope
import copy

tp.set_autodevice(False)
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
    for _ in range(10): a = LP(a)
    a.sum().backward()

torch.manual_seed(0)
with scope("test torch, gpu"):
    t_ = torch.randn(3000, 400, requires_grad=True).cuda()
    a_ = t_
    LP_ = torch.nn.Linear(400, 400)
    for _ in range(10): a_ = LP(a_).cuda()
    a_.sum().backward()

assert t.allclose(t_)
assert t._grad.allclose(t_._grad)
