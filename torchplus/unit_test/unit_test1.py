#! python3 -u
#  -*- coding: utf-8 -*-

##############################
## Author: Yuncheng Zhou
##############################

import sys
sys.path.append("/home1/zhangyiteng/pyctlib")


# import torchplus as tp
import torch
import torchplus as tp
from pyctlib import scope
import copy

tp.set_autodevice(False)
with scope("test tp, cpu"):
    a = tp.Tensor(3000, 400, requires_grad=True)
    LP = tp.nn.Linear(400, 400)
    for t in range(1000): a = LP(a)
    a.sum().backward()

with scope("test torch, cpu"):
    a = torch.Tensor(3000, 400).requires_grad_()
    LP = torch.nn.Linear(400, 400)
    for t in range(1000): a = LP(a)
    a.sum().backward()

with scope("test tp, gpu"):
    a = tp.Tensor(3000, 400, requires_grad=True)
    LP = tp.Tensor(400, 400, requires_grad=True)
    for t in range(10): a = a @ LP
    a.sum().backward()

with scope("test torch, gpu"):
    a = torch.Tensor(3000, 400).requires_grad_().cuda()
    LP = torch.Tensor(400, 400).requires_grad_().cuda()
    for t in range(10): a = a @ LP
    a.sum().backward()

with scope("test tp, gpu"):
    a = tp.Tensor(3000, 400, requires_grad=True)
    LP = tp.Tensor(400, 400, requires_grad=True)
    for t in range(10): a = a @ LP
    a.sum().backward()

with scope("test torch, gpu"):
    a = torch.Tensor(3000, 400).requires_grad_().cuda()
    LP = torch.Tensor(400, 400).requires_grad_().cuda()
    for t in range(10): a = a @ LP
    a.sum().backward()
