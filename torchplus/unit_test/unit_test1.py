#! python3 -u
#  -*- coding: utf-8 -*-

##############################
## Author: Yuncheng Zhou
##############################

import sys
# sys.path.append("/Users/zhangyiteng/Software/Python_Lib/new_pyctlib/pyctlib")
sys.path = ["/Users/zhangyiteng/Software/Python_Lib/new_pyctlib/pyctlib"] + sys.path


# import torchplus as tp
import torch
import torchplus as tp
print(tp.__file__)
from pyctlib import scope
import copy

tp.set_autodevice(False)
tp.manual_seed(0)
with scope("test tp, cpu"):
    t = tp.Tensor(3000, 400, requires_grad=True)
    a = t
    LP = tp.nn.Linear(400, 400)
    for t in range(10): a = LP(a)
    a.sum().backward()

torch.manual_seed(0)
with scope("test torch, cpu"):
    t_ = torch.Tensor(3000, 400).requires_grad_()
    a_ = t_
    LP_ = torch.nn.Linear(400, 400)
    for t in range(10): a_ = LP(a_)
    a_.sum().backward()

assert t.all_close(t_)
assert t._grad.all_close(t_._grad)



# with scope("test tp, gpu"):
#     a = tp.Tensor(3000, 400, requires_grad=True)
#     LP = tp.Tensor(400, 400, requires_grad=True)
#     for t in range(10): a = a @ LP
#     a.sum().backward()

# with scope("test torch, gpu"):
#     a = torch.Tensor(3000, 400).requires_grad_().cuda()
#     LP = torch.Tensor(400, 400).requires_grad_().cuda()
#     for t in range(10): a = a @ LP
#     a.sum().backward()

# with scope("test tp, gpu"):
#     a = tp.Tensor(3000, 400, requires_grad=True)
#     LP = tp.Tensor(400, 400, requires_grad=True)
#     for t in range(10): a = a @ LP
#     a.sum().backward()

# with scope("test torch, gpu"):
#     a = torch.Tensor(3000, 400).requires_grad_().cuda()
#     LP = torch.Tensor(400, 400).requires_grad_().cuda()
#     for t in range(10): a = a @ LP
#     a.sum().backward()
