#! python3 -u
#  -*- coding: utf-8 -*-

##############################
## Author: Yuncheng Zhou
##############################

import sys
sys.path.append("../..")

# import torchplus as tp
import torch
import torchplus as tp
print(tp.__file__)
from pyctlib import scope
import copy

print(tp.stack(tp.zeros(3, 4), tp.ones(3, 4), dim={1}))

#tp.set_autodevice(False)
#tp.manual_seed(0)
#with scope("test tp, cpu"):
#    t = tp.randn([3000, 400], requires_grad=True)
#    a = t
#    LP = tp.nn.Linear(400, 400)
#    for _ in range(10): a = LP(a)
#    a.sum().backward()
#
#torch.manual_seed(0)
#with scope("test torch, cpu"):
#    t_ = torch.randn([3000, 400], requires_grad=True)
#    a_ = t_
#    LP_ = torch.nn.Linear(400, 400)
#    for _ in range(10): a_ = LP(a_)
#    a_.sum().backward()
#
#assert t.allclose(t_)
#assert t._grad.allclose(t_._grad)



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
