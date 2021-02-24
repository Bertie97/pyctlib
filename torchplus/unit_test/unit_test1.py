#! python3 -u
#  -*- coding: utf-8 -*-

##############################
## Author: Yuncheng Zhou
##############################

import sys
sys.path.insert(4, "/Users/zhangyiteng/Software/Python_Lib/pyctlib")

import torchplus as tp
import torch
from pyctlib import scope
a = tp.zeros(3, 2, dtype=torch.float)
import copy
with scope("test 1"):
    a = torch.Tensor(3, 4)
with scope("test 2"):
    a = tp.Tensor(3, 4)
with scope('transpose'):
    b = a.unsqueeze(0).unsqueeze(0).unsqueeze(0).T
with scope('tensor'):
    c = tp.Tensor(b)
with scope('add'):
    c = a+b
print(b)
print(a+b)
