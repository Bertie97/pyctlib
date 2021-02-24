#! python3 -u
#  -*- coding: utf-8 -*-

##############################
## Author: Yuncheng Zhou
##############################

import sys

from import_pyctlib import *

# import torchplus as tp
import torch
from pyctlib import scope
import copy
# with scope("test 1"):
#     a = tp.Tensor(300, 400)
# with scope("test 2"):
#     a_ = torch.Tensor(300, 400)
# with scope('transpose'):
#     b = a.unsqueeze(0).unsqueeze(0).unsqueeze(0)
# with scope('transpose'):
#     b_ = a_.unsqueeze(0).unsqueeze(0).unsqueeze(0)
# with scope('add'):
#     c = a+b
# with scope('add'):
#     c = a_+b_

# print(b)
# print(a+b)

# def test(*args, t=1):
#     print(args, t)

a = tp.zeros(3, 2, dtype=torch.float)

LP = tp.nn.Linear(2, 5)

LP(a)
