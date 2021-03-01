#! python3 -u
#  -*- coding: utf-8 -*-

##############################
## Author: Yiteng Zhang
##############################

import sys
# sys.path.append("/Users/zhangyiteng/Software/Python_Lib/new_pyctlib/pyctlib")
# sys.path.append("../..")
import copy
import torch
sys.path = ["../.."] + sys.path


import torchplus as tp
from pyctlib import scope

##############################
## Test CPU
##############################

with scope("tp, cpu, cat"):
    tp.cat([tp.zeros(300, 300), tp.zeros(300, 300)])

with scope("torch, cpu, cat"):
    torch.cat([torch.zeros(300, 300), torch.zeros(300, 300)])
