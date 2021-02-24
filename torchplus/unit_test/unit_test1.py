#! python3 -u
#  -*- coding: utf-8 -*-

##############################
## Author: Yuncheng Zhou
##############################

import torchplus as tp
import torch
from pyctlib import scope
a = tp.zeros(3, 2, dtype=torch.float)
import copy
with scope('transpose'):
	b = a.unsqueeze(0).unsqueeze(0).unsqueeze(0).T
with scope('tensor'):
	c = tp.Tensor(b)
with scope('add'):
	c = a+b
print(b)
print(a+b)