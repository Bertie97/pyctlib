#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package torchplus
##############################

from .tensor import *
from .tensorfunc import *
from . import nn
# import nn
import torch as basic_torch
from . import _jit_internal

distributed = basic_torch.distributed
autograd = basic_torch.autograd
