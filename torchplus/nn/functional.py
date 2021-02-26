#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package torchplus.nn
##############################

import torch
import torchplus as tp
import torch.nn as nn
import torch.nn.functional as F
from pyctlib import vector
import inspect

F_key = vector(dir(F)).filter(lambda x: not x.startswith("_")).filter(lambda x: not inspect.isclass(eval("F.{}".format(x)))).filter(lambda x: x[0].islower())

# template = "@return_tensor_wrapper\ndef {key}(*args, **kwargs): return F.{key}(*args, **kwargs)"
template = "{key} = F.{key}"

# F_key.apply(lambda x: exec(template.format(key=x)))

for key in F_key:
    exec(template.format(key=key))

__all__ = list(F_key)
