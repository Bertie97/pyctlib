#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package torchplus.nn
##############################

import torch
# import torchplus as tp
import torch.nn as nn
import torch.nn.functional as F
from pyctlib import vector
import inspect

F_key = vector(dir(F)).filter(lambda x: not x.startswith("_")).filter(lambda x: not inspect.isclass(eval("F.{}".format(x)))).filter(lambda x: x[0].islower())

template = "{key} = F.{key}"

for key in F_key:
    exec(template.format(key=key))

__all__ = list(F_key)
