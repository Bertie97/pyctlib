#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package torchplus.nn
##############################

import torchplus
import torch
from collections import OrderedDict
from ..tensor import _auto_device
from torch import _C

class Parameter(torchplus.Tensor):
    r"""A kind of Tensor that is to be considered a module parameter.

    Parameters are :class:`~torch.Tensor` subclasses, that have a
    very special property when used with :class:`Module` s - when they're
    assigned as Module attributes they are automatically added to the list of
    its parameters, and will appear e.g. in :meth:`~Module.parameters` iterator.
    Assigning a Tensor doesn't have such effect. This is because one might
    want to cache some temporary state, like last hidden state of the RNN, in
    the model. If there was no such class as :class:`Parameter`, these
    temporaries would get registered too.

    Arguments:
        data (Tensor): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient. See
            :ref:`excluding-subgraphs` for more details. Default: `True`
    """

    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = torchplus.Tensor()
        assert isinstance(data, torch.Tensor)
        self = torchplus.Tensor._make_subclass(cls, data, auto_device=torchplus.is_autodevice(), requires_grad=requires_grad)
        return self

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            # result = type(self)(self.data.clone(memory_format=torch.preserve_format), self.auto_device, self.requires_grad, self.batch_dimension)
            result = type(self)(self.data.clone(memory_format=torch.preserve_format), self.requires_grad)
            memo[id(self)] = result
            return result

    def __repr__(self):
        return 'Parameter containing:\n' + super(Parameter, self).__repr__()

    def __reduce_ex__(self, proto):
        # See Note [Don't serialize hooks]
        return (
            torch._utils._rebuild_parameter,
            (self.data, self.requires_grad, OrderedDict())
        )

    def __hash__(self): return id(self)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        ret = super().__torch_function__(func, types, args, kwargs)
        with _C.DisableTorchFunction():
            return _convert(ret, torchplus.Tensor)

def _convert(ret, cls):

    if isinstance(ret, torch.Tensor):
        ret = ret.as_subclass(cls)

    if isinstance(ret, (tuple, list)):
        # Also handles things like namedtuples
        ret = type(ret)(_convert(r, cls) for r in ret)

    return ret
