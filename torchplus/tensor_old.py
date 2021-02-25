#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package torchplus
##############################

__all__ = """
    Device
    Tensor
    Size
""".split()

try:
    import torch
    import numpy as np
except ImportError:
    raise ImportError("'pyctlib.torchplus' cannot be used without dependency 'torch' and 'numpy'.")
import torch.nn as nn
import typing
import inspect
import builtins
from pyoverload import *
from pyctlib import raw_function, return_type_wrapper, touch
from functools import wraps
from typing import Union
from types import GeneratorType

import sys
from pyctlib.visual.debugger import profile
"""
from torchplus import Tensor
import torchplus as tp
"""

Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_auto_device = True

def set_autodevice(flag=True):
    _auto_device = flag

def unset_autodevice():
    _auto_device = False

def return_tensor_wrapper(*args_out):
    def output_wrapper(func, auto_device=_auto_device):
        func = raw_function(func)
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, Tensor):
               pass
            elif isinstance(result, torch.Tensor):
               result = Tensor(result, auto_device=auto_device, batch_dim=touch(lambda: args[0].batch_dimension) if isinstance(args[0], Tensor) else None)
            elif isinstance(result, tuple):
               result = tuple(Tensor(x) if isinstance(x, Tensor) else x for x in result)
            return result
        return wrapper

    if len(args_out) == 1 and (args_out[0] == True or args_out[0] == False):
        return lambda func: output_wrapper(func, args_out[0])
    else:
        return output_wrapper(args_out[0])

def _replace_key_with_sequence(original, sequence, key=-1):
    sequence = list(sequence)
    assert original.count(key) == len(sequence)
    result = list()
    index = 0
    for t in original:
        if t == key:
            result.append(sequence[index])
            index += 1
        else:
            result.append(t)
    return original.__class__(result)

def _replace_sequence_with_key(original, sequence, key=-1):
    sequence = list(sequence)
    result = list()
    index = 0
    for t in original:
        if index < len(sequence) and t == sequence[index]:
            result.append(key)
            index += 1
        else:
            result.append(t)
    assert index == len(sequence)
    return original.__class__(result)

def totensor(x) -> 'Tensor':
    if isinstance(x, Tensor):
        return x
    elif isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.tensor(x)
    elif isinstance(x, GeneratorType):
        return torch.tensor(list(x))
    else:
        return torch.tensor(x)

def tofloat(x):
    if isinstance(x, Tensor):
        return x._float_torch()
    elif isinstance(x, torch.Tensor):
        return x.float()

class GradWrapper:
    def __init__(self, name, gf):
        self.gf = gf
        self.__class__.__name__ = name
    def __str__(self): return "<{} object at {}>".format(self.__class__.__name__, '%x'%id(self.gf))
    __repr__ = __str__
    def __call__(self, *args, **kwargs): return self.gf(*args, **kwargs)

SizeRep = Tuple[IntScalar, List[0], (List|'set')<<IntScalar>>[1]]

class Size(tuple):

    NegSizeError = TypeError("Size cannot have negative values except -1 indicating arbitrary number. ")

    def __new__(cls, *args, batch_dim=None, channel_dim=None):
        if len(args) == 0:
            size = super().__new__(cls)
        elif len(args) == 1:
            if isinstance(args[0], builtins.int):
                size = super().__new__(cls, args)
            else:
                size = super().__new__(cls, args[0])
        else:
            size = super().__new__(cls, args)
        size._batch_dimension = batch_dim
        size._channel_dimension = channel_dim
        return size

    @property
    def batch_dimension(self): return touch(lambda: self._batch_dimension)

    @batch_dimension.setter
    def batch_dimension(self, value):
        self._batch_dimension = None
        if value is not None:
            if 0 <= value < self.ndim: self._batch_dimension = value
            elif value == self._channel_dimension: raise ValueError(f"batch_dimension can not be the same as channel_dimension: {value}")
            else: raise TypeError(f"batch_dimension should be a dimension index which is smaller than {self.ndim}")

    def batch_dimension_(self, value: IntScalar|Null):
        self.batch_dimension = value
        return self

    @property
    def batch_size(self):
        if self.batch_dimension is None:
            raise ValueError("There is no batch dimension provided. ")
        return self[self.batch_dimension]

    @property
    def channel_dimension(self): return touch(lambda: self._channel_dimension)

    @channel_dimension.setter
    def channel_dimension(self, value):
        self._channel_dimension = None
        if value is not None:
            if 0 <= value < self.ndim: self._channel_dimension = value
            elif value == self._batch_dimension: raise ValueError(f"channel_dimension can not be the same as batch_dimension: {value}")
            else: raise TypeError(f"channel_dimension should be a dimension index which is smaller than {self.dim()}")

    def channel_dimension_(self, value):
        self.channel_dimension = value
        return self

    @property
    def channel_size(self):
        if self.channel_dimension is None:
            raise ValueError("There is no channel dimension provided. ")
        return self[self.channel_dimension]

    @property
    def special(self): return sorted([x for x in [self.batch_dimension, self.channel_dimension] if x is not None])

    @property
    def bc(self): return [x for x in [self.batch_dimension, self.channel_dimension] if x is not None]

    @property
    def space(self):
        s = self.special
        if len(s) == 0: return self.remove_special()
        elif len(s) == 1: return (self[:s[0]] + self[s[0]+1:]).remove_special()
        return (self[:s[0]] + self[s[0]+1:s[1]] + self[s[1]+1:]).remove_special()

    @property
    def nele(self):
        if -1 in self: return -1
        p = 1
        for i in self: p *= i
        return p

    @property
    def ndim(self): return len(self)

    @property
    def nbatch(self): return self.batch_size

    @property
    def nchannel(self): return self.channel_size

    @property
    def nspace(self): return self.ndim - self.has_batch - self.has_channel

    @property
    def has_batch(self): return self.batch_dimension is not None

    @property
    def has_channel(self): return self.channel_dimension is not None

    @property
    def has_special(self): return self.has_batch or self.has_channel

    def remove_special(self):
        self.batch_dimension = None
        self.channel_dimension = None
        return self

    def copy(self): return Size(self.python_repr)

    def __add__(self, other):
        other = Size(other)
        if self.has_batch and other.has_batch: raise TypeError("Batch dimension conflict in addition. ")
        if self.has_channel and other.has_channel: raise TypeError("Channel dimension conflict in addition. ")
        ibatch = ichannel = None
        if self.has_batch:
            ibatch = self.batch_dimension
            other.batch_dimension = None
        elif other.has_batch: ibatch = other.batch_dimension + self.ndim
        if self.has_channel:
            ichannel = self.channel_dimension
            other.channel_dimension = None
        elif other.has_channel: ichannel = other.channel_dimension + self.ndim
        res = Size(tuple(self) + tuple(other))
        res.batch_dimension = ibatch
        res.channel_dimension = ichannel
        return res

    def __radd__(self, other): return Size(other) + self
    __iadd__ = __add__

    def __mul__(self, value):
        res = Size(tuple(self) * value)
        res.batch_dimension = self.batch_dimension
        res.channel_dimension = self.channel_dimension
        return res
    __imul__ = __rmul__ = __mul__

    @staticmethod
    def __op__(a, b, *, op):
        def getvalue(x, y):
            if x < 0 or y < 0: return -1
            z = op(x, y)
            if z < 0: raise Size.NegSizeError
            return builtins.int(z)
        if a.ndim == b.ndim:
            if a.has_batch and b.has_batch and a.batch_dimension != b.batch_dimension or\
                a.has_channel and b.has_channel and a.channel_dimension != b.channel_dimension:
                raise TypeError("Only Sizes with same batch/channel dimension can be operated. ")
            ibatch, ichannel = None, None
            if a.has_batch: ibatch = a.batch_dimension
            if b.has_batch: ibatch = b.batch_dimension
            if a.has_channel: ichannel = a.channel_dimension
            if b.has_channel: ichannel = b.channel_dimension
            c = Size(getvalue(x, y) for x, y in zip(a, b))
            c.batch_dimension = ibatch
            c.channel_dimension = ichannel
            return c
        if b.nspace == 0: pass
        elif a.nspace == 0: a, b = b, a
        elif a.nspace % b.nspace == 0: pass
        elif b.nspace % a.nspace == 0: a, b = b, a
        else: raise TypeError("Size object can not operate with another Size with different dimension, " +
            "please consider identify the batch/channel dimension or use + to concatenate. ")
        if b.nspace == 0: k = 1
        else: k = a.nspace // b.nspace
        if k == 1 and not a.has_special and b.has_special: a, b = b, a
        if not a.has_special:
            if b.has_special: raise TypeError("Please identify the batch/channel dimension for the longer size in operation. ")
            return Size(x + y for x, y in zip(a, tuple(b) * k))
        nbatch, nchannel = Size([touch(lambda: b.batch_size, 0)]), Size({touch(lambda: b.channel_size, 0)})
        b = b.space
        b = b * k
        s = a.special
        if len(s) == 0: pass
        elif len(s) == 1: b = b[:s[0]] + (nbatch if a.has_batch else nchannel,) + b[s[0]:]
        else:
            order = s == a.bc
            b = b[:s[0]] + (nbatch if order else nchannel) + b[s[0]:s[1]-1] + (nchannel if order else nbatch) + b[s[1]-1:]
        return op(a, b)

    def __lshift__(self, other): return self.__op__(self, Size(other), op=lambda x, y: x + y)
    __ilshift__ = __rlshift__ = __lshift__

    def __rshift__(self, other): return Size.__op__(self, Size(other), op=lambda x, y: x - y)

    def __rrshift__(self, other): return Size.__op__(Size(other), self, op=lambda x, y: x - y)
    __irshift__ = __rshift__

    def __pow__(self, other): return Size.__op__(self, Size(other), op=lambda x, y: x * y)
    __ipow__ = __rpow__ = __pow__

    def __floordiv__(self, other): return Size.__op__(self, Size(other), op=lambda x, y: x // y)

    def __rfloordiv__(self, other): return Size.__op__(Size(other), self, op=lambda x, y: x // y)
    __ifloordiv__ = __floordiv__

    def __matmul__(self, other):
        other = Size(other)
        if other.special == other.bc and self.special != self.bc or other.special != other.bc and self.special == self.bc:
            if self.nspace == 0 and self.ndim == 2: self = self[::-1]
            elif other.nspace == 0 and other.ndim == 2: other = other[::-1]
            else: raise TypeError(f"Batch and channel order not matched for {self} and {other}")
        if self.has_batch and other.has_batch:
            if self.batch_size in (1, -1): nbatch = other.batch_size
            elif other.batch_size in (1, -1): nbatch = self.batch_size
            elif self.batch_size == other.batch_size: nbatch = self.batch_size
            else: raise TypeError("Batch size should be the same or ±1")
            return self[:self.batch_dimension]@other[:other.batch_dimension] + \
                Size([nbatch]) + self[self.batch_dimension+1:]@other[other.batch_dimension+1:]
        if self.has_channel and other.has_channel:
            if self.channel_size in (1, -1): nchannel = other.channel_size
            elif other.channel_size in (1, -1): nchannel = self.channel_size
            elif self.channel_size == other.channel_size: nchannel = self.channel_size
            else: raise TypeError("Channel size should be the same or ±1")
            return self[:self.channel_dimension]@other[:other.channel_dimension] + \
                Size({nchannel}) + self[self.channel_dimension+1:]@other[other.channel_dimension+1:]
        if self.ndim < other.ndim: shorter, longer = self, other
        else: shorter, longer = other, self
        res = []
        for offset in builtins.range(longer.ndim - shorter.ndim + 1):
            i = 0
            j = offset
            res = list(longer[:offset].python_repr)
            while i < len(shorter) and j < len(longer):
                if i in shorter.special and j in longer.special:
                    if shorter.has_batch:
                        res.extend([[shorter[i]], {longer[j]}])
                        i += 1
                        j += 1
                    else:
                        res.extend([[longer[j]], {shorter[i]}])
                        i += 1
                        j += 1
                    continue
                if i == shorter.batch_dimension or j == longer.batch_dimension: nest = lambda x: [x]
                elif i == shorter.channel_dimension or j == longer.channel_dimension: nest = lambda x: {x}
                else: nest = lambda x: x
                if shorter[i] in (1, -1):
                    res.append(nest(longer[j]))
                    i += 1
                    j += 1
                elif longer[j] in (1, -1):
                    res.append(nest(shorter[i]))
                    i += 1
                    j += 1
                elif shorter[i] == longer[j]:
                    res.append(nest(shorter[i]))
                    i += 1
                    j += 1
                elif i in shorter.special:
                    res.append(shorter.python_repr[i])
                    i += 1
                elif j in longer.special:
                    res.append(longer.python_repr[j])
                    j += 1
                else: break
            else:
                if i < len(shorter): res.extend(shorter.python_repr[i:])
                elif j < len(longer): res.extend(longer.python_repr[j:])
                break
        else: raise TypeError(f"Unable to expand sizes {self} and {other}. ")
        return Size(*res)

    def __getitem__(self, k):
        if isoftype(k, IntScalar): return super().__getitem__(k)
        return Size(self.python_repr[k])

    @property
    def python_repr(self):
        args = list(self)
        if self.batch_dimension is not None:
            if isoftype(args[self.batch_dimension], IntScalar):
                args[self.batch_dimension] = [args[self.batch_dimension]]
            if not isoftype(args[self.batch_dimension], List[IntScalar][1]):
                raise TypeError(f"Batch dimension conflicts with notations: {tuple(self)} with batch {self.batch_dimension}" + 
                    (f" and channel {self.channel_dimension}" if self.has_channel else '') + '. ')
        if self.channel_dimension is not None:
            if isoftype(args[self.channel_dimension], IntScalar):
                args[self.channel_dimension] = {args[self.channel_dimension]}
            if not isoftype(args[self.channel_dimension], Type(set)[IntScalar][1]):
                raise TypeError(f"Channel dimension conflicts with notations: {tuple(self)} with " + 
                    (f"batch {self.batch_dimension} and " if self.has_batch else '') + f"channel {self.channel_dimension}. ")
        return tuple(args)

    def __str__(self):
        rep = tuple(self.python_repr)
        if len(rep) == 1: rep = str(rep).replace(',', '')
        return f"torchplus.Size{rep}"
    __repr__ = __str__

class Tensor(torch.Tensor):

    wrapper = return_tensor_wrapper

    @staticmethod
    def get_default_tensor_type():
        default_dtype = torch.get_default_dtype()
        if torch.Tensor(1).is_cuda:
            if default_dtype == torch.float32:
                return torch.cuda.FloatTensor
            if default_dtype == torch.float64:
                return torch.cuda.DoubleTensor
            if default_dtype == torch.float16:
                return torch.cuda.HalfTensor
        else:
            if default_dtype == torch.float32:
                return torch.FloatTensor
            if default_dtype == torch.float64:
                return torch.DoubleTensor
            if default_dtype == torch.float16:
                return torch.HalfTensor

    @staticmethod
    def _make_subclass(cls, data, auto_device=_auto_device, requires_grad=None):
        if isinstance(data, torch.Tensor):
            if auto_device:
                data = data.to(Device)
        else:
            if auto_device:
                data = torch.as_tensor(data, device=Device)
            else:
                data = torch.as_tensor(data)
        if requires_grad is None:
            requires_grad = data.requires_grad
        tensor = torch.Tensor._make_subclass(cls, data, requires_grad)
        return tensor

    # @profile
    def __new__(cls, *args, auto_device=_auto_device, requires_grad=None, batch_dim=None, channel_dim=None):


        if len(args) == 1 and isinstance(args, Tensor):
            self = args[0]
            if requires_grad is not None:
                self.requires_grad = requires_grad
            if auto_device is True:
                self = self.to(Device)
            return self

        # print("__new__ fucntion called", args)

        if len(args) > 1 or len(args) == 0:
            data = torch.Tensor(*args)
        else:
            if isinstance(args[0], Size):
                if auto_device is True:
                    data = torch.Tensor(device=Device)
                else:
                    data = torch.Tensor()
            elif isinstance(args[0], builtins.int):
                if auto_device is True:
                    data = torch.Tensor(args[0], device=Device)
                else:
                    data = torch.Tensor(args[0])
            else:
                if auto_device:
                    data = torch.as_tensor(args[0], device=Device)
                else:
                    data = torch.as_tensor(args[0])

        if requires_grad is None:
            requires_grad = data.requires_grad

        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        # self = Tensor._make_subclass(cls, data, auto_device=auto_device, requires_grad=requires_grad)

        # _batch_dimension = None
        # _channel_dimension = None
        # if isinstance(args[0], Size):
        #     _batch_dimension = args[0].batch_dimension
        #     _channel_dimension = args[0].channel_dimension
        # if batch_dim:
        #     _batch_dimension = batch_dim
        # if channel_dim:
        #     _channel_dimension = channel_dim

        # self._shape = Size(data.shape, batch_dim=_batch_dimension, channel_dim=_channel_dimension)

        return self

    # @property
    # def shape(self):
    #     return self._shape
        # return super().shape
        # shape = Size(*super().shape)
        # batch_dim = touch(lambda: self._batch_dimension)
        # channel_dim = touch(lambda: self._channel_dimension)
        # # if batch_dim is not None: shape.batch_dimension = batch_dim
        # # if channel_dim is not None: shape.channel_dimension = channel_dim
        # if touch(lambda: self.names):
        #     if not shape.has_batch:
        #         isbatch = ['batch' in x for x in self.names if x]
        #         if builtins.any(isbatch):
        #             ibatch = isbatch.index(True)
        #             self.batch_dim = ibatch
        #             shape.batch_dimension = ibatch
        #     if not shape.has_channel:
        #         ischannel = ['channel' in x for x in self.names if x]
        #         if builtins.any(ischannel):
        #             ichannel = ischannel.index(True)
        #             self.channel_dim = ichannel
        #             shape.channel_dimension = ichannel
        # return shape

    # @property
    # def batch_dimension(self):
    #     return self.shape.batch_dimension

    # @batch_dimension.setter
    # def batch_dimension(self, value):
    #     self._batch_dimension = value

    # def batch_dimension_(self, value):
    #     self.batch_dimension = value
    #     return self

    # @property
    # def batch_size(self): return self.shape.batch_size

    # @property
    # def channel_dimension(self):
    #     return self.shape.channel_dimension

    # @channel_dimension.setter
    # def channel_dimension(self, value):
    #     self._channel_dimension = value

    # def channel_dimension_(self, value):
    #     self.channel_dimension = value
    #     return self

    # @property
    # def channel_size(self): return self.shape.channel_size

    # @property
    # def space(self): return self.shape.space
    # @property
    # def ndim(self): return self.shape.ndim
    # @property
    # def nele(self): return self.shape.nele
    # @property
    # def numel(self): return self.shape.nele
    # @property
    # def nbatch(self): return self.batch_size
    # @property
    # def nchannel(self): return self.channel_size
    # @property
    # def nspace(self): return self.ndim - self.has_batch - self.has_channel
    # @property
    # def has_batch(self): return self.batch_dimension is not None
    # @property
    # def has_channel(self): return self.channel_dimension is not None
    # @property
    # def has_special(self): return self.has_batch or self.has_channel

    # def remove_special(self):
    #     self.batch_dimension = None
    #     self.channel_dimension = None

    @staticmethod
    def tensor_type(dtype, is_cuda=False):
        if not is_cuda:
            if dtype == torch.float32:
                return torch.Tensor
            elif dtype == torch.float64:
                return torch.DoubleTensor
            elif dtype == torch.float16:
                return torch.HalfTensor
            elif dtype == torch.bfloat16:
                return torch.BFloat16Tensor
            elif dtype == torch.int64:
                return torch.LongTensor
            elif dtype == torch.int16:
                return torch.ShortTensor
            elif dtype == torch.int8:
                return torch.ByteTensor
            elif dtype == torch.int32:
                return torch.IntTensor
            elif dtype == torch.bool:
                return torch.BoolTensor
            else:
                return torch.Tensor
        else:
            if dtype == torch.float32:
                return torch.cuda.FloatTensor
            elif dtype == torch.float64:
                return torch.cuda.DoubleTensor
            elif dtype == torch.float16:
                return torch.cuda.HalfTensor
            elif dtype == torch.bfloat16:
                return torch.cuda.BFloat16Tensor
            elif dtype == torch.int64:
                return torch.cuda.LongTensor
            elif dtype == torch.int16:
                return torch.cuda.ShortTensor
            elif dtype == torch.int8:
                return torch.cuda.ByteTensor
            elif dtype == torch.int32:
                return torch.cuda.IntTensor
            elif dtype == torch.bool:
                return torch.cuda.BoolTensor

    def tensor(self):
        if self.dim() > 0:
            return Tensor.tensor_type(self.dtype, self.is_cuda)(self.data)
        else:
            return Tensor.tensor_type(self.dtype, self.is_cuda)([self.item()]).sum()

    def numpy(self): return super(torch.Tensor, self.cpu().detach()).numpy()

    def dim(self):
        return super().dim()

    # def size(self, *k):
    #     if len(k) == 0:
    #         return self.shape
    #     i = [(self.names.index(x) if x in self.names else None) if isoftype(x, str) else x for x in k]
    #     if None in i:
    #         return super().size(k[i.index(None)])
    #     if len(i) == 1:
    #         return self.shape[i[0]]
    #     return tuple(self.shape[x] for x in i)

    # def numel(self): return self.nele

    def expand_to(self, target):
        target = Size(target)
        if target.special == target.bc and self.shape.special != self.shape.bc or target.special != target.bc and self.shape.special == self.shape.bc:
            if self.nspace == 0 and self.ndim == 2: self = self[::-1]
            else: raise TypeError(f"Batch and channel order not matched for {self.shape} and {target}")
        axis_map = list(builtins.range(self.ndim))
        p = 0
        for i, s in enumerate(self.shape):
            if i == self.batch_dimension:
                axis_map[i] = target.batch_dimension
                p = target.batch_dimension + 1
            elif i == self.channel_dimension:
                axis_map[i] = target.channel_dimension
                p = target.channel_dimension + 1
            elif s in (1, -1):
                axis_map[i] = p
                p += 1
            else:
                while p < target.ndim and target[p] != s: p += 1
                axis_map[i] = p
                p += 1
            if p >= target.ndim  + 1: raise TypeError(f"Unable to expand sizes {self.shape} to {target}. ")
        return self.unsqueeze_to(target, axis_map)


    # @overload
    # def unsqueeze_to(self, target: Array | 'Tensor', axis_place: List):
    #     return self.expand_to(target.shape, axis_place)

    # @overload
    # def unsqueeze_to(self, target: Tuple[IntScalar] | 'Size', axis_place: List):
    #     target = Size(target)
    #     if target.has_batch and self.has_batch and axis_place[self.batch_dimension] != target.batch_dimension:
    #         raise TypeError("Conflict of batch dimension in 'unsqueeze_to'. ")
    #     if target.has_channel and self.has_channel and axis_place[self.channel_dimension] != target.channel_dimension:
    #         raise TypeError("Conflict of channel dimension in 'unsqueeze_to'. ")
    #     new_size = list(target)
    #     for i in builtins.range(len(new_size)):
    #         if i not in axis_place or self.shape[axis_place.index(i)] in (1, -1):
    #             new_size[i] = 1
    #     return self.view(*new_size)

    @return_tensor_wrapper
    def sample(self, dim: int = None, number: int = 1, random: bool = True) -> 'Tensor':
        """
        sample(self, dim: int = self.batch_dimension, numbder: int = 1, random: bool = True) -> Tensor

        Sample a few subspaces from a given dimension.
        data.sample(2, 1, random=False) is equivalant to data[:, :, 0, ...].
        """
        if dim is None: dim = self.batch_dimension
        if dim is None: raise TypeError("Argument 'dim' needed for sampling Tensors with no batch dimension. ")
        sample_indices = [slice(None)] * self.dim()
        if self.shape[dim] < number: raise TypeError(f"Too many elements needed to be sampled from dimension {dim}")
        if random:
            import random
            samples = random.choice(range(self.shape[dim]), k = number)
        else: samples = list(range(number))
        sample_indices[dim] = samples
        output_tensor = Tensor(self[tuple(sample_indices)])
        output_tensor.indices = samples
        return output_tensor

    @property
    @return_tensor_wrapper
    def T(self: 'Tensor') -> 'Tensor':
        return super().T
        # if not self.has_special: return super().T
        # s = self.shape.special
        # if len(s) == 1: permute_dim = tuple(range(s[0]))[::-1] + (s[0],) + tuple(range(s[0]+1, self.ndim))[::-1]
        # elif len(s) == 2: permute_dim = tuple(range(s[0]))[::-1] + (s[0],) + tuple(range(s[0]+1, s[1]))[::-1] + (s[1],) + tuple(range(s[1]+1, self.ndim))[::-1]
        # return self.permute(permute_dim)

    @return_tensor_wrapper
    def abs(self) -> 'Tensor':
        """
        abs(input, out=None) -> Tensor

        Computes the element-wise absolute value of the given :attr:`input` tensor.

        .. math::
            \text{out}_{i} = |\text{input}_{i}|

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> torch.abs(torch.tensor([-1, -2, 3]))
            tensor([ 1,  2,  3])
        """
        return super().abs()

    @return_tensor_wrapper
    def abs_(self) -> 'Tensor':
        """
        abs_() -> Tensor

        In-place version of :meth:`~Tensor.abs`
        """
        return super().abs_()

    @return_tensor_wrapper
    def absolute(self) -> 'Tensor':
        """
        absolute() -> Tensor

        Alias for :func:`abs`
        """
        return super().absolute()

    @return_tensor_wrapper
    def absolute_(self) -> 'Tensor':
        """
        absolute_() -> Tensor

        In-place version of :meth:`~Tensor.absolute`
        Alias for :func:`abs_`
        """
        return super().absolute_()

    @return_tensor_wrapper
    def acos(self) -> 'Tensor':
        """
        acos(input, out=None) -> Tensor

        Returns a new tensor with the arccosine  of the elements of :attr:`input`.

        .. math::
            \text{out}_{i} = \cos^{-1}(\text{input}_{i})

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4)
            >>> a
            tensor([ 0.3348, -0.5889,  0.2005, -0.1584])
            >>> torch.acos(a)
            tensor([ 1.2294,  2.2004,  1.3690,  1.7298])
        """
        return super().acos()

    @return_tensor_wrapper
    def acos_(self) -> 'Tensor':
        """
        acos_() -> Tensor

        In-place version of :meth:`~Tensor.acos`
        """
        return super().acos_()

    @return_tensor_wrapper
    def acosh(self) -> 'Tensor':
        """
        acosh(input, out=None) -> Tensor

        Returns a new tensor with the inverse hyperbolic cosine of the elements of :attr:`input`.

        Note:
            The domain of the inverse hyperbolic cosine is `[1, inf)` and values outside this range
            will be mapped to ``NaN``, except for `+ INF` for which the output is mapped to `+ INF`.

        .. math::
            \text{out}_{i} = \cosh^{-1}(\text{input}_{i})

        Args:
            input (Tensor): the input tensor.

        Keyword arguments:
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4).uniform_(1, 2)
            >>> a
            tensor([ 1.3192, 1.9915, 1.9674, 1.7151 ])
            >>> torch.acosh(a)
            tensor([ 0.7791, 1.3120, 1.2979, 1.1341 ])
        """
        return super().acosh()

    @return_tensor_wrapper
    def acosh_(self) -> 'Tensor':
        """
        acosh_() -> Tensor

        In-place version of :meth:`~Tensor.acosh`
        """
        return super().acosh_()

    @return_tensor_wrapper
    def add(self, other, *, alpha=1) -> 'Tensor':
        """
        add(input, other, out=None)

        Adds the scalar :attr:`other` to each element of the input :attr:`input`
        and returns a new resulting tensor.

        .. math::
            \text{out} = \text{input} + \text{other}

        If :attr:`input` is of type FloatTensor or DoubleTensor, :attr:`other` must be
        a real number, otherwise it should be an integer.

        Args:
            input (Tensor): the input tensor.
            value (Number): the number to be added to each element of :attr:`input`

        Keyword arguments:
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4)
            >>> a
            tensor([ 0.0202,  1.0985,  1.3506, -0.6056])
            >>> torch.add(a, 20)
            tensor([ 20.0202,  21.0985,  21.3506,  19.3944])

        .. function:: add(input, other, *, alpha=1, out=None)

        Each element of the tensor :attr:`other` is multiplied by the scalar
        :attr:`alpha` and added to each element of the tensor :attr:`input`.
        The resulting tensor is returned.

        The shapes of :attr:`input` and :attr:`other` must be
        :ref:`broadcastable <broadcasting-semantics>`.

        .. math::
            \text{out} = \text{input} + \text{alpha} \times \text{other}

        If :attr:`other` is of type FloatTensor or DoubleTensor, :attr:`alpha` must be
        a real number, otherwise it should be an integer.

        Args:
            input (Tensor): the first input tensor
            other (Tensor): the second input tensor
            alpha (Number): the scalar multiplier for :attr:`other`

        Keyword arguments:
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4)
            >>> a
            tensor([-0.9732, -0.3497,  0.6245,  0.4022])
            >>> b = torch.randn(4, 1)
            >>> b
            tensor([[ 0.3743],
                    [-1.7724],
                    [-0.5811],
                    [-0.8017]])
            >>> torch.add(a, b, alpha=10)
            tensor([[  2.7695,   3.3930,   4.3672,   4.1450],
                    [-18.6971, -18.0736, -17.0994, -17.3216],
                    [ -6.7845,  -6.1610,  -5.1868,  -5.4090],
                    [ -8.9902,  -8.3667,  -7.3925,  -7.6147]])
        """
        return super().add(other, alpha=alpha)

    @return_tensor_wrapper
    def add_(self, other, *, alpha=1) -> 'Tensor':
        """
        add_(other, *, alpha=1) -> Tensor

        In-place version of :meth:`~Tensor.add`
        """
        return super().add_(other, alpha=alpha)

    @return_tensor_wrapper
    def addbmm(self, batch1, batch2, *, beta=1, alpha=1) -> 'Tensor':
        """
        addbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None) -> Tensor

        Performs a batch matrix-matrix product of matrices stored
        in :attr:`batch1` and :attr:`batch2`,
        with a reduced add step (all matrix multiplications get accumulated
        along the first dimension).
        :attr:`input` is added to the final result.

        :attr:`batch1` and :attr:`batch2` must be 3-D tensors each containing the
        same number of matrices.

        If :attr:`batch1` is a :math:`(b \times n \times m)` tensor, :attr:`batch2` is a
        :math:`(b \times m \times p)` tensor, :attr:`input` must be
        :ref:`broadcastable <broadcasting-semantics>` with a :math:`(n \times p)` tensor
        and :attr:`out` will be a :math:`(n \times p)` tensor.

        .. math::
            out = \beta\ \text{input} + \alpha\ (\sum_{i=0}^{b-1} \text{batch1}_i \mathbin{@} \text{batch2}_i)

        For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and :attr:`alpha`
        must be real numbers, otherwise they should be integers.

        Args:
            batch1 (Tensor): the first batch of matrices to be multiplied
            batch2 (Tensor): the second batch of matrices to be multiplied
            beta (Number, optional): multiplier for :attr:`input` (:math:`\beta`)
            input (Tensor): matrix to be added
            alpha (Number, optional): multiplier for `batch1 @ batch2` (:math:`\alpha`)
            out (Tensor, optional): the output tensor.

        Example::

            >>> M = torch.randn(3, 5)
            >>> batch1 = torch.randn(10, 3, 4)
            >>> batch2 = torch.randn(10, 4, 5)
            >>> torch.addbmm(M, batch1, batch2)
            tensor([[  6.6311,   0.0503,   6.9768, -12.0362,  -2.1653],
                    [ -4.8185,  -1.4255,  -6.6760,   8.9453,   2.5743],
                    [ -3.8202,   4.3691,   1.0943,  -1.1109,   5.4730]])
        """
        return super().addbmm(batch1, batch2, beta=beta, alpha=alpha)

    @return_tensor_wrapper
    def addbmm_(self, batch1, batch2, *, beta=1, alpha=1) -> 'Tensor':
        """
        addbmm_(batch1, batch2, *, beta=1, alpha=1) -> Tensor

        In-place version of :meth:`~Tensor.addbmm`
        """
        return super().addbmm_(batch1, batch2, beta=beta, alpha=alpha)

    @return_tensor_wrapper
    def addcdiv(self, tensor1, tensor2, *, value=1) -> 'Tensor':
        """
        addcdiv(input, tensor1, tensor2, *, value=1, out=None) -> Tensor

        Performs the element-wise division of :attr:`tensor1` by :attr:`tensor2`,
        multiply the result by the scalar :attr:`value` and add it to :attr:`input`.

        .. warning::
            Integer division with addcdiv is no longer supported, and in a future release
            addcdiv will perform a true division of :attr:`tensor1` and :attr:`tensor2`.
            The historic addcdiv behavior can be implemented using :func:`floor_divide`
            for integral inputs
            (:attr:`input` + :attr:`value` * :attr:`tensor1` // :attr:`tensor2`)
            and :func:`div` for float inputs
            (:attr:`input` + :attr:`value` * :attr:`tensor1` / :attr:`tensor2`).
            The future addcdiv behavior can be implemented with :func:`true_divide`
            (:attr:`input` + :attr:`value` * torch.true_divide(:attr:`tensor1`,
            :attr:`tensor2`).

        .. math::
            \text{out}_i = \text{input}_i + \text{value} \times \frac{\text{tensor1}_i}{\text{tensor2}_i}


        The shapes of :attr:`input`, :attr:`tensor1`, and :attr:`tensor2` must be
        :ref:`broadcastable <broadcasting-semantics>`.

        For inputs of type `FloatTensor` or `DoubleTensor`, :attr:`value` must be
        a real number, otherwise an integer.

        Args:
            input (Tensor): the tensor to be added
            tensor1 (Tensor): the numerator tensor
            tensor2 (Tensor): the denominator tensor
            value (Number, optional): multiplier for :math:`\text{tensor1} / \text{tensor2}`
            out (Tensor, optional): the output tensor.

        Example::

            >>> t = torch.randn(1, 3)
            >>> t1 = torch.randn(3, 1)
            >>> t2 = torch.randn(1, 3)
            >>> torch.addcdiv(t, t1, t2, value=0.1)
            tensor([[-0.2312, -3.6496,  0.1312],
                    [-1.0428,  3.4292, -0.1030],
                    [-0.5369, -0.9829,  0.0430]])
        """
        return super().addcdiv(tensor1, tensor2, value=value)

    @return_tensor_wrapper
    def addcdiv_(self, tensor1, tensor2, *, value=1) -> 'Tensor':
        """
        addcdiv_(tensor1, tensor2, *, value=1) -> Tensor

        In-place version of :meth:`~Tensor.addcdiv`
        """
        return super().addcdiv_(tensor1, tensor2, value=value)

    @return_tensor_wrapper
    def addcmul(self, tensor1, tensor2, *, value=1) -> 'Tensor':
        """
        addcmul(input, tensor1, tensor2, *, value=1, out=None) -> Tensor

        Performs the element-wise multiplication of :attr:`tensor1`
        by :attr:`tensor2`, multiply the result by the scalar :attr:`value`
        and add it to :attr:`input`.

        .. math::
            \text{out}_i = \text{input}_i + \text{value} \times \text{tensor1}_i \times \text{tensor2}_i

        The shapes of :attr:`tensor`, :attr:`tensor1`, and :attr:`tensor2` must be
        :ref:`broadcastable <broadcasting-semantics>`.

        For inputs of type `FloatTensor` or `DoubleTensor`, :attr:`value` must be
        a real number, otherwise an integer.

        Args:
            input (Tensor): the tensor to be added
            tensor1 (Tensor): the tensor to be multiplied
            tensor2 (Tensor): the tensor to be multiplied
            value (Number, optional): multiplier for :math:`tensor1 .* tensor2`
            out (Tensor, optional): the output tensor.

        Example::

            >>> t = torch.randn(1, 3)
            >>> t1 = torch.randn(3, 1)
            >>> t2 = torch.randn(1, 3)
            >>> torch.addcmul(t, t1, t2, value=0.1)
            tensor([[-0.8635, -0.6391,  1.6174],
                    [-0.7617, -0.5879,  1.7388],
                    [-0.8353, -0.6249,  1.6511]])
        """
        return super().addcmul(tensor1, tensor2, value=value)

    @return_tensor_wrapper
    def addcmul_(self, tensor1, tensor2, *, value=1) -> 'Tensor':
        """
        addcmul_(tensor1, tensor2, *, value=1) -> Tensor

        In-place version of :meth:`~Tensor.addcmul`
        """
        return super().addcmul_(tensor1, tensor2, value=value)

    @return_tensor_wrapper
    def addmm(self, mat1, mat2, *, beta=1, alpha=1) -> 'Tensor':
        """
        addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None) -> Tensor

        Performs a matrix multiplication of the matrices :attr:`mat1` and :attr:`mat2`.
        The matrix :attr:`input` is added to the final result.

        If :attr:`mat1` is a :math:`(n \times m)` tensor, :attr:`mat2` is a
        :math:`(m \times p)` tensor, then :attr:`input` must be
        :ref:`broadcastable <broadcasting-semantics>` with a :math:`(n \times p)` tensor
        and :attr:`out` will be a :math:`(n \times p)` tensor.

        :attr:`alpha` and :attr:`beta` are scaling factors on matrix-vector product between
        :attr:`mat1` and :attr:`mat2` and the added matrix :attr:`input` respectively.

        .. math::
            \text{out} = \beta\ \text{input} + \alpha\ (\text{mat1}_i \mathbin{@} \text{mat2}_i)

        For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and
        :attr:`alpha` must be real numbers, otherwise they should be integers.

        Args:
            input (Tensor): matrix to be added
            mat1 (Tensor): the first matrix to be multiplied
            mat2 (Tensor): the second matrix to be multiplied
            beta (Number, optional): multiplier for :attr:`input` (:math:`\beta`)
            alpha (Number, optional): multiplier for :math:`mat1 @ mat2` (:math:`\alpha`)
            out (Tensor, optional): the output tensor.

        Example::

            >>> M = torch.randn(2, 3)
            >>> mat1 = torch.randn(2, 3)
            >>> mat2 = torch.randn(3, 3)
            >>> torch.addmm(M, mat1, mat2)
            tensor([[-4.8716,  1.4671, -1.3746],
                    [ 0.7573, -3.9555, -2.8681]])
        """
        return super().addmm(mat1, mat2, beta=beta, alpha=alpha)

    @return_tensor_wrapper
    def addmm_(self, mat1, mat2, *, beta=1, alpha=1) -> 'Tensor':
        """
        addmm_(mat1, mat2, *, beta=1, alpha=1) -> Tensor

        In-place version of :meth:`~Tensor.addmm`
        """
        return super().addmm_(mat1, mat2, beta=beta, alpha=alpha)

    @return_tensor_wrapper
    def addmv(self, mat, vec, *, beta=1, alpha=1) -> 'Tensor':
        """
        addmv(input, mat, vec, *, beta=1, alpha=1, out=None) -> Tensor

        Performs a matrix-vector product of the matrix :attr:`mat` and
        the vector :attr:`vec`.
        The vector :attr:`input` is added to the final result.

        If :attr:`mat` is a :math:`(n \times m)` tensor, :attr:`vec` is a 1-D tensor of
        size `m`, then :attr:`input` must be
        :ref:`broadcastable <broadcasting-semantics>` with a 1-D tensor of size `n` and
        :attr:`out` will be 1-D tensor of size `n`.

        :attr:`alpha` and :attr:`beta` are scaling factors on matrix-vector product between
        :attr:`mat` and :attr:`vec` and the added tensor :attr:`input` respectively.

        .. math::
            \text{out} = \beta\ \text{input} + \alpha\ (\text{mat} \mathbin{@} \text{vec})

        For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and
        :attr:`alpha` must be real numbers, otherwise they should be integers

        Args:
            input (Tensor): vector to be added
            mat (Tensor): matrix to be multiplied
            vec (Tensor): vector to be multiplied
            beta (Number, optional): multiplier for :attr:`input` (:math:`\beta`)
            alpha (Number, optional): multiplier for :math:`mat @ vec` (:math:`\alpha`)
            out (Tensor, optional): the output tensor.

        Example::

            >>> M = torch.randn(2)
            >>> mat = torch.randn(2, 3)
            >>> vec = torch.randn(3)
            >>> torch.addmv(M, mat, vec)
            tensor([-0.3768, -5.5565])
        """
        return super().addmv(mat, vec, beta=beta, alpha=alpha)

    @return_tensor_wrapper
    def addmv_(self, mat, vec, *, beta=1, alpha=1) -> 'Tensor':
        """
        addmv_(mat, vec, *, beta=1, alpha=1) -> Tensor

        In-place version of :meth:`~Tensor.addmv`
        """
        return super().addmv_(mat, vec, beta=beta, alpha=alpha)

    @return_tensor_wrapper
    def addr(self, vec1, vec2, *, beta=1, alpha=1) -> 'Tensor':
        """
        addr(input, vec1, vec2, *, beta=1, alpha=1, out=None) -> Tensor

        Performs the outer-product of vectors :attr:`vec1` and :attr:`vec2`
        and adds it to the matrix :attr:`input`.

        Optional values :attr:`beta` and :attr:`alpha` are scaling factors on the
        outer product between :attr:`vec1` and :attr:`vec2` and the added matrix
        :attr:`input` respectively.

        .. math::
            \text{out} = \beta\ \text{input} + \alpha\ (\text{vec1} \otimes \text{vec2})

        If :attr:`vec1` is a vector of size `n` and :attr:`vec2` is a vector
        of size `m`, then :attr:`input` must be
        :ref:`broadcastable <broadcasting-semantics>` with a matrix of size
        :math:`(n \times m)` and :attr:`out` will be a matrix of size
        :math:`(n \times m)`.

        For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and
        :attr:`alpha` must be real numbers, otherwise they should be integers

        Args:
            input (Tensor): matrix to be added
            vec1 (Tensor): the first vector of the outer product
            vec2 (Tensor): the second vector of the outer product
            beta (Number, optional): multiplier for :attr:`input` (:math:`\beta`)
            alpha (Number, optional): multiplier for :math:`\text{vec1} \otimes \text{vec2}` (:math:`\alpha`)
            out (Tensor, optional): the output tensor.

        Example::

            >>> vec1 = torch.arange(1., 4.)
            >>> vec2 = torch.arange(1., 3.)
            >>> M = torch.zeros(3, 2)
            >>> torch.addr(M, vec1, vec2)
            tensor([[ 1.,  2.],
                    [ 2.,  4.],
                    [ 3.,  6.]])
        """
        return super().addr(vec1, vec2, beta=beta, alpha=alpha)

    @return_tensor_wrapper
    def addr_(self, vec1, vec2, *, beta=1, alpha=1) -> 'Tensor':
        """
        addr_(vec1, vec2, *, beta=1, alpha=1) -> Tensor

        In-place version of :meth:`~Tensor.addr`
        """
        return super().addr_(vec1, vec2, beta=beta, alpha=alpha)

    @return_tensor_wrapper
    def align_as(self, other) -> 'Tensor':
        """
        align_as(other) -> Tensor

        Permutes the dimensions of the :attr:`self` tensor to match the dimension order
        in the :attr:`other` tensor, adding size-one dims for any new names.

        This operation is useful for explicit broadcasting by names (see examples).

        All of the dims of :attr:`self` must be named in order to use this method.
        The resulting tensor is a view on the original tensor.

        All dimension names of :attr:`self` must be present in ``other.names``.
        :attr:`other` may contain named dimensions that are not in ``self.names``;
        the output tensor has a size-one dimension for each of those new names.

        To align a tensor to a specific order, use :meth:`~Tensor.align_to`.

        Examples::

            # Example 1: Applying a mask
            >>> mask = torch.randint(2, [127, 128], dtype=torch.bool).refine_names('W', 'H')
            >>> imgs = torch.randn(32, 128, 127, 3, names=('N', 'H', 'W', 'C'))
            >>> imgs.masked_fill_(mask.align_as(imgs), 0)


            # Example 2: Applying a per-channel-scale
            >>> def scale_channels(input, scale):
            >>>    scale = scale.refine_names('C')
            >>>    return input * scale.align_as(input)

            >>> num_channels = 3
            >>> scale = torch.randn(num_channels, names=('C',))
            >>> imgs = torch.rand(32, 128, 128, num_channels, names=('N', 'H', 'W', 'C'))
            >>> more_imgs = torch.rand(32, num_channels, 128, 128, names=('N', 'C', 'H', 'W'))
            >>> videos = torch.randn(3, num_channels, 128, 128, 128, names=('N', 'C', 'H', 'W', 'D'))

            # scale_channels is agnostic to the dimension order of the input
            >>> scale_channels(imgs, scale)
            >>> scale_channels(more_imgs, scale)
            >>> scale_channels(videos, scale)

        .. warning::
            The named tensor API is experimental and subject to change.
        """
        return super().align_as(other)

    @return_tensor_wrapper
    def allclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False) -> 'Tensor':
        """
        allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False) -> bool

        This function checks if all :attr:`input` and :attr:`other` satisfy the condition:

        .. math::
            \lvert \text{input} - \text{other} \rvert \leq \texttt{atol} + \texttt{rtol} \times \lvert \text{other} \rvert

        elementwise, for all elements of :attr:`input` and :attr:`other`. The behaviour of this function is analogous to
        `numpy.allclose <https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html>`_

        Args:
            input (Tensor): first tensor to compare
            other (Tensor): second tensor to compare
            atol (float, optional): absolute tolerance. Default: 1e-08
            rtol (float, optional): relative tolerance. Default: 1e-05
            equal_nan (bool, optional): if ``True``, then two ``NaN`` s will be considered equal. Default: ``False``

        Example::

            >>> torch.allclose(torch.tensor([10000., 1e-07]), torch.tensor([10000.1, 1e-08]))
            False
            >>> torch.allclose(torch.tensor([10000., 1e-08]), torch.tensor([10000.1, 1e-09]))
            True
            >>> torch.allclose(torch.tensor([1.0, float('nan')]), torch.tensor([1.0, float('nan')]))
            False
            >>> torch.allclose(torch.tensor([1.0, float('nan')]), torch.tensor([1.0, float('nan')]), equal_nan=True)
            True
        """
        return super().allclose(other, rtol=rtol, atol=atol, equal_nan=equal_nan)

    @return_tensor_wrapper
    def angle(self) -> 'Tensor':
        """
        angle(input, out=None) -> Tensor

        Computes the element-wise angle (in radians) of the given :attr:`input` tensor.

        .. math::
            \text{out}_{i} = angle(\text{input}_{i})

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> torch.angle(torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j]))*180/3.14159
            tensor([ 135.,  135,  -45])
        """
        return super().angle()

    @return_tensor_wrapper
    def apply_(self, callable) -> 'Tensor':
        """
        apply_(callable) -> Tensor

        Applies the function :attr:`callable` to each element in the tensor, replacing
        each element with the value returned by :attr:`callable`.

        .. note::

            This function only works with CPU tensors and should not be used in code
            sections that require high performance.
        """
        return super().apply_(callable)

    @return_tensor_wrapper
    def argmax(self, dim=None, keepdim=False) -> 'LongTensor':
        """
        argmax(input) -> LongTensor

        Returns the indices of the maximum value of all elements in the :attr:`input` tensor.

        This is the second value returned by :meth:`torch.max`. See its
        documentation for the exact semantics of this method.

        Args:
            input (Tensor): the input tensor.

        Example::

            >>> a = torch.randn(4, 4)
            >>> a
            tensor([[ 1.3398,  0.2663, -0.2686,  0.2450],
                    [-0.7401, -0.8805, -0.3402, -1.1936],
                    [ 0.4907, -1.3948, -1.0691, -0.3132],
                    [-1.6092,  0.5419, -0.2993,  0.3195]])
            >>> torch.argmax(a)
            tensor(0)

        .. function:: argmax(input, dim, keepdim=False) -> LongTensor

        Returns the indices of the maximum values of a tensor across a dimension.

        This is the second value returned by :meth:`torch.max`. See its
        documentation for the exact semantics of this method.

        Args:
            input (Tensor): the input tensor.
            dim (int): the dimension to reduce. If ``None``, the argmax of the flattened input is returned.
            keepdim (bool): whether the output tensor has :attr:`dim` retained or not. Ignored if ``dim=None``.

        Example::

            >>> a = torch.randn(4, 4)
            >>> a
            tensor([[ 1.3398,  0.2663, -0.2686,  0.2450],
                    [-0.7401, -0.8805, -0.3402, -1.1936],
                    [ 0.4907, -1.3948, -1.0691, -0.3132],
                    [-1.6092,  0.5419, -0.2993,  0.3195]])
            >>> torch.argmax(a, dim=1)
            tensor([ 0,  2,  0,  1])
        """
        return super().argmax(dim=dim, keepdim=keepdim)

    @return_tensor_wrapper
    def argmin(self, dim=None, keepdim=False) -> 'LongTensor':
        """
        argmin(input) -> LongTensor

        Returns the indices of the minimum value of all elements in the :attr:`input` tensor.

        This is the second value returned by :meth:`torch.min`. See its
        documentation for the exact semantics of this method.

        Args:
            input (Tensor): the input tensor.

        Example::

            >>> a = torch.randn(4, 4)
            >>> a
            tensor([[ 0.1139,  0.2254, -0.1381,  0.3687],
                    [ 1.0100, -1.1975, -0.0102, -0.4732],
                    [-0.9240,  0.1207, -0.7506, -1.0213],
                    [ 1.7809, -1.2960,  0.9384,  0.1438]])
            >>> torch.argmin(a)
            tensor(13)

        .. function:: argmin(input, dim, keepdim=False, out=None) -> LongTensor

        Returns the indices of the minimum values of a tensor across a dimension.

        This is the second value returned by :meth:`torch.min`. See its
        documentation for the exact semantics of this method.

        Args:
            input (Tensor): the input tensor.
            dim (int): the dimension to reduce. If ``None``, the argmin of the flattened input is returned.
            keepdim (bool): whether the output tensor has :attr:`dim` retained or not. Ignored if ``dim=None``.

        Example::

            >>> a = torch.randn(4, 4)
            >>> a
            tensor([[ 0.1139,  0.2254, -0.1381,  0.3687],
                    [ 1.0100, -1.1975, -0.0102, -0.4732],
                    [-0.9240,  0.1207, -0.7506, -1.0213],
                    [ 1.7809, -1.2960,  0.9384,  0.1438]])
            >>> torch.argmin(a, dim=1)
            tensor([ 2,  1,  3,  1])
        """
        return super().argmin(dim=dim, keepdim=keepdim)

    @return_tensor_wrapper
    def argsort(self, dim=-1, descending=False) -> 'LongTensor':
        """
        argsort(input, dim=-1, descending=False) -> LongTensor

        Returns the indices that sort a tensor along a given dimension in ascending
        order by value.

        This is the second value returned by :meth:`torch.sort`.  See its documentation
        for the exact semantics of this method.

        Args:
            input (Tensor): the input tensor.
            dim (int, optional): the dimension to sort along
            descending (bool, optional): controls the sorting order (ascending or descending)

        Example::

            >>> a = torch.randn(4, 4)
            >>> a
            tensor([[ 0.0785,  1.5267, -0.8521,  0.4065],
                    [ 0.1598,  0.0788, -0.0745, -1.2700],
                    [ 1.2208,  1.0722, -0.7064,  1.2564],
                    [ 0.0669, -0.2318, -0.8229, -0.9280]])


            >>> torch.argsort(a, dim=1)
            tensor([[2, 0, 3, 1],
                    [3, 2, 1, 0],
                    [2, 1, 0, 3],
                    [3, 2, 1, 0]])
        """
        return super().argsort(dim=dim, descending=descending)

    @return_tensor_wrapper
    def as_strided(self, size, stride, storage_offset=0) -> 'Tensor':
        """
        as_strided(input, size, stride, storage_offset=0) -> Tensor

        Create a view of an existing `torch.Tensor` :attr:`input` with specified
        :attr:`size`, :attr:`stride` and :attr:`storage_offset`.

        .. warning::
            More than one element of a created tensor may refer to a single memory
            location. As a result, in-place operations (especially ones that are
            vectorized) may result in incorrect behavior. If you need to write to
            the tensors, please clone them first.

            Many PyTorch functions, which return a view of a tensor, are internally
            implemented with this function. Those functions, like
            :meth:`torch.Tensor.expand`, are easier to read and are therefore more
            advisable to use.


        Args:
            input (Tensor): the input tensor.
            size (tuple or ints): the shape of the output tensor
            stride (tuple or ints): the stride of the output tensor
            storage_offset (int, optional): the offset in the underlying storage of the output tensor

        Example::

            >>> x = torch.randn(3, 3)
            >>> x
            tensor([[ 0.9039,  0.6291,  1.0795],
                    [ 0.1586,  2.1939, -0.4900],
                    [-0.1909, -0.7503,  1.9355]])
            >>> t = torch.as_strided(x, (2, 2), (1, 2))
            >>> t
            tensor([[0.9039, 1.0795],
                    [0.6291, 0.1586]])
            >>> t = torch.as_strided(x, (2, 2), (1, 2), 1)
            tensor([[0.6291, 0.1586],
                    [1.0795, 2.1939]])
        """
        return super().as_strided(size, stride, storage_offset=storage_offset)

    @return_tensor_wrapper
    def as_subclass(self, cls) -> 'Tensor':
        """
        as_subclass(cls) -> Tensor

        Makes a ``cls`` instance with the same data pointer as ``self``. Changes
        in the output mirror changes in ``self``, and the output stays attached
        to the autograd graph. ``cls`` must be a subclass of ``Tensor``.
        """
        return super().as_subclass(cls)

    @return_tensor_wrapper
    def asin(self) -> 'Tensor':
        """
        asin(input, out=None) -> Tensor

        Returns a new tensor with the arcsine  of the elements of :attr:`input`.

        .. math::
            \text{out}_{i} = \sin^{-1}(\text{input}_{i})

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4)
            >>> a
            tensor([-0.5962,  1.4985, -0.4396,  1.4525])
            >>> torch.asin(a)
            tensor([-0.6387,     nan, -0.4552,     nan])
        """
        return super().asin()

    @return_tensor_wrapper
    def asin_(self) -> 'Tensor':
        """
        asin_() -> Tensor

        In-place version of :meth:`~Tensor.asin`
        """
        return super().asin_()

    @return_tensor_wrapper
    def asinh(self) -> 'Tensor':
        """
        asinh(input, out=None) -> Tensor

        Returns a new tensor with the inverse hyperbolic sine of the elements of :attr:`input`.

        .. math::
            \text{out}_{i} = \sinh^{-1}(\text{input}_{i})

        Args:
            input (Tensor): the input tensor.

        Keyword arguments:
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4)
            >>> a
            tensor([ 0.1606, -1.4267, -1.0899, -1.0250 ])
            >>> torch.asinh(a)
            tensor([ 0.1599, -1.1534, -0.9435, -0.8990 ])
        """
        return super().asinh()

    @return_tensor_wrapper
    def asinh_(self) -> 'Tensor':
        """
        asinh_() -> Tensor

        In-place version of :meth:`~Tensor.asinh`
        """
        return super().asinh_()

    @return_tensor_wrapper
    def atan(self) -> 'Tensor':
        """
        atan(input, out=None) -> Tensor

        Returns a new tensor with the arctangent  of the elements of :attr:`input`.

        .. math::
            \text{out}_{i} = \tan^{-1}(\text{input}_{i})

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4)
            >>> a
            tensor([ 0.2341,  0.2539, -0.6256, -0.6448])
            >>> torch.atan(a)
            tensor([ 0.2299,  0.2487, -0.5591, -0.5727])
        """
        return super().atan()

    @return_tensor_wrapper
    def atan2(self, other) -> 'Tensor':
        """
        atan2(input, other, out=None) -> Tensor

        Element-wise arctangent of :math:`\text{input}_{i} / \text{other}_{i}`
        with consideration of the quadrant. Returns a new tensor with the signed angles
        in radians between vector :math:`(\text{other}_{i}, \text{input}_{i})`
        and vector :math:`(1, 0)`. (Note that :math:`\text{other}_{i}`, the second
        parameter, is the x-coordinate, while :math:`\text{input}_{i}`, the first
        parameter, is the y-coordinate.)

        The shapes of ``input`` and ``other`` must be
        :ref:`broadcastable <broadcasting-semantics>`.

        Args:
            input (Tensor): the first input tensor
            other (Tensor): the second input tensor
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4)
            >>> a
            tensor([ 0.9041,  0.0196, -0.3108, -2.4423])
            >>> torch.atan2(a, torch.randn(4))
            tensor([ 0.9833,  0.0811, -1.9743, -1.4151])
        """
        return super().atan2(other)

    @return_tensor_wrapper
    def atan2_(self, other) -> 'Tensor':
        """
        atan2_(other) -> Tensor

        In-place version of :meth:`~Tensor.atan2`
        """
        return super().atan2_(other)

    @return_tensor_wrapper
    def atan_(self) -> 'Tensor':
        """
        atan_() -> Tensor

        In-place version of :meth:`~Tensor.atan`
        """
        return super().atan_()

    @return_tensor_wrapper
    def atanh(self) -> 'Tensor':
        """
        atanh(input, out=None) -> Tensor

        Returns a new tensor with the inverse hyperbolic tangent of the elements of :attr:`input`.

        Note:
            The domain of the inverse hyperbolic tangent is `(-1, 1)` and values outside this range
            will be mapped to ``NaN``, except for the values `1` and `-1` for which the output is
            mapped to `+/-INF` respectively.

        .. math::
            \text{out}_{i} = \tanh^{-1}(\text{input}_{i})

        Args:
            input (Tensor): the input tensor.

        Keyword arguments:
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4).uniform_(-1, 1)
            >>> a
            tensor([ -0.9385, 0.2968, -0.8591, -0.1871 ])
            >>> torch.atanh(a)
            tensor([ -1.7253, 0.3060, -1.2899, -0.1893 ])
        """
        return super().atanh()

    @return_tensor_wrapper
    def baddbmm(self, batch1, batch2, *, beta=1, alpha=1) -> 'Tensor':
        """
        baddbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None) -> Tensor

        Performs a batch matrix-matrix product of matrices in :attr:`batch1`
        and :attr:`batch2`.
        :attr:`input` is added to the final result.

        :attr:`batch1` and :attr:`batch2` must be 3-D tensors each containing the same
        number of matrices.

        If :attr:`batch1` is a :math:`(b \times n \times m)` tensor, :attr:`batch2` is a
        :math:`(b \times m \times p)` tensor, then :attr:`input` must be
        :ref:`broadcastable <broadcasting-semantics>` with a
        :math:`(b \times n \times p)` tensor and :attr:`out` will be a
        :math:`(b \times n \times p)` tensor. Both :attr:`alpha` and :attr:`beta` mean the
        same as the scaling factors used in :meth:`torch.addbmm`.

        .. math::
            \text{out}_i = \beta\ \text{input}_i + \alpha\ (\text{batch1}_i \mathbin{@} \text{batch2}_i)

        For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and
        :attr:`alpha` must be real numbers, otherwise they should be integers.

        Args:
            input (Tensor): the tensor to be added
            batch1 (Tensor): the first batch of matrices to be multiplied
            batch2 (Tensor): the second batch of matrices to be multiplied
            beta (Number, optional): multiplier for :attr:`input` (:math:`\beta`)
            alpha (Number, optional): multiplier for :math:`\text{batch1} \mathbin{@} \text{batch2}` (:math:`\alpha`)
            out (Tensor, optional): the output tensor.

        Example::

            >>> M = torch.randn(10, 3, 5)
            >>> batch1 = torch.randn(10, 3, 4)
            >>> batch2 = torch.randn(10, 4, 5)
            >>> torch.baddbmm(M, batch1, batch2).size()
            torch.Size([10, 3, 5])
        """
        return super().baddbmm(batch1, batch2, beta=beta, alpha=alpha)

    @return_tensor_wrapper
    def baddbmm_(self, batch1, batch2, *, beta=1, alpha=1) -> 'Tensor':
        """
        baddbmm_(batch1, batch2, *, beta=1, alpha=1) -> Tensor

        In-place version of :meth:`~Tensor.baddbmm`
        """
        return super().baddbmm_(batch1, batch2, beta=beta, alpha=alpha)

    @return_tensor_wrapper
    def bernoulli(self, *, generator=None) -> 'Tensor':
        """
        bernoulli(input, *, generator=None, out=None) -> Tensor

        Draws binary random numbers (0 or 1) from a Bernoulli distribution.

        The :attr:`input` tensor should be a tensor containing probabilities
        to be used for drawing the binary random number.
        Hence, all values in :attr:`input` have to be in the range:
        :math:`0 \leq \text{input}_i \leq 1`.

        The :math:`\text{i}^{th}` element of the output tensor will draw a
        value :math:`1` according to the :math:`\text{i}^{th}` probability value given
        in :attr:`input`.

        .. math::
            \text{out}_{i} \sim \mathrm{Bernoulli}(p = \text{input}_{i})

        The returned :attr:`out` tensor only has values 0 or 1 and is of the same
        shape as :attr:`input`.

        :attr:`out` can have integral ``dtype``, but :attr:`input` must have floating
        point ``dtype``.

        Args:
            input (Tensor): the input tensor of probability values for the Bernoulli distribution
            generator (:class:`torch.Generator`, optional): a pseudorandom number generator for sampling
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.empty(3, 3).uniform_(0, 1)  # generate a uniform random matrix with range [0, 1]
            >>> a
            tensor([[ 0.1737,  0.0950,  0.3609],
                    [ 0.7148,  0.0289,  0.2676],
                    [ 0.9456,  0.8937,  0.7202]])
            >>> torch.bernoulli(a)
            tensor([[ 1.,  0.,  0.],
                    [ 0.,  0.,  0.],
                    [ 1.,  1.,  1.]])

            >>> a = torch.ones(3, 3) # probability of drawing "1" is 1
            >>> torch.bernoulli(a)
            tensor([[ 1.,  1.,  1.],
                    [ 1.,  1.,  1.],
                    [ 1.,  1.,  1.]])
            >>> a = torch.zeros(3, 3) # probability of drawing "1" is 0
            >>> torch.bernoulli(a)
            tensor([[ 0.,  0.,  0.],
                    [ 0.,  0.,  0.],
                    [ 0.,  0.,  0.]])
        """
        return super().bernoulli(generator=generator)

    @return_tensor_wrapper
    def bfloat16(self, memory_format=torch.preserve_format) -> 'Tensor':
        """
        bfloat16(memory_format=torch.preserve_format) -> Tensor
        ``self.bfloat16()`` is equivalent to ``self.to(torch.bfloat16)``. See :func:`to`.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
        return super().bfloat16(memory_format=memory_format)

    @return_tensor_wrapper
    def bincount(self, weights=None, minlength=0) -> 'Tensor':
        """
        bincount(input, weights=None, minlength=0) -> Tensor

        Count the frequency of each value in an array of non-negative ints.

        The number of bins (size 1) is one larger than the largest value in
        :attr:`input` unless :attr:`input` is empty, in which case the result is a
        tensor of size 0. If :attr:`minlength` is specified, the number of bins is at least
        :attr:`minlength` and if :attr:`input` is empty, then the result is tensor of size
        :attr:`minlength` filled with zeros. If ``n`` is the value at position ``i``,
        ``out[n] += weights[i]`` if :attr:`weights` is specified else
        ``out[n] += 1``.

        Note:
            In some circumstances when using the CUDA backend with CuDNN, this operator
            may select a nondeterministic algorithm to increase performance. If this is
            undesirable, you can try to make the operation deterministic (potentially at
            a performance cost) by setting ``torch.backends.cudnn.deterministic =
            True``.
            Please see the notes on :doc:`/notes/randomness` for background.

        Arguments:
            input (Tensor): 1-d int tensor
            weights (Tensor): optional, weight for each value in the input tensor.
                Should be of same size as input tensor.
            minlength (int): optional, minimum number of bins. Should be non-negative.

        Returns:
            output (Tensor): a tensor of shape ``Size([max(input) + 1])`` if
            :attr:`input` is non-empty, else ``Size(0)``

        Example::

            >>> input = torch.randint(0, 8, (5,), dtype=torch.int64)
            >>> weights = torch.linspace(0, 1, steps=5)
            >>> input, weights
            (tensor([4, 3, 6, 3, 4]),
             tensor([ 0.0000,  0.2500,  0.5000,  0.7500,  1.0000])

            >>> torch.bincount(input)
            tensor([0, 0, 0, 2, 2, 0, 1])

            >>> input.bincount(weights)
            tensor([0.0000, 0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 0.5000])
        """
        return super().bincount(weights=weights, minlength=minlength)

    @return_tensor_wrapper
    def bitwise_and(self) -> 'Tensor':
        """
        bitwise_and(input, other, out=None) -> Tensor

        Computes the bitwise AND of :attr:`input` and :attr:`other`. The input tensor must be of
        integral or Boolean types. For bool tensors, it computes the logical AND.

        Args:
            input: the first input tensor
            other: the second input tensor
            out (Tensor, optional): the output tensor.

        Example:

            >>> torch.bitwise_and(torch.tensor([-1, -2, 3], dtype=torch.int8), torch.tensor([1, 0, 3], dtype=torch.int8))
            tensor([1, 0,  3], dtype=torch.int8)
            >>> torch.bitwise_and(torch.tensor([True, True, False]), torch.tensor([False, True, False]))
            tensor([ False, True, False])
        """
        return super().bitwise_and()

    @return_tensor_wrapper
    def bitwise_and_(self) -> 'Tensor':
        """
        bitwise_and_() -> Tensor

        In-place version of :meth:`~Tensor.bitwise_and`
        """
        return super().bitwise_and_()

    @return_tensor_wrapper
    def bitwise_not(self) -> 'Tensor':
        """
        bitwise_not(input, out=None) -> Tensor

        Computes the bitwise NOT of the given input tensor. The input tensor must be of
        integral or Boolean types. For bool tensors, it computes the logical NOT.

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example:

            >>> torch.bitwise_not(torch.tensor([-1, -2, 3], dtype=torch.int8))
            tensor([ 0,  1, -4], dtype=torch.int8)
        """
        return super().bitwise_not()

    @return_tensor_wrapper
    def bitwise_not_(self) -> 'Tensor':
        """
        bitwise_not_() -> Tensor

        In-place version of :meth:`~Tensor.bitwise_not`
        """
        return super().bitwise_not_()

    @return_tensor_wrapper
    def bitwise_or(self) -> 'Tensor':
        """
        bitwise_or(input, other, out=None) -> Tensor

        Computes the bitwise OR of :attr:`input` and :attr:`other`. The input tensor must be of
        integral or Boolean types. For bool tensors, it computes the logical OR.

        Args:
            input: the first input tensor
            other: the second input tensor
            out(Tensor, optional): the output tensor.

        Example:

            >>> torch.bitwise_or(torch.tensor([-1, -2, 3], dtype=torch.int8), torch.tensor([1, 0, 3], dtype=torch.int8))
            tensor([-1, -2,  3], dtype=torch.int8)
            >>> torch.bitwise_or(torch.tensor([True, True, False]), torch.tensor([False, True, False]))
            tensor([ True, True, False])
        """
        return super().bitwise_or()

    @return_tensor_wrapper
    def bitwise_or_(self) -> 'Tensor':
        """
        bitwise_or_() -> Tensor

        In-place version of :meth:`~Tensor.bitwise_or`
        """
        return super().bitwise_or_()

    @return_tensor_wrapper
    def bitwise_xor(self) -> 'Tensor':
        """
        bitwise_xor(input, other, out=None) -> Tensor

        Computes the bitwise XOR of :attr:`input` and :attr:`other`. The input tensor must be of
        integral or Boolean types. For bool tensors, it computes the logical XOR.

        Args:
            input: the first input tensor
            other: the second input tensor
            out (Tensor, optional): the output tensor.

        Example:

            >>> torch.bitwise_xor(torch.tensor([-1, -2, 3], dtype=torch.int8), torch.tensor([1, 0, 3], dtype=torch.int8))
            tensor([-2, -2,  0], dtype=torch.int8)
            >>> torch.bitwise_xor(torch.tensor([True, True, False]), torch.tensor([False, True, False]))
            tensor([ True, False, False])
        """
        return super().bitwise_xor()

    @return_tensor_wrapper
    def bitwise_xor_(self) -> 'Tensor':
        """
        bitwise_xor_() -> Tensor

        In-place version of :meth:`~Tensor.bitwise_xor`
        """
        return super().bitwise_xor_()

    @return_tensor_wrapper
    def bmm(self, batch2) -> 'Tensor':
        """
        bmm(input, mat2, deterministic=False, out=None) -> Tensor

        Performs a batch matrix-matrix product of matrices stored in :attr:`input`
        and :attr:`mat2`.

        :attr:`input` and :attr:`mat2` must be 3-D tensors each containing
        the same number of matrices.

        If :attr:`input` is a :math:`(b \times n \times m)` tensor, :attr:`mat2` is a
        :math:`(b \times m \times p)` tensor, :attr:`out` will be a
        :math:`(b \times n \times p)` tensor.

        .. math::
            \text{out}_i = \text{input}_i \mathbin{@} \text{mat2}_i

        .. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.
                  For broadcasting matrix products, see :func:`torch.matmul`.

        Args:
            input (Tensor): the first batch of matrices to be multiplied
            mat2 (Tensor): the second batch of matrices to be multiplied
            deterministic (bool, optional): flag to choose between a faster non-deterministic
                                            calculation, or a slower deterministic calculation.
                                            This argument is only available for sparse-dense CUDA bmm.
                                            Default: ``False``
            out (Tensor, optional): the output tensor.

        Example::

            >>> input = torch.randn(10, 3, 4)
            >>> mat2 = torch.randn(10, 4, 5)
            >>> res = torch.bmm(input, mat2)
            >>> res.size()
            torch.Size([10, 3, 5])
        """
        return super().bmm(batch2)

    @return_tensor_wrapper
    def bool(self, memory_format=torch.preserve_format) -> 'Tensor':
        """
        bool(memory_format=torch.preserve_format) -> Tensor

        ``self.bool()`` is equivalent to ``self.to(torch.bool)``. See :func:`to`.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
        return super().bool(memory_format=memory_format)

    @return_tensor_wrapper
    def byte(self, memory_format=torch.preserve_format) -> 'Tensor':
        """
        byte(memory_format=torch.preserve_format) -> Tensor

        ``self.byte()`` is equivalent to ``self.to(torch.uint8)``. See :func:`to`.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
        return super().byte(memory_format=memory_format)

    @return_tensor_wrapper
    def cauchy_(self, median=0, sigma=1, *, generator=None) -> 'Tensor':
        """
        cauchy_(median=0, sigma=1, *, generator=None) -> Tensor

        Fills the tensor with numbers drawn from the Cauchy distribution:

        .. math::

            f(x) = \dfrac{1}{\pi} \dfrac{\sigma}{(x - \text{median})^2 + \sigma^2}
        """
        return super().cauchy_(median=median, sigma=sigma, generator=generator)

    @return_tensor_wrapper
    def ceil(self) -> 'Tensor':
        """
        ceil(input, out=None) -> Tensor

        Returns a new tensor with the ceil of the elements of :attr:`input`,
        the smallest integer greater than or equal to each element.

        .. math::
            \text{out}_{i} = \left\lceil \text{input}_{i} \right\rceil = \left\lfloor \text{input}_{i} \right\rfloor + 1

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4)
            >>> a
            tensor([-0.6341, -1.4208, -1.0900,  0.5826])
            >>> torch.ceil(a)
            tensor([-0., -1., -1.,  1.])
        """
        return super().ceil()

    @return_tensor_wrapper
    def ceil_(self) -> 'Tensor':
        """
        ceil_() -> Tensor

        In-place version of :meth:`~Tensor.ceil`
        """
        return super().ceil_()

    @return_tensor_wrapper
    def char(self, memory_format=torch.preserve_format) -> 'Tensor':
        """
        char(memory_format=torch.preserve_format) -> Tensor

        ``self.char()`` is equivalent to ``self.to(torch.int8)``. See :func:`to`.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
        return super().char(memory_format=memory_format)

    @return_tensor_wrapper
    def cholesky(self, upper=False) -> 'Tensor':
        """
        cholesky(input, upper=False, out=None) -> Tensor

        Computes the Cholesky decomposition of a symmetric positive-definite
        matrix :math:`A` or for batches of symmetric positive-definite matrices.

        If :attr:`upper` is ``True``, the returned matrix ``U`` is upper-triangular, and
        the decomposition has the form:

        .. math::

          A = U^TU

        If :attr:`upper` is ``False``, the returned matrix ``L`` is lower-triangular, and
        the decomposition has the form:

        .. math::

            A = LL^T

        If :attr:`upper` is ``True``, and :math:`A` is a batch of symmetric positive-definite
        matrices, then the returned tensor will be composed of upper-triangular Cholesky factors
        of each of the individual matrices. Similarly, when :attr:`upper` is ``False``, the returned
        tensor will be composed of lower-triangular Cholesky factors of each of the individual
        matrices.

        Args:
            input (Tensor): the input tensor :math:`A` of size :math:`(*, n, n)` where `*` is zero or more
                        batch dimensions consisting of symmetric positive-definite matrices.
            upper (bool, optional): flag that indicates whether to return a
                                    upper or lower triangular matrix. Default: ``False``
            out (Tensor, optional): the output matrix

        Example::

            >>> a = torch.randn(3, 3)
            >>> a = torch.mm(a, a.t()) # make symmetric positive-definite
            >>> l = torch.cholesky(a)
            >>> a
            tensor([[ 2.4112, -0.7486,  1.4551],
                    [-0.7486,  1.3544,  0.1294],
                    [ 1.4551,  0.1294,  1.6724]])
            >>> l
            tensor([[ 1.5528,  0.0000,  0.0000],
                    [-0.4821,  1.0592,  0.0000],
                    [ 0.9371,  0.5487,  0.7023]])
            >>> torch.mm(l, l.t())
            tensor([[ 2.4112, -0.7486,  1.4551],
                    [-0.7486,  1.3544,  0.1294],
                    [ 1.4551,  0.1294,  1.6724]])
            >>> a = torch.randn(3, 2, 2)
            >>> a = torch.matmul(a, a.transpose(-1, -2)) + 1e-03 # make symmetric positive-definite
            >>> l = torch.cholesky(a)
            >>> z = torch.matmul(l, l.transpose(-1, -2))
            >>> torch.max(torch.abs(z - a)) # Max non-zero
            tensor(2.3842e-07)
        """
        return super().cholesky(upper=upper)

    @return_tensor_wrapper
    def cholesky_inverse(self, upper=False) -> 'Tensor':
        """
        cholesky_inverse(input, upper=False, out=None) -> Tensor

        Computes the inverse of a symmetric positive-definite matrix :math:`A` using its
        Cholesky factor :math:`u`: returns matrix ``inv``. The inverse is computed using
        LAPACK routines ``dpotri`` and ``spotri`` (and the corresponding MAGMA routines).

        If :attr:`upper` is ``False``, :math:`u` is lower triangular
        such that the returned tensor is

        .. math::
            inv = (uu^{{T}})^{{-1}}

        If :attr:`upper` is ``True`` or not provided, :math:`u` is upper
        triangular such that the returned tensor is

        .. math::
            inv = (u^T u)^{{-1}}

        Args:
            input (Tensor): the input 2-D tensor :math:`u`, a upper or lower triangular
                   Cholesky factor
            upper (bool, optional): whether to return a lower (default) or upper triangular matrix
            out (Tensor, optional): the output tensor for `inv`

        Example::

            >>> a = torch.randn(3, 3)
            >>> a = torch.mm(a, a.t()) + 1e-05 * torch.eye(3) # make symmetric positive definite
            >>> u = torch.cholesky(a)
            >>> a
            tensor([[  0.9935,  -0.6353,   1.5806],
                    [ -0.6353,   0.8769,  -1.7183],
                    [  1.5806,  -1.7183,  10.6618]])
            >>> torch.cholesky_inverse(u)
            tensor([[ 1.9314,  1.2251, -0.0889],
                    [ 1.2251,  2.4439,  0.2122],
                    [-0.0889,  0.2122,  0.1412]])
            >>> a.inverse()
            tensor([[ 1.9314,  1.2251, -0.0889],
                    [ 1.2251,  2.4439,  0.2122],
                    [-0.0889,  0.2122,  0.1412]])
        """
        return super().cholesky_inverse(upper=upper)

    @return_tensor_wrapper
    def cholesky_solve(self, input2, upper=False) -> 'Tensor':
        """
        cholesky_solve(input, input2, upper=False, out=None) -> Tensor

        Solves a linear system of equations with a positive semidefinite
        matrix to be inverted given its Cholesky factor matrix :math:`u`.

        If :attr:`upper` is ``False``, :math:`u` is and lower triangular and `c` is
        returned such that:

        .. math::
            c = (u u^T)^{{-1}} b

        If :attr:`upper` is ``True`` or not provided, :math:`u` is upper triangular
        and `c` is returned such that:

        .. math::
            c = (u^T u)^{{-1}} b

        `torch.cholesky_solve(b, u)` can take in 2D inputs `b, u` or inputs that are
        batches of 2D matrices. If the inputs are batches, then returns
        batched outputs `c`

        Args:
            input (Tensor): input matrix :math:`b` of size :math:`(*, m, k)`,
                        where :math:`*` is zero or more batch dimensions
            input2 (Tensor): input matrix :math:`u` of size :math:`(*, m, m)`,
                        where :math:`*` is zero of more batch dimensions composed of
                        upper or lower triangular Cholesky factor
            upper (bool, optional): whether to consider the Cholesky factor as a
                                    lower or upper triangular matrix. Default: ``False``.
            out (Tensor, optional): the output tensor for `c`

        Example::

            >>> a = torch.randn(3, 3)
            >>> a = torch.mm(a, a.t()) # make symmetric positive definite
            >>> u = torch.cholesky(a)
            >>> a
            tensor([[ 0.7747, -1.9549,  1.3086],
                    [-1.9549,  6.7546, -5.4114],
                    [ 1.3086, -5.4114,  4.8733]])
            >>> b = torch.randn(3, 2)
            >>> b
            tensor([[-0.6355,  0.9891],
                    [ 0.1974,  1.4706],
                    [-0.4115, -0.6225]])
            >>> torch.cholesky_solve(b, u)
            tensor([[ -8.1625,  19.6097],
                    [ -5.8398,  14.2387],
                    [ -4.3771,  10.4173]])
            >>> torch.mm(a.inverse(), b)
            tensor([[ -8.1626,  19.6097],
                    [ -5.8398,  14.2387],
                    [ -4.3771,  10.4173]])
        """
        return super().cholesky_solve(input2, upper=upper)

    @return_tensor_wrapper
    def clamp(self, min, max) -> 'Tensor':
        """
        clamp(input, min, max, out=None) -> Tensor

        Clamp all elements in :attr:`input` into the range `[` :attr:`min`, :attr:`max` `]` and return
        a resulting tensor:

        .. math::
            y_i = \begin{cases}
                \text{min} & \text{if } x_i < \text{min} \\
                x_i & \text{if } \text{min} \leq x_i \leq \text{max} \\
                \text{max} & \text{if } x_i > \text{max}
            \end{cases}

        If :attr:`input` is of type `FloatTensor` or `DoubleTensor`, args :attr:`min`
        and :attr:`max` must be real numbers, otherwise they should be integers.

        Args:
            input (Tensor): the input tensor.
            min (Number): lower-bound of the range to be clamped to
            max (Number): upper-bound of the range to be clamped to
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4)
            >>> a
            tensor([-1.7120,  0.1734, -0.0478, -0.0922])
            >>> torch.clamp(a, min=-0.5, max=0.5)
            tensor([-0.5000,  0.1734, -0.0478, -0.0922])

        .. function:: clamp(input, *, min, out=None) -> Tensor

        Clamps all elements in :attr:`input` to be larger or equal :attr:`min`.

        If :attr:`input` is of type `FloatTensor` or `DoubleTensor`, :attr:`value`
        should be a real number, otherwise it should be an integer.

        Args:
            input (Tensor): the input tensor.
            value (Number): minimal value of each element in the output
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4)
            >>> a
            tensor([-0.0299, -2.3184,  2.1593, -0.8883])
            >>> torch.clamp(a, min=0.5)
            tensor([ 0.5000,  0.5000,  2.1593,  0.5000])

        .. function:: clamp(input, *, max, out=None) -> Tensor

        Clamps all elements in :attr:`input` to be smaller or equal :attr:`max`.

        If :attr:`input` is of type `FloatTensor` or `DoubleTensor`, :attr:`value`
        should be a real number, otherwise it should be an integer.

        Args:
            input (Tensor): the input tensor.
            value (Number): maximal value of each element in the output
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4)
            >>> a
            tensor([ 0.7753, -0.4702, -0.4599,  1.1899])
            >>> torch.clamp(a, max=0.5)
            tensor([ 0.5000, -0.4702, -0.4599,  0.5000])
        """
        return super().clamp(min, max)

    @return_tensor_wrapper
    def clamp_(self, min, max) -> 'Tensor':
        """
        clamp_(min, max) -> Tensor

        In-place version of :meth:`~Tensor.clamp`
        """
        return super().clamp_(min, max)

    @return_tensor_wrapper
    def clone(self, memory_format=torch.preserve_format) -> 'Tensor':
        """
        clone(memory_format=torch.preserve_format) -> Tensor

        Returns a copy of the :attr:`self` tensor. The copy has the same size and data
        type as :attr:`self`.

        .. note::

            Unlike `copy_()`, this function is recorded in the computation graph. Gradients
            propagating to the cloned tensor will propagate to the original tensor.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
        return super().clone(memory_format=memory_format)

    @return_tensor_wrapper
    def conj(self) -> 'Tensor':
        """
        conj(input, out=None) -> Tensor

        Computes the element-wise conjugate of the given :attr:`input` tensor.

        .. math::
            \text{out}_{i} = conj(\text{input}_{i})

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> torch.conj(torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j]))
            tensor([-1 - 1j, -2 - 2j, 3 + 3j])
        """
        return super().conj()

    @return_tensor_wrapper
    def contiguous(self, memory_format=torch.contiguous_format) -> 'Tensor':
        """
        contiguous(memory_format=torch.contiguous_format) -> Tensor

        Returns a contiguous in memory tensor containing the same data as :attr:`self` tensor. If
        :attr:`self` tensor is already in the specified memory format, this function returns the
        :attr:`self` tensor.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.contiguous_format``.
        """
        return super().contiguous(memory_format=memory_format)

    @return_tensor_wrapper
    def copy_(self, src, non_blocking=False) -> 'Tensor':
        """
        copy_(src, non_blocking=False) -> Tensor

        Copies the elements from :attr:`src` into :attr:`self` tensor and returns
        :attr:`self`.

        The :attr:`src` tensor must be :ref:`broadcastable <broadcasting-semantics>`
        with the :attr:`self` tensor. It may be of a different data type or reside on a
        different device.

        Args:
            src (Tensor): the source tensor to copy from
            non_blocking (bool): if ``True`` and this copy is between CPU and GPU,
                the copy may occur asynchronously with respect to the host. For other
                cases, this argument has no effect.
        """
        return super().copy_(src, non_blocking=non_blocking)

    @return_tensor_wrapper
    def cos(self) -> 'Tensor':
        """
        cos(input, out=None) -> Tensor

        Returns a new tensor with the cosine  of the elements of :attr:`input`.

        .. math::
            \text{out}_{i} = \cos(\text{input}_{i})

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4)
            >>> a
            tensor([ 1.4309,  1.2706, -0.8562,  0.9796])
            >>> torch.cos(a)
            tensor([ 0.1395,  0.2957,  0.6553,  0.5574])
        """
        return super().cos()

    @return_tensor_wrapper
    def cos_(self) -> 'Tensor':
        """
        cos_() -> Tensor

        In-place version of :meth:`~Tensor.cos`
        """
        return super().cos_()

    @return_tensor_wrapper
    def cosh(self) -> 'Tensor':
        """
        cosh(input, out=None) -> Tensor

        Returns a new tensor with the hyperbolic cosine  of the elements of
        :attr:`input`.

        .. math::
            \text{out}_{i} = \cosh(\text{input}_{i})

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4)
            >>> a
            tensor([ 0.1632,  1.1835, -0.6979, -0.7325])
            >>> torch.cosh(a)
            tensor([ 1.0133,  1.7860,  1.2536,  1.2805])
        """
        return super().cosh()

    @return_tensor_wrapper
    def cosh_(self) -> 'Tensor':
        """
        cosh_() -> Tensor

        In-place version of :meth:`~Tensor.cosh`
        """
        return super().cosh_()

    @return_tensor_wrapper(False)
    def cpu(self, memory_format=torch.preserve_format) -> 'Tensor':
        """
        cpu(memory_format=torch.preserve_format) -> Tensor

        Returns a copy of this object in CPU memory.

        If this object is already in CPU memory and on the correct device,
        then no copy is performed and the original object is returned.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
        return super().cpu(memory_format=memory_format)

    @return_tensor_wrapper
    def cross(self, other, dim=-1) -> 'Tensor':
        """
        cross(input, other, dim=-1, out=None) -> Tensor


        Returns the cross product of vectors in dimension :attr:`dim` of :attr:`input`
        and :attr:`other`.

        :attr:`input` and :attr:`other` must have the same size, and the size of their
        :attr:`dim` dimension should be 3.

        If :attr:`dim` is not given, it defaults to the first dimension found with the
        size 3.

        Args:
            input (Tensor): the input tensor.
            other (Tensor): the second input tensor
            dim  (int, optional): the dimension to take the cross-product in.
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4, 3)
            >>> a
            tensor([[-0.3956,  1.1455,  1.6895],
                    [-0.5849,  1.3672,  0.3599],
                    [-1.1626,  0.7180, -0.0521],
                    [-0.1339,  0.9902, -2.0225]])
            >>> b = torch.randn(4, 3)
            >>> b
            tensor([[-0.0257, -1.4725, -1.2251],
                    [-1.1479, -0.7005, -1.9757],
                    [-1.3904,  0.3726, -1.1836],
                    [-0.9688, -0.7153,  0.2159]])
            >>> torch.cross(a, b, dim=1)
            tensor([[ 1.0844, -0.5281,  0.6120],
                    [-2.4490, -1.5687,  1.9792],
                    [-0.8304, -1.3037,  0.5650],
                    [-1.2329,  1.9883,  1.0551]])
            >>> torch.cross(a, b)
            tensor([[ 1.0844, -0.5281,  0.6120],
                    [-2.4490, -1.5687,  1.9792],
                    [-0.8304, -1.3037,  0.5650],
                    [-1.2329,  1.9883,  1.0551]])
        """
        return super().cross(other, dim=dim)

    @return_tensor_wrapper(False)
    def cuda(self, device=None, non_blocking=False, memory_format=torch.preserve_format) -> 'Tensor':
        """
        cuda(device=None, non_blocking=False, memory_format=torch.preserve_format) -> Tensor

        Returns a copy of this object in CUDA memory.

        If this object is already in CUDA memory and on the correct device,
        then no copy is performed and the original object is returned.

        Args:
            device (:class:`torch.device`): The destination GPU device.
                Defaults to the current CUDA device.
            non_blocking (bool): If ``True`` and the source is in pinned memory,
                the copy will be asynchronous with respect to the host.
                Otherwise, the argument has no effect. Default: ``False``.
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
        return super().cuda(device=device, non_blocking=non_blocking, memory_format=memory_format)

    @return_tensor_wrapper
    def cumprod(self, dim, dtype=None) -> 'Tensor':
        """
        cumprod(input, dim, out=None, dtype=None) -> Tensor

        Returns the cumulative product of elements of :attr:`input` in the dimension
        :attr:`dim`.

        For example, if :attr:`input` is a vector of size N, the result will also be
        a vector of size N, with elements.

        .. math::
            y_i = x_1 \times x_2\times x_3\times \dots \times x_i

        Args:
            input (Tensor): the input tensor.
            dim  (int): the dimension to do the operation over
            dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
                If specified, the input tensor is casted to :attr:`dtype` before the operation
                is performed. This is useful for preventing data type overflows. Default: None.
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(10)
            >>> a
            tensor([ 0.6001,  0.2069, -0.1919,  0.9792,  0.6727,  1.0062,  0.4126,
                    -0.2129, -0.4206,  0.1968])
            >>> torch.cumprod(a, dim=0)
            tensor([ 0.6001,  0.1241, -0.0238, -0.0233, -0.0157, -0.0158, -0.0065,
                     0.0014, -0.0006, -0.0001])

            >>> a[5] = 0.0
            >>> torch.cumprod(a, dim=0)
            tensor([ 0.6001,  0.1241, -0.0238, -0.0233, -0.0157, -0.0000, -0.0000,
                     0.0000, -0.0000, -0.0000])
        """
        return super().cumprod(dim, dtype=dtype)

    @return_tensor_wrapper
    def cumsum(self, dim, dtype=None) -> 'Tensor':
        """
        cumsum(input, dim, out=None, dtype=None) -> Tensor

        Returns the cumulative sum of elements of :attr:`input` in the dimension
        :attr:`dim`.

        For example, if :attr:`input` is a vector of size N, the result will also be
        a vector of size N, with elements.

        .. math::
            y_i = x_1 + x_2 + x_3 + \dots + x_i

        Args:
            input (Tensor): the input tensor.
            dim  (int): the dimension to do the operation over
            dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
                If specified, the input tensor is casted to :attr:`dtype` before the operation
                is performed. This is useful for preventing data type overflows. Default: None.
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(10)
            >>> a
            tensor([-0.8286, -0.4890,  0.5155,  0.8443,  0.1865, -0.1752, -2.0595,
                     0.1850, -1.1571, -0.4243])
            >>> torch.cumsum(a, dim=0)
            tensor([-0.8286, -1.3175, -0.8020,  0.0423,  0.2289,  0.0537, -2.0058,
                    -1.8209, -2.9780, -3.4022])
        """
        return super().cumsum(dim, dtype=dtype)

    @return_tensor_wrapper
    def deg2rad(self) -> 'Tensor':
        """
        deg2rad(input, out=None) -> Tensor

        Returns a new tensor with each of the elements of :attr:`input`
        converted from angles in degrees to radians.

        Args:
            input (Tensor): the input tensor.

        Keyword arguments:
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.tensor([[180.0, -180.0], [360.0, -360.0], [90.0, -90.0]])
            >>> torch.deg2rad(a)
            tensor([[ 3.1416, -3.1416],
                    [ 6.2832, -6.2832],
                    [ 1.5708, -1.5708]])
        """
        return super().deg2rad()

    @return_tensor_wrapper
    def deg2rad_(self) -> 'Tensor':
        """
        deg2rad_() -> Tensor

        In-place version of :meth:`~Tensor.deg2rad`
        """
        return super().deg2rad_()

    @return_tensor_wrapper
    def dequantize(self) -> 'Tensor':
        """
        dequantize() -> Tensor

        Given a quantized Tensor, dequantize it and return the dequantized float Tensor.
        """
        return super().dequantize()

    @return_tensor_wrapper
    def det(self) -> 'Tensor':
        """
        det(input) -> Tensor

        Calculates determinant of a square matrix or batches of square matrices.

        .. note::
            Backward through :meth:`det` internally uses SVD results when :attr:`input` is
            not invertible. In this case, double backward through :meth:`det` will be
            unstable in when :attr:`input` doesn't have distinct singular values. See
            :meth:`~torch.svd` for details.

        Arguments:
            input (Tensor): the input tensor of size ``(*, n, n)`` where ``*`` is zero or more
                        batch dimensions.

        Example::

            >>> A = torch.randn(3, 3)
            >>> torch.det(A)
            tensor(3.7641)

            >>> A = torch.randn(3, 2, 2)
            >>> A
            tensor([[[ 0.9254, -0.6213],
                     [-0.5787,  1.6843]],

                    [[ 0.3242, -0.9665],
                     [ 0.4539, -0.0887]],

                    [[ 1.1336, -0.4025],
                     [-0.7089,  0.9032]]])
            >>> A.det()
            tensor([1.1990, 0.4099, 0.7386])
        """
        return super().det()

    @return_tensor_wrapper
    def diag(self, diagonal=0) -> 'Tensor':
        """
        diag(input, diagonal=0, out=None) -> Tensor

        - If :attr:`input` is a vector (1-D tensor), then returns a 2-D square tensor
          with the elements of :attr:`input` as the diagonal.
        - If :attr:`input` is a matrix (2-D tensor), then returns a 1-D tensor with
          the diagonal elements of :attr:`input`.

        The argument :attr:`diagonal` controls which diagonal to consider:

        - If :attr:`diagonal` = 0, it is the main diagonal.
        - If :attr:`diagonal` > 0, it is above the main diagonal.
        - If :attr:`diagonal` < 0, it is below the main diagonal.

        Args:
            input (Tensor): the input tensor.
            diagonal (int, optional): the diagonal to consider
            out (Tensor, optional): the output tensor.

        .. seealso::

                :func:`torch.diagonal` always returns the diagonal of its input.

                :func:`torch.diagflat` always constructs a tensor with diagonal elements
                specified by the input.

        Examples:

        Get the square matrix where the input vector is the diagonal::

            >>> a = torch.randn(3)
            >>> a
            tensor([ 0.5950,-0.0872, 2.3298])
            >>> torch.diag(a)
            tensor([[ 0.5950, 0.0000, 0.0000],
                    [ 0.0000,-0.0872, 0.0000],
                    [ 0.0000, 0.0000, 2.3298]])
            >>> torch.diag(a, 1)
            tensor([[ 0.0000, 0.5950, 0.0000, 0.0000],
                    [ 0.0000, 0.0000,-0.0872, 0.0000],
                    [ 0.0000, 0.0000, 0.0000, 2.3298],
                    [ 0.0000, 0.0000, 0.0000, 0.0000]])

        Get the k-th diagonal of a given matrix::

            >>> a = torch.randn(3, 3)
            >>> a
            tensor([[-0.4264, 0.0255,-0.1064],
                    [ 0.8795,-0.2429, 0.1374],
                    [ 0.1029,-0.6482,-1.6300]])
            >>> torch.diag(a, 0)
            tensor([-0.4264,-0.2429,-1.6300])
            >>> torch.diag(a, 1)
            tensor([ 0.0255, 0.1374])
        """
        return super().diag(diagonal=diagonal)

    @return_tensor_wrapper
    def diag_embed(self, offset=0, dim1=-2, dim2=-1) -> 'Tensor':
        """
        diag_embed(input, offset=0, dim1=-2, dim2=-1) -> Tensor

        Creates a tensor whose diagonals of certain 2D planes (specified by
        :attr:`dim1` and :attr:`dim2`) are filled by :attr:`input`.
        To facilitate creating batched diagonal matrices, the 2D planes formed by
        the last two dimensions of the returned tensor are chosen by default.

        The argument :attr:`offset` controls which diagonal to consider:

        - If :attr:`offset` = 0, it is the main diagonal.
        - If :attr:`offset` > 0, it is above the main diagonal.
        - If :attr:`offset` < 0, it is below the main diagonal.

        The size of the new matrix will be calculated to make the specified diagonal
        of the size of the last input dimension.
        Note that for :attr:`offset` other than :math:`0`, the order of :attr:`dim1`
        and :attr:`dim2` matters. Exchanging them is equivalent to changing the
        sign of :attr:`offset`.

        Applying :meth:`torch.diagonal` to the output of this function with
        the same arguments yields a matrix identical to input. However,
        :meth:`torch.diagonal` has different default dimensions, so those
        need to be explicitly specified.

        Args:
            input (Tensor): the input tensor. Must be at least 1-dimensional.
            offset (int, optional): which diagonal to consider. Default: 0
                (main diagonal).
            dim1 (int, optional): first dimension with respect to which to
                take diagonal. Default: -2.
            dim2 (int, optional): second dimension with respect to which to
                take diagonal. Default: -1.

        Example::

            >>> a = torch.randn(2, 3)
            >>> torch.diag_embed(a)
            tensor([[[ 1.5410,  0.0000,  0.0000],
                     [ 0.0000, -0.2934,  0.0000],
                     [ 0.0000,  0.0000, -2.1788]],

                    [[ 0.5684,  0.0000,  0.0000],
                     [ 0.0000, -1.0845,  0.0000],
                     [ 0.0000,  0.0000, -1.3986]]])

            >>> torch.diag_embed(a, offset=1, dim1=0, dim2=2)
            tensor([[[ 0.0000,  1.5410,  0.0000,  0.0000],
                     [ 0.0000,  0.5684,  0.0000,  0.0000]],

                    [[ 0.0000,  0.0000, -0.2934,  0.0000],
                     [ 0.0000,  0.0000, -1.0845,  0.0000]],

                    [[ 0.0000,  0.0000,  0.0000, -2.1788],
                     [ 0.0000,  0.0000,  0.0000, -1.3986]],

                    [[ 0.0000,  0.0000,  0.0000,  0.0000],
                     [ 0.0000,  0.0000,  0.0000,  0.0000]]])
        """
        return super().diag_embed(offset=offset, dim1=dim1, dim2=dim2)

    @return_tensor_wrapper
    def diagflat(self, offset=0) -> 'Tensor':
        """
        diagflat(input, offset=0) -> Tensor

        - If :attr:`input` is a vector (1-D tensor), then returns a 2-D square tensor
          with the elements of :attr:`input` as the diagonal.
        - If :attr:`input` is a tensor with more than one dimension, then returns a
          2-D tensor with diagonal elements equal to a flattened :attr:`input`.

        The argument :attr:`offset` controls which diagonal to consider:

        - If :attr:`offset` = 0, it is the main diagonal.
        - If :attr:`offset` > 0, it is above the main diagonal.
        - If :attr:`offset` < 0, it is below the main diagonal.

        Args:
            input (Tensor): the input tensor.
            offset (int, optional): the diagonal to consider. Default: 0 (main
                diagonal).

        Examples::

            >>> a = torch.randn(3)
            >>> a
            tensor([-0.2956, -0.9068,  0.1695])
            >>> torch.diagflat(a)
            tensor([[-0.2956,  0.0000,  0.0000],
                    [ 0.0000, -0.9068,  0.0000],
                    [ 0.0000,  0.0000,  0.1695]])
            >>> torch.diagflat(a, 1)
            tensor([[ 0.0000, -0.2956,  0.0000,  0.0000],
                    [ 0.0000,  0.0000, -0.9068,  0.0000],
                    [ 0.0000,  0.0000,  0.0000,  0.1695],
                    [ 0.0000,  0.0000,  0.0000,  0.0000]])

            >>> a = torch.randn(2, 2)
            >>> a
            tensor([[ 0.2094, -0.3018],
                    [-0.1516,  1.9342]])
            >>> torch.diagflat(a)
            tensor([[ 0.2094,  0.0000,  0.0000,  0.0000],
                    [ 0.0000, -0.3018,  0.0000,  0.0000],
                    [ 0.0000,  0.0000, -0.1516,  0.0000],
                    [ 0.0000,  0.0000,  0.0000,  1.9342]])
        """
        return super().diagflat(offset=offset)

    @return_tensor_wrapper
    def diagonal(self, offset=0, dim1=0, dim2=1) -> 'Tensor':
        """
        diagonal(input, offset=0, dim1=0, dim2=1) -> Tensor

        Returns a partial view of :attr:`input` with the its diagonal elements
        with respect to :attr:`dim1` and :attr:`dim2` appended as a dimension
        at the end of the shape.

        The argument :attr:`offset` controls which diagonal to consider:

        - If :attr:`offset` = 0, it is the main diagonal.
        - If :attr:`offset` > 0, it is above the main diagonal.
        - If :attr:`offset` < 0, it is below the main diagonal.

        Applying :meth:`torch.diag_embed` to the output of this function with
        the same arguments yields a diagonal matrix with the diagonal entries
        of the input. However, :meth:`torch.diag_embed` has different default
        dimensions, so those need to be explicitly specified.

        Args:
            input (Tensor): the input tensor. Must be at least 2-dimensional.
            offset (int, optional): which diagonal to consider. Default: 0
                (main diagonal).
            dim1 (int, optional): first dimension with respect to which to
                take diagonal. Default: 0.
            dim2 (int, optional): second dimension with respect to which to
                take diagonal. Default: 1.

        .. note::  To take a batch diagonal, pass in dim1=-2, dim2=-1.

        Examples::

            >>> a = torch.randn(3, 3)
            >>> a
            tensor([[-1.0854,  1.1431, -0.1752],
                    [ 0.8536, -0.0905,  0.0360],
                    [ 0.6927, -0.3735, -0.4945]])


            >>> torch.diagonal(a, 0)
            tensor([-1.0854, -0.0905, -0.4945])


            >>> torch.diagonal(a, 1)
            tensor([ 1.1431,  0.0360])


            >>> x = torch.randn(2, 5, 4, 2)
            >>> torch.diagonal(x, offset=-1, dim1=1, dim2=2)
            tensor([[[-1.2631,  0.3755, -1.5977, -1.8172],
                     [-1.1065,  1.0401, -0.2235, -0.7938]],

                    [[-1.7325, -0.3081,  0.6166,  0.2335],
                     [ 1.0500,  0.7336, -0.3836, -1.1015]]])
        """
        return super().diagonal(offset=offset, dim1=dim1, dim2=dim2)

    @return_tensor_wrapper
    def digamma(self) -> 'Tensor':
        """
        digamma(input, out=None) -> Tensor

        Computes the logarithmic derivative of the gamma function on `input`.

        .. math::
            \psi(x) = \frac{d}{dx} \ln\left(\Gamma\left(x\right)\right) = \frac{\Gamma'(x)}{\Gamma(x)}

        Args:
            input (Tensor): the tensor to compute the digamma function on

        Example::

            >>> a = torch.tensor([1, 0.5])
            >>> torch.digamma(a)
            tensor([-0.5772, -1.9635])
        """
        return super().digamma()

    @return_tensor_wrapper
    def digamma_(self) -> 'Tensor':
        """
        digamma_() -> Tensor

        In-place version of :meth:`~Tensor.digamma`
        """
        return super().digamma_()

    @return_tensor_wrapper
    def dist(self, other, p=2) -> 'Tensor':
        """
        dist(input, other, p=2) -> Tensor

        Returns the p-norm of (:attr:`input` - :attr:`other`)

        The shapes of :attr:`input` and :attr:`other` must be
        :ref:`broadcastable <broadcasting-semantics>`.

        Args:
            input (Tensor): the input tensor.
            other (Tensor): the Right-hand-side input tensor
            p (float, optional): the norm to be computed

        Example::

            >>> x = torch.randn(4)
            >>> x
            tensor([-1.5393, -0.8675,  0.5916,  1.6321])
            >>> y = torch.randn(4)
            >>> y
            tensor([ 0.0967, -1.0511,  0.6295,  0.8360])
            >>> torch.dist(x, y, 3.5)
            tensor(1.6727)
            >>> torch.dist(x, y, 3)
            tensor(1.6973)
            >>> torch.dist(x, y, 0)
            tensor(inf)
            >>> torch.dist(x, y, 1)
            tensor(2.6537)
        """
        return super().dist(other, p=p)

    @return_tensor_wrapper
    def div(self, value) -> 'Tensor':
        """
        div(input, other, out=None) -> Tensor

        Divides each element of the input ``input`` with the scalar ``other`` and
        returns a new resulting tensor.

        .. warning::
            Integer division using div is no longer supported, and in a future release
            div will perform true division as in Python 3. Use :func:`torch.true_divide`
            or :func:`torch.floor_divide` (// in Python), instead.

        .. math::
            \text{out}_i = \frac{\text{input}_i}{\text{other}}

        If the :class:`torch.dtype` of ``input`` and ``other`` differ, the
        :class:`torch.dtype` of the result tensor is determined following rules
        described in the type promotion :ref:`documentation <type-promotion-doc>`. If
        ``out`` is specified, the result must be :ref:`castable <type-promotion-doc>`
        to the :class:`torch.dtype` of the specified output tensor. Integral division
        by zero leads to undefined behavior.

        Args:
            input (Tensor): the input tensor.
            other (Number): the number to be divided to each element of ``input``

        Keyword args:
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(5)
            >>> a
            tensor([ 0.3810,  1.2774, -0.2972, -0.3719,  0.4637])
            >>> torch.div(a, 0.5)
            tensor([ 0.7620,  2.5548, -0.5944, -0.7439,  0.9275])

        .. function:: div(input, other, out=None) -> Tensor

        Each element of the tensor ``input`` is divided by each element of the tensor
        ``other``. The resulting tensor is returned.

        .. math::
            \text{out}_i = \frac{\text{input}_i}{\text{other}_i}

        The shapes of ``input`` and ``other`` must be :ref:`broadcastable
        <broadcasting-semantics>`. If the :class:`torch.dtype` of ``input`` and
        ``other`` differ, the :class:`torch.dtype` of the result tensor is determined
        following rules described in the type promotion :ref:`documentation
        <type-promotion-doc>`. If ``out`` is specified, the result must be
        :ref:`castable <type-promotion-doc>` to the :class:`torch.dtype` of the
        specified output tensor. Integral division by zero leads to undefined behavior.

        Args:
            input (Tensor): the numerator tensor
            other (Tensor): the denominator tensor

        Keyword args:
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4, 4)
            >>> a
            tensor([[-0.3711, -1.9353, -0.4605, -0.2917],
                    [ 0.1815, -1.0111,  0.9805, -1.5923],
                    [ 0.1062,  1.4581,  0.7759, -1.2344],
                    [-0.1830, -0.0313,  1.1908, -1.4757]])
            >>> b = torch.randn(4)
            >>> b
            tensor([ 0.8032,  0.2930, -0.8113, -0.2308])
            >>> torch.div(a, b)
            tensor([[-0.4620, -6.6051,  0.5676,  1.2637],
                    [ 0.2260, -3.4507, -1.2086,  6.8988],
                    [ 0.1322,  4.9764, -0.9564,  5.3480],
                    [-0.2278, -0.1068, -1.4678,  6.3936]])
        """
        return super().div(value)

    @return_tensor_wrapper
    def div_(self, value) -> 'Tensor':
        """
        div_(value) -> Tensor

        In-place version of :meth:`~Tensor.div`
        """
        return super().div_(value)

    @return_tensor_wrapper
    def dot(self, tensor2) -> 'Tensor':
        """
        dot(input, tensor) -> Tensor

        Computes the dot product (inner product) of two tensors.

        .. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.

        Example::

            >>> torch.dot(torch.tensor([2, 3]), torch.tensor([2, 1]))
            tensor(7)
        """
        return super().dot(tensor2)

    @return_tensor_wrapper
    def double(self, memory_format=torch.preserve_format) -> 'Tensor':
        """
        double(memory_format=torch.preserve_format) -> Tensor

        ``self.double()`` is equivalent to ``self.to(torch.float64)``. See :func:`to`.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
        return super().double(memory_format=memory_format)

    @return_tensor_wrapper
    def eq(self, other) -> 'Tensor':
        """
        eq(input, other, out=None) -> Tensor

        Computes element-wise equality

        The second argument can be a number or a tensor whose shape is
        :ref:`broadcastable <broadcasting-semantics>` with the first argument.

        Args:
            input (Tensor): the tensor to compare
            other (Tensor or float): the tensor or value to compare
            out (Tensor, optional): the output tensor. Must be a `ByteTensor`

        Returns:
            Tensor: A ``torch.BoolTensor`` containing a True at each location where comparison is true

        Example::

            >>> torch.eq(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
            tensor([[ True, False],
                    [False, True]])
        """
        return super().eq(other)

    @return_tensor_wrapper
    def eq_(self, other) -> 'Tensor':
        """
        eq_(other) -> Tensor

        In-place version of :meth:`~Tensor.eq`
        """
        return super().eq_(other)

    @return_tensor_wrapper
    def erf(self) -> 'Tensor':
        """
        erf(input, out=None) -> Tensor

        Computes the error function of each element. The error function is defined as follows:

        .. math::
            \mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2} dt

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> torch.erf(torch.tensor([0, -1., 10.]))
            tensor([ 0.0000, -0.8427,  1.0000])
        """
        return super().erf()

    @return_tensor_wrapper
    def erf_(self) -> 'Tensor':
        """
        erf_() -> Tensor

        In-place version of :meth:`~Tensor.erf`
        """
        return super().erf_()

    @return_tensor_wrapper
    def erfc(self) -> 'Tensor':
        """
        erfc(input, out=None) -> Tensor

        Computes the complementary error function of each element of :attr:`input`.
        The complementary error function is defined as follows:

        .. math::
            \mathrm{erfc}(x) = 1 - \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2} dt

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> torch.erfc(torch.tensor([0, -1., 10.]))
            tensor([ 1.0000, 1.8427,  0.0000])
        """
        return super().erfc()

    @return_tensor_wrapper
    def erfc_(self) -> 'Tensor':
        """
        erfc_() -> Tensor

        In-place version of :meth:`~Tensor.erfc`
        """
        return super().erfc_()

    @return_tensor_wrapper
    def erfinv(self) -> 'Tensor':
        """
        erfinv(input, out=None) -> Tensor

        Computes the inverse error function of each element of :attr:`input`.
        The inverse error function is defined in the range :math:`(-1, 1)` as:

        .. math::
            \mathrm{erfinv}(\mathrm{erf}(x)) = x

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> torch.erfinv(torch.tensor([0, 0.5, -1.]))
            tensor([ 0.0000,  0.4769,    -inf])
        """
        return super().erfinv()

    @return_tensor_wrapper
    def erfinv_(self) -> 'Tensor':
        """
        erfinv_() -> Tensor

        In-place version of :meth:`~Tensor.erfinv`
        """
        return super().erfinv_()

    @return_tensor_wrapper
    def exp(self) -> 'Tensor':
        """
        exp(input, out=None) -> Tensor

        Returns a new tensor with the exponential of the elements
        of the input tensor :attr:`input`.

        .. math::
            y_{i} = e^{x_{i}}

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> torch.exp(torch.tensor([0, math.log(2.)]))
            tensor([ 1.,  2.])
        """
        return super().exp()

    @return_tensor_wrapper
    def exp_(self) -> 'Tensor':
        """
        exp_() -> Tensor

        In-place version of :meth:`~Tensor.exp`
        """
        return super().exp_()

    @return_tensor_wrapper
    def expand(self, *sizes, strict: bool = False) -> 'Tensor':
        """
        expand(*sizes) -> Tensor

        Returns a new view of the :attr:`self` tensor with singleton dimensions expanded
        to a larger size.

        Passing -1 as the size for a dimension means not changing the size of
        that dimension.

        Tensor can be also expanded to a larger number of dimensions, and the
        new ones will be appended at the front. For the new dimensions, the
        size cannot be set to -1.

        Expanding a tensor does not allocate new memory, but only creates a
        new view on the existing tensor where a dimension of size one is
        expanded to a larger size by setting the ``stride`` to 0. Any dimension
        of size 1 can be expanded to an arbitrary value without allocating new
        memory.

        Args:
            *sizes (torch.Size or int...): the desired expanded size

        .. warning::

            More than one element of an expanded tensor may refer to a single
            memory location. As a result, in-place operations (especially ones that
            are vectorized) may result in incorrect behavior. If you need to write
            to the tensors, please clone them first.

        Example::

            >>> x = torch.tensor([[1], [2], [3]])
            >>> x.size()
            torch.Size([3, 1])
            >>> x.expand(3, 4)
            tensor([[ 1,  1,  1,  1],
                    [ 2,  2,  2,  2],
                    [ 3,  3,  3,  3]])
            >>> x.expand(-1, 4)   # -1 means not changing the size of that dimension
            tensor([[ 1,  1,  1,  1],
                    [ 2,  2,  2,  2],
                    [ 3,  3,  3,  3]])
        """
        if strict:
            return super().expand(*sizes)
        if len(sizes) == 1 and iterable(sizes[0]):
            sizes = tuple(sizes[0])

        assert len(sizes) >= self.dim()
        assert 0 not in sizes

        if len(sizes) == self.dim():
            return super().expand(*sizes)
        if sizes.count(-1) == self.dim():
            new_dim = _replace_key_with_sequence(tuple(1 if x > 0 else -1 for x in sizes), self.shape)
            return self.view(*new_dim).expand(*sizes)
        if sizes.count(-1) == 0:
            return self.expand(*_replace_sequence_with_key(sizes, self.shape))
        return self.expand(sizes)

    @return_tensor_wrapper
    def expand_as(self, other) -> 'Tensor':
        """
        expand_as(other) -> Tensor

        Expand this tensor to the same size as :attr:`other`.
        ``self.expand_as(other)`` is equivalent to ``self.expand(other.size())``.

        Please see :meth:`~Tensor.expand` for more information about ``expand``.

        Args:
            other (:class:`torch.Tensor`): The result tensor has the same size
                as :attr:`other`.
        """
        return self.expand(other.shape)

    @return_tensor_wrapper
    def expm1(self) -> 'Tensor':
        """
        expm1(input, out=None) -> Tensor

        Returns a new tensor with the exponential of the elements minus 1
        of :attr:`input`.

        .. math::
            y_{i} = e^{x_{i}} - 1

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> torch.expm1(torch.tensor([0, math.log(2.)]))
            tensor([ 0.,  1.])
        """
        return super().expm1()

    @return_tensor_wrapper
    def expm1_(self) -> 'Tensor':
        """
        expm1_() -> Tensor

        In-place version of :meth:`~Tensor.expm1`
        """
        return super().expm1_()

    @return_tensor_wrapper
    def exponential_(self, lambd=1, *, generator=None) -> 'Tensor':
        """
        exponential_(lambd=1, *, generator=None) -> Tensor

        Fills :attr:`self` tensor with elements drawn from the exponential distribution:

        .. math::

            f(x) = \lambda e^{-\lambda x}
        """
        return super().exponential_(lambd=lambd, generator=generator)

    @return_tensor_wrapper
    def fft(self, signal_ndim, normalized=False) -> 'Tensor':
        """
        fft(input, signal_ndim, normalized=False) -> Tensor

        Complex-to-complex Discrete Fourier Transform

        This method computes the complex-to-complex discrete Fourier transform.
        Ignoring the batch dimensions, it computes the following expression:

        .. math::
            X[\omega_1, \dots, \omega_d] =
                \sum_{n_1=0}^{N_1-1} \dots \sum_{n_d=0}^{N_d-1} x[n_1, \dots, n_d]
                 e^{-j\ 2 \pi \sum_{i=0}^d \frac{\omega_i n_i}{N_i}},

        where :math:`d` = :attr:`signal_ndim` is number of dimensions for the
        signal, and :math:`N_i` is the size of signal dimension :math:`i`.

        This method supports 1D, 2D and 3D complex-to-complex transforms, indicated
        by :attr:`signal_ndim`. :attr:`input` must be a tensor with last dimension
        of size 2, representing the real and imaginary components of complex
        numbers, and should have at least ``signal_ndim + 1`` dimensions with optionally
        arbitrary number of leading batch dimensions. If :attr:`normalized` is set to
        ``True``, this normalizes the result by dividing it with
        :math:`\sqrt{\prod_{i=1}^K N_i}` so that the operator is unitary.

        Returns the real and the imaginary parts together as one tensor of the same
        shape of :attr:`input`.

        The inverse of this function is :func:`~torch.ifft`.

        .. note::
            For CUDA tensors, an LRU cache is used for cuFFT plans to speed up
            repeatedly running FFT methods on tensors of same geometry with same
            configuration. See :ref:`cufft-plan-cache` for more details on how to
            monitor and control the cache.

        .. warning::
            For CPU tensors, this method is currently only available with MKL. Use
            :func:`torch.backends.mkl.is_available` to check if MKL is installed.

        Arguments:
            input (Tensor): the input tensor of at least :attr:`signal_ndim` ``+ 1``
                dimensions
            signal_ndim (int): the number of dimensions in each signal.
                :attr:`signal_ndim` can only be 1, 2 or 3
            normalized (bool, optional): controls whether to return normalized results.
                Default: ``False``

        Returns:
            Tensor: A tensor containing the complex-to-complex Fourier transform result

        Example::

            >>> # unbatched 2D FFT
            >>> x = torch.randn(4, 3, 2)
            >>> torch.fft(x, 2)
            tensor([[[-0.0876,  1.7835],
                     [-2.0399, -2.9754],
                     [ 4.4773, -5.0119]],

                    [[-1.5716,  2.7631],
                     [-3.8846,  5.2652],
                     [ 0.2046, -0.7088]],

                    [[ 1.9938, -0.5901],
                     [ 6.5637,  6.4556],
                     [ 2.9865,  4.9318]],

                    [[ 7.0193,  1.1742],
                     [-1.3717, -2.1084],
                     [ 2.0289,  2.9357]]])
            >>> # batched 1D FFT
            >>> torch.fft(x, 1)
            tensor([[[ 1.8385,  1.2827],
                     [-0.1831,  1.6593],
                     [ 2.4243,  0.5367]],

                    [[-0.9176, -1.5543],
                     [-3.9943, -2.9860],
                     [ 1.2838, -2.9420]],

                    [[-0.8854, -0.6860],
                     [ 2.4450,  0.0808],
                     [ 1.3076, -0.5768]],

                    [[-0.1231,  2.7411],
                     [-0.3075, -1.7295],
                     [-0.5384, -2.0299]]])
            >>> # arbitrary number of batch dimensions, 2D FFT
            >>> x = torch.randn(3, 3, 5, 5, 2)
            >>> y = torch.fft(x, 2)
            >>> y.shape
            torch.Size([3, 3, 5, 5, 2])
        """
        return super().fft(signal_ndim, normalized=normalized)

    @return_tensor_wrapper
    def fill_(self, value) -> 'Tensor':
        """
        fill_(value) -> Tensor

        Fills :attr:`self` tensor with the specified value.
        """
        return super().fill_(value)

    @return_tensor_wrapper
    def fill_diagonal_(self, fill_value, wrap=False) -> 'Tensor':
        """
        fill_diagonal_(fill_value, wrap=False) -> Tensor

        Fill the main diagonal of a tensor that has at least 2-dimensions.
        When dims>2, all dimensions of input must be of equal length.
        This function modifies the input tensor in-place, and returns the input tensor.

        Arguments:
            fill_value (Scalar): the fill value
            wrap (bool): the diagonal 'wrapped' after N columns for tall matrices.

        Example::

            >>> a = torch.zeros(3, 3)
            >>> a.fill_diagonal_(5)
            tensor([[5., 0., 0.],
                    [0., 5., 0.],
                    [0., 0., 5.]])
            >>> b = torch.zeros(7, 3)
            >>> b.fill_diagonal_(5)
            tensor([[5., 0., 0.],
                    [0., 5., 0.],
                    [0., 0., 5.],
                    [0., 0., 0.],
                    [0., 0., 0.],
                    [0., 0., 0.],
                    [0., 0., 0.]])
            >>> c = torch.zeros(7, 3)
            >>> c.fill_diagonal_(5, wrap=True)
            tensor([[5., 0., 0.],
                    [0., 5., 0.],
                    [0., 0., 5.],
                    [0., 0., 0.],
                    [5., 0., 0.],
                    [0., 5., 0.],
                    [0., 0., 5.]])
        """
        return super().fill_diagonal_(fill_value, wrap=wrap)

    @return_tensor_wrapper
    def flatten(self, input, start_dim=0, end_dim=-1) -> 'Tensor':
        """
        flatten(input, start_dim=0, end_dim=-1) -> Tensor

        see :func:`torch.flatten`
        """
        return super().flatten(input, start_dim=start_dim, end_dim=end_dim)

    @return_tensor_wrapper
    def flip(self, dims) -> 'Tensor':
        """
        flip(input, dims) -> Tensor

        Reverse the order of a n-D tensor along given axis in dims.

        Args:
            input (Tensor): the input tensor.
            dims (a list or tuple): axis to flip on

        Example::

            >>> x = torch.arange(8).view(2, 2, 2)
            >>> x
            tensor([[[ 0,  1],
                     [ 2,  3]],

                    [[ 4,  5],
                     [ 6,  7]]])
            >>> torch.flip(x, [0, 1])
            tensor([[[ 6,  7],
                     [ 4,  5]],

                    [[ 2,  3],
                     [ 0,  1]]])
        """
        return super().flip(dims)

    @return_tensor_wrapper
    def fliplr(self) -> 'Tensor':
        """
        fliplr(input) -> Tensor

        Flip array in the left/right direction, returning a new tensor.

        Flip the entries in each row in the left/right direction.
        Columns are preserved, but appear in a different order than before.

        Note:
            Equivalent to input[:,::-1]. Requires the array to be at least 2-D.

        Args:
            input (Tensor): Must be at least 2-dimensional.

        Example::

            >>> x = torch.arange(4).view(2, 2)
            >>> x
            tensor([[0, 1],
                    [2, 3]])
            >>> torch.fliplr(x)
            tensor([[1, 0],
                    [3, 2]])
        """
        return super().fliplr()

    @return_tensor_wrapper
    def flipud(self) -> 'Tensor':
        """
        flipud(input) -> Tensor

        Flip array in the up/down direction, returning a new tensor.

        Flip the entries in each column in the up/down direction.
        Rows are preserved, but appear in a different order than before.

        Note:
            Equivalent to input[::-1,...]. Requires the array to be at least 1-D.

        Args:
            input (Tensor): Must be at least 1-dimensional.

        Example::

            >>> x = torch.arange(4).view(2, 2)
            >>> x
            tensor([[0, 1],
                    [2, 3]])
            >>> torch.flipud(x)
            tensor([[2, 3],
                    [0, 1]])
        """
        return super().flipud()

    def _float_torch(self, memory_format=torch.preserve_format) -> 'Tensor':
        """
        float(memory_format=torch.preserve_format) -> Tensor

        ``self.float()`` is equivalent to ``self.to(torch.float32)``. See :func:`to`.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
        return super().float(memory_format=memory_format)

    @return_tensor_wrapper
    def float(self, memory_format=torch.preserve_format) -> 'Tensor':
        """
        float(memory_format=torch.preserve_format) -> Tensor

        ``self.float()`` is equivalent to ``self.to(torch.float32)``. See :func:`to`.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
        return super().float(memory_format=memory_format)

    @return_tensor_wrapper
    def floor(self) -> 'Tensor':
        """
        floor(input, out=None) -> Tensor

        Returns a new tensor with the floor of the elements of :attr:`input`,
        the largest integer less than or equal to each element.

        .. math::
            \text{out}_{i} = \left\lfloor \text{input}_{i} \right\rfloor

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4)
            >>> a
            tensor([-0.8166,  1.5308, -0.2530, -0.2091])
            >>> torch.floor(a)
            tensor([-1.,  1., -1., -1.])
        """
        return super().floor()

    @return_tensor_wrapper
    def floor_(self) -> 'Tensor':
        """
        floor_() -> Tensor

        In-place version of :meth:`~Tensor.floor`
        """
        return super().floor_()

    @return_tensor_wrapper
    def floor_divide(self, value) -> 'Tensor':
        """
        floor_divide(input, other, out=None) -> Tensor

        Return the division of the inputs rounded down to the nearest integer. See :func:`torch.div`
        for type promotion and broadcasting rules.

        .. math::
            \text{{out}}_i = \left\lfloor \frac{{\text{{input}}_i}}{{\text{{other}}_i}} \right\rfloor


        Args:
            input (Tensor): the numerator tensor
            other (Tensor or Scalar): the denominator

        Keyword args:
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.tensor([4.0, 3.0])
            >>> b = torch.tensor([2.0, 2.0])
            >>> torch.floor_divide(a, b)
            tensor([2.0, 1.0])
            >>> torch.floor_divide(a, 1.4)
            tensor([2.0, 2.0])
        """
        return super().floor_divide(value)

    @return_tensor_wrapper
    def floor_divide_(self, value) -> 'Tensor':
        """
        floor_divide_(value) -> Tensor

        In-place version of :meth:`~Tensor.floor_divide`
        """
        return super().floor_divide_(value)

    @return_tensor_wrapper
    def fmod(self, divisor) -> 'Tensor':
        """
        fmod(input, other, out=None) -> Tensor

        Computes the element-wise remainder of division.

        The dividend and divisor may contain both for integer and floating point
        numbers. The remainder has the same sign as the dividend :attr:`input`.

        When :attr:`other` is a tensor, the shapes of :attr:`input` and
        :attr:`other` must be :ref:`broadcastable <broadcasting-semantics>`.

        Args:
            input (Tensor): the dividend
            other (Tensor or float): the divisor, which may be either a number or a tensor of the same shape as the dividend
            out (Tensor, optional): the output tensor.

        Example::

            >>> torch.fmod(torch.tensor([-3., -2, -1, 1, 2, 3]), 2)
            tensor([-1., -0., -1.,  1.,  0.,  1.])
            >>> torch.fmod(torch.tensor([1., 2, 3, 4, 5]), 1.5)
            tensor([ 1.0000,  0.5000,  0.0000,  1.0000,  0.5000])
        """
        return super().fmod(divisor)

    @return_tensor_wrapper
    def fmod_(self, divisor) -> 'Tensor':
        """
        fmod_(divisor) -> Tensor

        In-place version of :meth:`~Tensor.fmod`
        """
        return super().fmod_(divisor)

    @return_tensor_wrapper
    def frac(self) -> 'Tensor':
        """
        frac(input, out=None) -> Tensor

        Computes the fractional portion of each element in :attr:`input`.

        .. math::
            \text{out}_{i} = \text{input}_{i} - \left\lfloor |\text{input}_{i}| \right\rfloor * \operatorname{sgn}(\text{input}_{i})

        Example::

            >>> torch.frac(torch.tensor([1, 2.5, -3.2]))
            tensor([ 0.0000,  0.5000, -0.2000])
        """
        return super().frac()

    @return_tensor_wrapper
    def frac_(self) -> 'Tensor':
        """
        frac_() -> Tensor

        In-place version of :meth:`~Tensor.frac`
        """
        return super().frac_()

    @return_tensor_wrapper
    def gather(self, dim, index) -> 'Tensor':
        """
        gather(input, dim, index, out=None, sparse_grad=False) -> Tensor

        Gathers values along an axis specified by `dim`.

        For a 3-D tensor the output is specified by::

            out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
            out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
            out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

        If :attr:`input` is an n-dimensional tensor with size
        :math:`(x_0, x_1..., x_{i-1}, x_i, x_{i+1}, ..., x_{n-1})`
        and ``dim = i``, then :attr:`index` must be an :math:`n`-dimensional tensor with
        size :math:`(x_0, x_1, ..., x_{i-1}, y, x_{i+1}, ..., x_{n-1})` where :math:`y \geq 1`
        and :attr:`out` will have the same size as :attr:`index`.

        Args:
            input (Tensor): the source tensor
            dim (int): the axis along which to index
            index (LongTensor): the indices of elements to gather
            out (Tensor, optional): the destination tensor
            sparse_grad(bool,optional): If ``True``, gradient w.r.t. :attr:`input` will be a sparse tensor.

        Example::

            >>> t = torch.tensor([[1,2],[3,4]])
            >>> torch.gather(t, 1, torch.tensor([[0,0],[1,0]]))
            tensor([[ 1,  1],
                    [ 4,  3]])
        """
        return super().gather(dim, index)

    @return_tensor_wrapper
    def ge(self, other) -> 'Tensor':
        """
        ge(input, other, out=None) -> Tensor

        Computes :math:`\text{input} \geq \text{other}` element-wise.

        The second argument can be a number or a tensor whose shape is
        :ref:`broadcastable <broadcasting-semantics>` with the first argument.

        Args:
            input (Tensor): the tensor to compare
            other (Tensor or float): the tensor or value to compare
            out (Tensor, optional): the output tensor that must be a `BoolTensor`

        Returns:
            Tensor: A ``torch.BoolTensor`` containing a True at each location where comparison is true

        Example::

            >>> torch.ge(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
            tensor([[True, True], [False, True]])
        """
        return super().ge(other)

    @return_tensor_wrapper
    def ge_(self, other) -> 'Tensor':
        """
        ge_(other) -> Tensor

        In-place version of :meth:`~Tensor.ge`
        """
        return super().ge_(other)

    @return_tensor_wrapper
    def geometric_(self, p, *, generator=None) -> 'Tensor':
        """
        geometric_(p, *, generator=None) -> Tensor

        Fills :attr:`self` tensor with elements drawn from the geometric distribution:

        .. math::

            f(X=k) = p^{k - 1} (1 - p)
        """
        return super().geometric_(p, generator=generator)

    @return_tensor_wrapper
    def ger(self, vec2) -> 'Tensor':
        """
        ger(input, vec2, out=None) -> Tensor

        Outer product of :attr:`input` and :attr:`vec2`.
        If :attr:`input` is a vector of size :math:`n` and :attr:`vec2` is a vector of
        size :math:`m`, then :attr:`out` must be a matrix of size :math:`(n \times m)`.

        .. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.

        Args:
            input (Tensor): 1-D input vector
            vec2 (Tensor): 1-D input vector
            out (Tensor, optional): optional output matrix

        Example::

            >>> v1 = torch.arange(1., 5.)
            >>> v2 = torch.arange(1., 4.)
            >>> torch.ger(v1, v2)
            tensor([[  1.,   2.,   3.],
                    [  2.,   4.,   6.],
                    [  3.,   6.,   9.],
                    [  4.,   8.,  12.]])
        """
        return super().ger(vec2)

    @return_tensor_wrapper
    def gt(self, other) -> 'Tensor':
        """
        gt(input, other, out=None) -> Tensor

        Computes :math:`\text{input} > \text{other}` element-wise.

        The second argument can be a number or a tensor whose shape is
        :ref:`broadcastable <broadcasting-semantics>` with the first argument.

        Args:
            input (Tensor): the tensor to compare
            other (Tensor or float): the tensor or value to compare
            out (Tensor, optional): the output tensor that must be a `BoolTensor`

        Returns:
            Tensor: A ``torch.BoolTensor`` containing a True at each location where comparison is true

        Example::

            >>> torch.gt(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
            tensor([[False, True], [False, False]])
        """
        return super().gt(other)

    @return_tensor_wrapper
    def gt_(self, other) -> 'Tensor':
        """
        gt_(other) -> Tensor

        In-place version of :meth:`~Tensor.gt`
        """
        return super().gt_(other)

    @return_tensor_wrapper
    def half(self, memory_format=torch.preserve_format) -> 'Tensor':
        """
        half(memory_format=torch.preserve_format) -> Tensor

        ``self.half()`` is equivalent to ``self.to(torch.float16)``. See :func:`to`.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
        return super().half(memory_format=memory_format)

    @return_tensor_wrapper
    def hardshrink(self, lambd=0.5) -> 'Tensor':
        """
            hardshrink(input, lambd=0.5) -> Tensor

            Applies the hard shrinkage function element-wise

            See :class:`~torch.nn.Hardshrink` for more details.

        """
        return super().hardshrink(lambd=lambd)

    @return_tensor_wrapper
    def histc(self, bins=100, min=0, max=0) -> 'Tensor':
        """
        histc(input, bins=100, min=0, max=0, out=None) -> Tensor

        Computes the histogram of a tensor.

        The elements are sorted into equal width bins between :attr:`min` and
        :attr:`max`. If :attr:`min` and :attr:`max` are both zero, the minimum and
        maximum values of the data are used.

        Elements lower than min and higher than max are ignored.

        Args:
            input (Tensor): the input tensor.
            bins (int): number of histogram bins
            min (int): lower end of the range (inclusive)
            max (int): upper end of the range (inclusive)
            out (Tensor, optional): the output tensor.

        Returns:
            Tensor: Histogram represented as a tensor

        Example::

            >>> torch.histc(torch.tensor([1., 2, 1]), bins=4, min=0, max=3)
            tensor([ 0.,  2.,  1.,  0.])
        """
        return super().histc(bins=bins, min=min, max=max)

    @return_tensor_wrapper
    def ifft(self, signal_ndim, normalized=False) -> 'Tensor':
        """
        ifft(input, signal_ndim, normalized=False) -> Tensor

        Complex-to-complex Inverse Discrete Fourier Transform

        This method computes the complex-to-complex inverse discrete Fourier
        transform. Ignoring the batch dimensions, it computes the following
        expression:

        .. math::
            X[\omega_1, \dots, \omega_d] =
                \frac{1}{\prod_{i=1}^d N_i} \sum_{n_1=0}^{N_1-1} \dots \sum_{n_d=0}^{N_d-1} x[n_1, \dots, n_d]
                 e^{\ j\ 2 \pi \sum_{i=0}^d \frac{\omega_i n_i}{N_i}},

        where :math:`d` = :attr:`signal_ndim` is number of dimensions for the
        signal, and :math:`N_i` is the size of signal dimension :math:`i`.

        The argument specifications are almost identical with :func:`~torch.fft`.
        However, if :attr:`normalized` is set to ``True``, this instead returns the
        results multiplied by :math:`\sqrt{\prod_{i=1}^d N_i}`, to become a unitary
        operator. Therefore, to invert a :func:`~torch.fft`, the :attr:`normalized`
        argument should be set identically for :func:`~torch.fft`.

        Returns the real and the imaginary parts together as one tensor of the same
        shape of :attr:`input`.

        The inverse of this function is :func:`~torch.fft`.

        .. note::
            For CUDA tensors, an LRU cache is used for cuFFT plans to speed up
            repeatedly running FFT methods on tensors of same geometry with same
            configuration. See :ref:`cufft-plan-cache` for more details on how to
            monitor and control the cache.

        .. warning::
            For CPU tensors, this method is currently only available with MKL. Use
            :func:`torch.backends.mkl.is_available` to check if MKL is installed.

        Arguments:
            input (Tensor): the input tensor of at least :attr:`signal_ndim` ``+ 1``
                dimensions
            signal_ndim (int): the number of dimensions in each signal.
                :attr:`signal_ndim` can only be 1, 2 or 3
            normalized (bool, optional): controls whether to return normalized results.
                Default: ``False``

        Returns:
            Tensor: A tensor containing the complex-to-complex inverse Fourier transform result

        Example::

            >>> x = torch.randn(3, 3, 2)
            >>> x
            tensor([[[ 1.2766,  1.3680],
                     [-0.8337,  2.0251],
                     [ 0.9465, -1.4390]],

                    [[-0.1890,  1.6010],
                     [ 1.1034, -1.9230],
                     [-0.9482,  1.0775]],

                    [[-0.7708, -0.8176],
                     [-0.1843, -0.2287],
                     [-1.9034, -0.2196]]])
            >>> y = torch.fft(x, 2)
            >>> torch.ifft(y, 2)  # recover x
            tensor([[[ 1.2766,  1.3680],
                     [-0.8337,  2.0251],
                     [ 0.9465, -1.4390]],

                    [[-0.1890,  1.6010],
                     [ 1.1034, -1.9230],
                     [-0.9482,  1.0775]],

                    [[-0.7708, -0.8176],
                     [-0.1843, -0.2287],
                     [-1.9034, -0.2196]]])
        """
        return super().ifft(signal_ndim, normalized=normalized)

    @return_tensor_wrapper
    def index_add(self, tensor1, dim, index, tensor2) -> 'Tensor':
        """
        index_add(tensor1, dim, index, tensor2) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.index_add_`.
        `tensor1` corresponds to `self` in :meth:`torch.Tensor.index_add_`.
        """
        return super().index_add(tensor1, dim, index, tensor2)

    @return_tensor_wrapper
    def index_add_(self, dim, index, tensor) -> 'Tensor':
        """
        index_add_(dim, index, tensor) -> Tensor

        Accumulate the elements of :attr:`tensor` into the :attr:`self` tensor by adding
        to the indices in the order given in :attr:`index`. For example, if ``dim == 0``
        and ``index[i] == j``, then the ``i``\ th row of :attr:`tensor` is added to the
        ``j``\ th row of :attr:`self`.

        The :attr:`dim`\ th dimension of :attr:`tensor` must have the same size as the
        length of :attr:`index` (which must be a vector), and all other dimensions must
        match :attr:`self`, or an error will be raised.

        Note:
            In some circumstances when using the CUDA backend with CuDNN, this operator
            may select a nondeterministic algorithm to increase performance. If this is
            undesirable, you can try to make the operation deterministic (potentially at
            a performance cost) by setting ``torch.backends.cudnn.deterministic =
            True``.
            Please see the notes on :doc:`/notes/randomness` for background.

        Args:
            dim (int): dimension along which to index
            index (LongTensor): indices of :attr:`tensor` to select from
            tensor (Tensor): the tensor containing values to add

        Example::

            >>> x = torch.ones(5, 3)
            >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
            >>> index = torch.tensor([0, 4, 2])
            >>> x.index_add_(0, index, t)
            tensor([[  2.,   3.,   4.],
                    [  1.,   1.,   1.],
                    [  8.,   9.,  10.],
                    [  1.,   1.,   1.],
                    [  5.,   6.,   7.]])
        """
        return super().index_add_(dim, index, tensor)

    @return_tensor_wrapper
    def index_copy(self, tensor1, dim, index, tensor2) -> 'Tensor':
        """
        index_copy(tensor1, dim, index, tensor2) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.index_copy_`.
        `tensor1` corresponds to `self` in :meth:`torch.Tensor.index_copy_`.
        """
        return super().index_copy(tensor1, dim, index, tensor2)

    @return_tensor_wrapper
    def index_copy_(self, dim, index, tensor) -> 'Tensor':
        """
        index_copy_(dim, index, tensor) -> Tensor

        Copies the elements of :attr:`tensor` into the :attr:`self` tensor by selecting
        the indices in the order given in :attr:`index`. For example, if ``dim == 0``
        and ``index[i] == j``, then the ``i``\ th row of :attr:`tensor` is copied to the
        ``j``\ th row of :attr:`self`.

        The :attr:`dim`\ th dimension of :attr:`tensor` must have the same size as the
        length of :attr:`index` (which must be a vector), and all other dimensions must
        match :attr:`self`, or an error will be raised.

        Args:
            dim (int): dimension along which to index
            index (LongTensor): indices of :attr:`tensor` to select from
            tensor (Tensor): the tensor containing values to copy

        Example::

            >>> x = torch.zeros(5, 3)
            >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
            >>> index = torch.tensor([0, 4, 2])
            >>> x.index_copy_(0, index, t)
            tensor([[ 1.,  2.,  3.],
                    [ 0.,  0.,  0.],
                    [ 7.,  8.,  9.],
                    [ 0.,  0.,  0.],
                    [ 4.,  5.,  6.]])
        """
        return super().index_copy_(dim, index, tensor)

    @return_tensor_wrapper
    def index_fill(self, tensor1, dim, index, value) -> 'Tensor':
        """
        index_fill(tensor1, dim, index, value) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.index_fill_`.
        `tensor1` corresponds to `self` in :meth:`torch.Tensor.index_fill_`.
        """
        return super().index_fill(tensor1, dim, index, value)

    @return_tensor_wrapper
    def index_fill_(self, dim, index, val) -> 'Tensor':
        """
        index_fill_(dim, index, val) -> Tensor

        Fills the elements of the :attr:`self` tensor with value :attr:`val` by
        selecting the indices in the order given in :attr:`index`.

        Args:
            dim (int): dimension along which to index
            index (LongTensor): indices of :attr:`self` tensor to fill in
            val (float): the value to fill with

        Example::
            >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
            >>> index = torch.tensor([0, 2])
            >>> x.index_fill_(1, index, -1)
            tensor([[-1.,  2., -1.],
                    [-1.,  5., -1.],
                    [-1.,  8., -1.]])
        """
        return super().index_fill_(dim, index, val)

    @return_tensor_wrapper
    def index_put(self, tensor1, indices, value, accumulate=False) -> 'Tensor':
        """
        index_put(tensor1, indices, value, accumulate=False) -> Tensor

        Out-place version of :meth:`~Tensor.index_put_`.
        `tensor1` corresponds to `self` in :meth:`torch.Tensor.index_put_`.
        """
        return super().index_put(tensor1, indices, value, accumulate=accumulate)

    @return_tensor_wrapper
    def index_put_(self, indices, value, accumulate=False) -> 'Tensor':
        """
        index_put_(indices, value, accumulate=False) -> Tensor

        Puts values from the tensor :attr:`value` into the tensor :attr:`self` using
        the indices specified in :attr:`indices` (which is a tuple of Tensors). The
        expression ``tensor.index_put_(indices, value)`` is equivalent to
        ``tensor[indices] = value``. Returns :attr:`self`.

        If :attr:`accumulate` is ``True``, the elements in :attr:`value` are added to
        :attr:`self`. If accumulate is ``False``, the behavior is undefined if indices
        contain duplicate elements.

        Args:
            indices (tuple of LongTensor): tensors used to index into `self`.
            value (Tensor): tensor of same dtype as `self`.
            accumulate (bool): whether to accumulate into self
        """
        return super().index_put_(indices, value, accumulate=accumulate)

    @return_tensor_wrapper
    def index_select(self, dim, index) -> 'Tensor':
        """
        index_select(input, dim, index, out=None) -> Tensor

        Returns a new tensor which indexes the :attr:`input` tensor along dimension
        :attr:`dim` using the entries in :attr:`index` which is a `LongTensor`.

        The returned tensor has the same number of dimensions as the original tensor
        (:attr:`input`).  The :attr:`dim`\ th dimension has the same size as the length
        of :attr:`index`; other dimensions have the same size as in the original tensor.

        .. note:: The returned tensor does **not** use the same storage as the original
                  tensor.  If :attr:`out` has a different shape than expected, we
                  silently change it to the correct shape, reallocating the underlying
                  storage if necessary.

        Args:
            input (Tensor): the input tensor.
            dim (int): the dimension in which we index
            index (LongTensor): the 1-D tensor containing the indices to index
            out (Tensor, optional): the output tensor.

        Example::

            >>> x = torch.randn(3, 4)
            >>> x
            tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
                    [-0.4664,  0.2647, -0.1228, -1.1068],
                    [-1.1734, -0.6571,  0.7230, -0.6004]])
            >>> indices = torch.tensor([0, 2])
            >>> torch.index_select(x, 0, indices)
            tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
                    [-1.1734, -0.6571,  0.7230, -0.6004]])
            >>> torch.index_select(x, 1, indices)
            tensor([[ 0.1427, -0.5414],
                    [-0.4664, -0.1228],
                    [-1.1734,  0.7230]])
        """
        return super().index_select(dim, index)

    @return_tensor_wrapper
    def indices(self) -> 'Tensor':
        """
        indices() -> Tensor

        If :attr:`self` is a sparse COO tensor (i.e., with ``torch.sparse_coo`` layout),
        this returns a view of the contained indices tensor. Otherwise, this throws an
        error.

        See also :meth:`Tensor.values`.

        .. note::
          This method can only be called on a coalesced sparse tensor. See
          :meth:`Tensor.coalesce` for details.
        """
        return super().indices()

    @return_tensor_wrapper
    def int(self, memory_format=torch.preserve_format) -> 'Tensor':
        """
        int(memory_format=torch.preserve_format) -> Tensor

        ``self.int()`` is equivalent to ``self.to(torch.int32)``. See :func:`to`.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
        return super().int(memory_format=memory_format)

    @return_tensor_wrapper
    def int_repr(self) -> 'Tensor':
        """
        int_repr() -> Tensor

        Given a quantized Tensor,
        ``self.int_repr()`` returns a CPU Tensor with uint8_t as data type that stores the
        underlying uint8_t values of the given Tensor.
        """
        return super().int_repr()

    @return_tensor_wrapper
    def inverse(self) -> 'Tensor':
        """
        inverse(input, out=None) -> Tensor

        Takes the inverse of the square matrix :attr:`input`. :attr:`input` can be batches
        of 2D square tensors, in which case this function would return a tensor composed of
        individual inverses.

        .. note::

            Irrespective of the original strides, the returned tensors will be
            transposed, i.e. with strides like `input.contiguous().transpose(-2, -1).stride()`

        Args:
            input (Tensor): the input tensor of size :math:`(*, n, n)` where `*` is zero or more
                            batch dimensions
            out (Tensor, optional): the output tensor.

        Example::

            >>> x = torch.rand(4, 4)
            >>> y = torch.inverse(x)
            >>> z = torch.mm(x, y)
            >>> z
            tensor([[ 1.0000, -0.0000, -0.0000,  0.0000],
                    [ 0.0000,  1.0000,  0.0000,  0.0000],
                    [ 0.0000,  0.0000,  1.0000,  0.0000],
                    [ 0.0000, -0.0000, -0.0000,  1.0000]])
            >>> torch.max(torch.abs(z - torch.eye(4))) # Max non-zero
            tensor(1.1921e-07)
            >>> # Batched inverse example
            >>> x = torch.randn(2, 3, 4, 4)
            >>> y = torch.inverse(x)
            >>> z = torch.matmul(x, y)
            >>> torch.max(torch.abs(z - torch.eye(4).expand_as(x))) # Max non-zero
            tensor(1.9073e-06)
        """
        return super().inverse()

    @return_tensor_wrapper
    def irfft(self, signal_ndim, normalized=False, onesided=True, signal_sizes=None) -> 'Tensor':
        """
        irfft(input, signal_ndim, normalized=False, onesided=True, signal_sizes=None) -> Tensor

        Complex-to-real Inverse Discrete Fourier Transform

        This method computes the complex-to-real inverse discrete Fourier transform.
        It is mathematically equivalent with :func:`ifft` with differences only in
        formats of the input and output.

        The argument specifications are almost identical with :func:`~torch.ifft`.
        Similar to :func:`~torch.ifft`, if :attr:`normalized` is set to ``True``,
        this normalizes the result by multiplying it with
        :math:`\sqrt{\prod_{i=1}^K N_i}` so that the operator is unitary, where
        :math:`N_i` is the size of signal dimension :math:`i`.

        .. note::
            Due to the conjugate symmetry, :attr:`input` do not need to contain the full
            complex frequency values. Roughly half of the values will be sufficient, as
            is the case when :attr:`input` is given by :func:`~torch.rfft` with
            ``rfft(signal, onesided=True)``. In such case, set the :attr:`onesided`
            argument of this method to ``True``. Moreover, the original signal shape
            information can sometimes be lost, optionally set :attr:`signal_sizes` to be
            the size of the original signal (without the batch dimensions if in batched
            mode) to recover it with correct shape.

            Therefore, to invert an :func:`~torch.rfft`, the :attr:`normalized` and
            :attr:`onesided` arguments should be set identically for :func:`~torch.irfft`,
            and preferably a :attr:`signal_sizes` is given to avoid size mismatch. See the
            example below for a case of size mismatch.

            See :func:`~torch.rfft` for details on conjugate symmetry.

        The inverse of this function is :func:`~torch.rfft`.

        .. warning::
            Generally speaking, input to this function should contain values
            following conjugate symmetry. Note that even if :attr:`onesided` is
            ``True``, often symmetry on some part is still needed. When this
            requirement is not satisfied, the behavior of :func:`~torch.irfft` is
            undefined. Since :func:`torch.autograd.gradcheck` estimates numerical
            Jacobian with point perturbations, :func:`~torch.irfft` will almost
            certainly fail the check.

        .. note::
            For CUDA tensors, an LRU cache is used for cuFFT plans to speed up
            repeatedly running FFT methods on tensors of same geometry with same
            configuration. See :ref:`cufft-plan-cache` for more details on how to
            monitor and control the cache.

        .. warning::
            For CPU tensors, this method is currently only available with MKL. Use
            :func:`torch.backends.mkl.is_available` to check if MKL is installed.

        Arguments:
            input (Tensor): the input tensor of at least :attr:`signal_ndim` ``+ 1``
                dimensions
            signal_ndim (int): the number of dimensions in each signal.
                :attr:`signal_ndim` can only be 1, 2 or 3
            normalized (bool, optional): controls whether to return normalized results.
                Default: ``False``
            onesided (bool, optional): controls whether :attr:`input` was halfed to avoid
                redundancy, e.g., by :func:`rfft`. Default: ``True``
            signal_sizes (list or :class:`torch.Size`, optional): the size of the original
                signal (without batch dimension). Default: ``None``

        Returns:
            Tensor: A tensor containing the complex-to-real inverse Fourier transform result

        Example::

            >>> x = torch.randn(4, 4)
            >>> torch.rfft(x, 2, onesided=True).shape
            torch.Size([4, 3, 2])
            >>>
            >>> # notice that with onesided=True, output size does not determine the original signal size
            >>> x = torch.randn(4, 5)

            >>> torch.rfft(x, 2, onesided=True).shape
            torch.Size([4, 3, 2])
            >>>
            >>> # now we use the original shape to recover x
            >>> x
            tensor([[-0.8992,  0.6117, -1.6091, -0.4155, -0.8346],
                    [-2.1596, -0.0853,  0.7232,  0.1941, -0.0789],
                    [-2.0329,  1.1031,  0.6869, -0.5042,  0.9895],
                    [-0.1884,  0.2858, -1.5831,  0.9917, -0.8356]])
            >>> y = torch.rfft(x, 2, onesided=True)
            >>> torch.irfft(y, 2, onesided=True, signal_sizes=x.shape)  # recover x
            tensor([[-0.8992,  0.6117, -1.6091, -0.4155, -0.8346],
                    [-2.1596, -0.0853,  0.7232,  0.1941, -0.0789],
                    [-2.0329,  1.1031,  0.6869, -0.5042,  0.9895],
                    [-0.1884,  0.2858, -1.5831,  0.9917, -0.8356]])
        """
        return super().irfft(signal_ndim, normalized=normalized, onesided=onesided, signal_sizes=signal_sizes)

    @return_tensor_wrapper
    def isclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False) -> 'Tensor':
        """
        isclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False) -> Tensor

        Returns a new tensor with boolean elements representing if each element of
        :attr:`input` is "close" to the corresponding element of :attr:`other`.
        Closeness is defined as:

        .. math::
            \lvert \text{input} - \text{other} \rvert \leq \texttt{atol} + \texttt{rtol} \times \lvert \text{other} \rvert


        where :attr:`input` and :attr:`other` are finite. Where :attr:`input`
        and/or :attr:`other` are nonfinite they are close if and only if
        they are equal, with NaNs being considered equal to each other when
        :attr:`equal_nan` is True.

        Args:
            input (Tensor): first tensor to compare
            other (Tensor): second tensor to compare
            atol (float, optional): absolute tolerance. Default: 1e-08
            rtol (float, optional): relative tolerance. Default: 1e-05
            equal_nan (bool, optional): if ``True``, then two ``NaN`` s will be considered equal. Default: ``False``

        Examples::

            >>> torch.isclose(torch.tensor((1., 2, 3)), torch.tensor((1 + 1e-10, 3, 4)))
            tensor([ True, False, False])
            >>> torch.isclose(torch.tensor((float('inf'), 4)), torch.tensor((float('inf'), 6)), rtol=.5)
            tensor([True, True])
        """
        return super().isclose(other, rtol=rtol, atol=atol, equal_nan=equal_nan)

    @return_tensor_wrapper
    def isfinite(self) -> 'Tensor':
        """
        Returns a new tensor with boolean elements representing if each element is `finite` or not.

        Real values are finite when they are not NaN, negative infinity, or infinity.
        Complex values are finite when both their real and imaginary parts are finite.

            Arguments:
                tensor (Tensor): A tensor to check

            Returns:
                Tensor: ``A torch.Tensor with dtype torch.bool`` containing a True at each location of finite elements and False otherwise

            Example::

                >>> torch.isfinite(torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]))
                tensor([True,  False,  True,  False,  False])
        """
        return super().isfinite()

    @return_tensor_wrapper
    def isinf(self) -> 'Tensor':
        """
        Returns a new tensor with boolean elements representing if each element is `+/-INF` or not.
        Complex values are infinite when their real and/or imaginary part is infinite.

            Arguments:
                tensor (Tensor): A tensor to check

            Returns:
                Tensor: ``A torch.Tensor with dtype torch.bool`` containing a True at each location of `+/-INF` elements and False otherwise

            Example::

                >>> torch.isinf(torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]))
                tensor([False,  True,  False,  True,  False])
        """
        return super().isinf()

    @return_tensor_wrapper
    def isnan(self) -> 'Tensor':
        """
        Returns a new tensor with boolean elements representing if each element is `NaN` or not.
        Complex values are considered `NaN` when either their real and/or imaginary part is NaN.

        Arguments:
            input (Tensor): A tensor to check

        Returns:
            Tensor: A ``torch.BoolTensor`` containing a True at each location of `NaN` elements.

        Example::

            >>> torch.isnan(torch.tensor([1, float('nan'), 2]))
            tensor([False, True, False])
        """
        return super().isnan()

    @return_tensor_wrapper
    def le(self, other) -> 'Tensor':
        """
        le(input, other, out=None) -> Tensor

        Computes :math:`\text{input} \leq \text{other}` element-wise.

        The second argument can be a number or a tensor whose shape is
        :ref:`broadcastable <broadcasting-semantics>` with the first argument.

        Args:
            input (Tensor): the tensor to compare
            other (Tensor or float): the tensor or value to compare
            out (Tensor, optional): the output tensor that must be a `BoolTensor`

        Returns:
            Tensor: A ``torch.BoolTensor`` containing a True at each location where comparison is true

        Example::

            >>> torch.le(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
            tensor([[True, False], [True, True]])
        """
        return super().le(other)

    @return_tensor_wrapper
    def le_(self, other) -> 'Tensor':
        """
        le_(other) -> Tensor

        In-place version of :meth:`~Tensor.le`
        """
        return super().le_(other)

    @return_tensor_wrapper
    def lerp(self, end, weight) -> 'Tensor':
        """
        lerp(input, end, weight, out=None)

        Does a linear interpolation of two tensors :attr:`start` (given by :attr:`input`) and :attr:`end` based
        on a scalar or tensor :attr:`weight` and returns the resulting :attr:`out` tensor.

        .. math::
            \text{out}_i = \text{start}_i + \text{weight}_i \times (\text{end}_i - \text{start}_i)

        The shapes of :attr:`start` and :attr:`end` must be
        :ref:`broadcastable <broadcasting-semantics>`. If :attr:`weight` is a tensor, then
        the shapes of :attr:`weight`, :attr:`start`, and :attr:`end` must be :ref:`broadcastable <broadcasting-semantics>`.

        Args:
            input (Tensor): the tensor with the starting points
            end (Tensor): the tensor with the ending points
            weight (float or tensor): the weight for the interpolation formula
            out (Tensor, optional): the output tensor.

        Example::

            >>> start = torch.arange(1., 5.)
            >>> end = torch.empty(4).fill_(10)
            >>> start
            tensor([ 1.,  2.,  3.,  4.])
            >>> end
            tensor([ 10.,  10.,  10.,  10.])
            >>> torch.lerp(start, end, 0.5)
            tensor([ 5.5000,  6.0000,  6.5000,  7.0000])
            >>> torch.lerp(start, end, torch.full_like(start, 0.5))
            tensor([ 5.5000,  6.0000,  6.5000,  7.0000])
        """
        return super().lerp(end, weight)

    @return_tensor_wrapper
    def lerp_(self, end, weight) -> 'Tensor':
        """
        lerp_(end, weight) -> Tensor

        In-place version of :meth:`~Tensor.lerp`
        """
        return super().lerp_(end, weight)

    @return_tensor_wrapper
    def lgamma(self) -> 'Tensor':
        """
        lgamma(input, out=None) -> Tensor

        Computes the logarithm of the gamma function on :attr:`input`.

        .. math::
            \text{out}_{i} = \log \Gamma(\text{input}_{i})

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.arange(0.5, 2, 0.5)
            >>> torch.lgamma(a)
            tensor([ 0.5724,  0.0000, -0.1208])
        """
        return super().lgamma()

    @return_tensor_wrapper
    def lgamma_(self) -> 'Tensor':
        """
        lgamma_() -> Tensor

        In-place version of :meth:`~Tensor.lgamma`
        """
        return super().lgamma_()

    @return_tensor_wrapper
    def log(self) -> 'Tensor':
        """
        log(input, out=None) -> Tensor

        Returns a new tensor with the natural logarithm of the elements
        of :attr:`input`.

        .. math::
            y_{i} = \log_{e} (x_{i})

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(5)
            >>> a
            tensor([-0.7168, -0.5471, -0.8933, -1.4428, -0.1190])
            >>> torch.log(a)
            tensor([ nan,  nan,  nan,  nan,  nan])
        """
        return super().log()

    @return_tensor_wrapper
    def log10(self) -> 'Tensor':
        """
        log10(input, out=None) -> Tensor

        Returns a new tensor with the logarithm to the base 10 of the elements
        of :attr:`input`.

        .. math::
            y_{i} = \log_{10} (x_{i})

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.rand(5)
            >>> a
            tensor([ 0.5224,  0.9354,  0.7257,  0.1301,  0.2251])


            >>> torch.log10(a)
            tensor([-0.2820, -0.0290, -0.1392, -0.8857, -0.6476])
        """
        return super().log10()

    @return_tensor_wrapper
    def log10_(self) -> 'Tensor':
        """
        log10_() -> Tensor

        In-place version of :meth:`~Tensor.log10`
        """
        return super().log10_()

    @return_tensor_wrapper
    def log1p(self) -> 'Tensor':
        """
        log1p(input, out=None) -> Tensor

        Returns a new tensor with the natural logarithm of (1 + :attr:`input`).

        .. math::
            y_i = \log_{e} (x_i + 1)

        .. note:: This function is more accurate than :func:`torch.log` for small
                  values of :attr:`input`

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(5)
            >>> a
            tensor([-1.0090, -0.9923,  1.0249, -0.5372,  0.2492])
            >>> torch.log1p(a)
            tensor([    nan, -4.8653,  0.7055, -0.7705,  0.2225])
        """
        return super().log1p()

    @return_tensor_wrapper
    def log1p_(self) -> 'Tensor':
        """
        log1p_() -> Tensor

        In-place version of :meth:`~Tensor.log1p`
        """
        return super().log1p_()

    @return_tensor_wrapper
    def log2(self) -> 'Tensor':
        """
        log2(input, out=None) -> Tensor

        Returns a new tensor with the logarithm to the base 2 of the elements
        of :attr:`input`.

        .. math::
            y_{i} = \log_{2} (x_{i})

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.rand(5)
            >>> a
            tensor([ 0.8419,  0.8003,  0.9971,  0.5287,  0.0490])


            >>> torch.log2(a)
            tensor([-0.2483, -0.3213, -0.0042, -0.9196, -4.3504])
        """
        return super().log2()

    @return_tensor_wrapper
    def log2_(self) -> 'Tensor':
        """
        log2_() -> Tensor

        In-place version of :meth:`~Tensor.log2`
        """
        return super().log2_()

    @return_tensor_wrapper
    def log_(self) -> 'Tensor':
        """
        log_() -> Tensor

        In-place version of :meth:`~Tensor.log`
        """
        return super().log_()

    @return_tensor_wrapper
    def logaddexp(self, other) -> 'Tensor':
        """
        logaddexp(input, other, out=None) -> Tensor

        Logarithm of the sum of exponentiations of the inputs.

        Calculates pointwise :math:`\log\left(e^x + e^y\right)`. This function is useful
        in statistics where the calculated probabilities of events may be so small as to
        exceed the range of normal floating point numbers. In such cases the logarithm
        of the calculated probability is stored. This function allows adding
        probabilities stored in such a fashion.

        This op should be disambiguated with :func:`torch.logsumexp` which performs a
        reduction on a single tensor.

        Args:
            input (Tensor): the input tensor.
            other (Tensor): the second input tensor

        Keyword arguments:
            out (Tensor, optional): the output tensor.

        Example::

            >>> torch.logaddexp(torch.tensor([-1.0]), torch.tensor([-1.0, -2, -3]))
            tensor([-0.3069, -0.6867, -0.8731])
            >>> torch.logaddexp(torch.tensor([-100.0, -200, -300]), torch.tensor([-1.0, -2, -3]))
            tensor([-1., -2., -3.])
            >>> torch.logaddexp(torch.tensor([1.0, 2000, 30000]), torch.tensor([-1.0, -2, -3]))
            tensor([1.1269e+00, 2.0000e+03, 3.0000e+04])
        """
        return super().logaddexp(other)

    @return_tensor_wrapper
    def logaddexp2(self, other) -> 'Tensor':
        """
        logaddexp2(input, other, out=None) -> Tensor

        Logarithm of the sum of exponentiations of the inputs in base-2.

        Calculates pointwise :math:`\log_2\left(2^x + 2^y\right)`. See
        :func:`torch.logaddexp` for more details.

        Args:
            input (Tensor): the input tensor.
            other (Tensor): the second input tensor

        Keyword arguments:
            out (Tensor, optional): the output tensor.
        """
        return super().logaddexp2(other)

    @return_tensor_wrapper
    def logcumsumexp(self, dim) -> 'Tensor':
        """
        logcumsumexp(input, dim, out=None) -> Tensor
        Returns the logarithm of the cumulative summation of the exponentiation of
        elements of :attr:`input` in the dimension :attr:`dim`.

        For summation index :math:`j` given by `dim` and other indices :math:`i`, the result is

            .. math::
                \text{logcumsumexp}(x)_{ij} = \log \sum\limits_{j=0}^{i} \exp(x_{ij})

        Args:
            input (Tensor): the input tensor.
            dim  (int): the dimension to do the operation over
            out (Tensor, optional): the output tensor.
        Example::
            >>> a = torch.randn(10)
            >>> torch.logcumsumexp(a, dim=0)
            tensor([-0.42296738, -0.04462666,  0.86278635,  0.94622083,  1.05277811,
                     1.39202815,  1.83525007,  1.84492621,  2.06084887,  2.06844475]))
        """
        return super().logcumsumexp(dim)

    @return_tensor_wrapper
    def logdet(self) -> 'Tensor':
        """
        logdet(input) -> Tensor

        Calculates log determinant of a square matrix or batches of square matrices.

        .. note::
            Result is ``-inf`` if :attr:`input` has zero log determinant, and is ``nan`` if
            :attr:`input` has negative determinant.

        .. note::
            Backward through :meth:`logdet` internally uses SVD results when :attr:`input`
            is not invertible. In this case, double backward through :meth:`logdet` will
            be unstable in when :attr:`input` doesn't have distinct singular values. See
            :meth:`~torch.svd` for details.

        Arguments:
            input (Tensor): the input tensor of size ``(*, n, n)`` where ``*`` is zero or more
                        batch dimensions.

        Example::

            >>> A = torch.randn(3, 3)
            >>> torch.det(A)
            tensor(0.2611)
            >>> torch.logdet(A)
            tensor(-1.3430)
            >>> A
            tensor([[[ 0.9254, -0.6213],
                     [-0.5787,  1.6843]],

                    [[ 0.3242, -0.9665],
                     [ 0.4539, -0.0887]],

                    [[ 1.1336, -0.4025],
                     [-0.7089,  0.9032]]])
            >>> A.det()
            tensor([1.1990, 0.4099, 0.7386])
            >>> A.det().log()
            tensor([ 0.1815, -0.8917, -0.3031])
        """
        return super().logdet()

    @return_tensor_wrapper
    def logical_and(self) -> 'Tensor':
        """
        logical_and(input, other, out=None) -> Tensor

        Computes the element-wise logical AND of the given input tensors. Zeros are treated as ``False`` and nonzeros are
        treated as ``True``.

        Args:
            input (Tensor): the input tensor.
            other (Tensor): the tensor to compute AND with
            out (Tensor, optional): the output tensor.

        Example::

            >>> torch.logical_and(torch.tensor([True, False, True]), torch.tensor([True, False, False]))
            tensor([ True, False, False])
            >>> a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
            >>> b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
            >>> torch.logical_and(a, b)
            tensor([False, False,  True, False])
            >>> torch.logical_and(a.double(), b.double())
            tensor([False, False,  True, False])
            >>> torch.logical_and(a.double(), b)
            tensor([False, False,  True, False])
            >>> torch.logical_and(a, b, out=torch.empty(4, dtype=torch.bool))
            tensor([False, False,  True, False])
        """
        return super().logical_and()

    @return_tensor_wrapper
    def logical_and_(self) -> 'Tensor':
        """
        logical_and_() -> Tensor

        In-place version of :meth:`~Tensor.logical_and`
        """
        return super().logical_and_()

    @return_tensor_wrapper
    def logical_not(self) -> 'Tensor':
        """
        logical_not(input, out=None) -> Tensor

        Computes the element-wise logical NOT of the given input tensor. If not specified, the output tensor will have the bool
        dtype. If the input tensor is not a bool tensor, zeros are treated as ``False`` and non-zeros are treated as ``True``.

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> torch.logical_not(torch.tensor([True, False]))
            tensor([False,  True])
            >>> torch.logical_not(torch.tensor([0, 1, -10], dtype=torch.int8))
            tensor([ True, False, False])
            >>> torch.logical_not(torch.tensor([0., 1.5, -10.], dtype=torch.double))
            tensor([ True, False, False])
            >>> torch.logical_not(torch.tensor([0., 1., -10.], dtype=torch.double), out=torch.empty(3, dtype=torch.int16))
            tensor([1, 0, 0], dtype=torch.int16)
        """
        return super().logical_not()

    @return_tensor_wrapper
    def logical_not_(self) -> 'Tensor':
        """
        logical_not_() -> Tensor

        In-place version of :meth:`~Tensor.logical_not`
        """
        return super().logical_not_()

    @return_tensor_wrapper
    def logical_or(self) -> 'Tensor':
        """
        logical_or(input, other, out=None) -> Tensor

        Computes the element-wise logical OR of the given input tensors. Zeros are treated as ``False`` and nonzeros are
        treated as ``True``.

        Args:
            input (Tensor): the input tensor.
            other (Tensor): the tensor to compute OR with
            out (Tensor, optional): the output tensor.

        Example::

            >>> torch.logical_or(torch.tensor([True, False, True]), torch.tensor([True, False, False]))
            tensor([ True, False,  True])
            >>> a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
            >>> b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
            >>> torch.logical_or(a, b)
            tensor([ True,  True,  True, False])
            >>> torch.logical_or(a.double(), b.double())
            tensor([ True,  True,  True, False])
            >>> torch.logical_or(a.double(), b)
            tensor([ True,  True,  True, False])
            >>> torch.logical_or(a, b, out=torch.empty(4, dtype=torch.bool))
            tensor([ True,  True,  True, False])
        """
        return super().logical_or()

    @return_tensor_wrapper
    def logical_or_(self) -> 'Tensor':
        """
        logical_or_() -> Tensor

        In-place version of :meth:`~Tensor.logical_or`
        """
        return super().logical_or_()

    @return_tensor_wrapper
    def logical_xor(self) -> 'Tensor':
        """
        logical_xor(input, other, out=None) -> Tensor

        Computes the element-wise logical XOR of the given input tensors. Zeros are treated as ``False`` and nonzeros are
        treated as ``True``.

        Args:
            input (Tensor): the input tensor.
            other (Tensor): the tensor to compute XOR with
            out (Tensor, optional): the output tensor.

        Example::

            >>> torch.logical_xor(torch.tensor([True, False, True]), torch.tensor([True, False, False]))
            tensor([False, False,  True])
            >>> a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
            >>> b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
            >>> torch.logical_xor(a, b)
            tensor([ True,  True, False, False])
            >>> torch.logical_xor(a.double(), b.double())
            tensor([ True,  True, False, False])
            >>> torch.logical_xor(a.double(), b)
            tensor([ True,  True, False, False])
            >>> torch.logical_xor(a, b, out=torch.empty(4, dtype=torch.bool))
            tensor([ True,  True, False, False])
        """
        return super().logical_xor()

    @return_tensor_wrapper
    def logical_xor_(self) -> 'Tensor':
        """
        logical_xor_() -> Tensor

        In-place version of :meth:`~Tensor.logical_xor`
        """
        return super().logical_xor_()

    @return_tensor_wrapper
    def logsumexp(self, dim, keepdim=False) -> 'Tensor':
        """
        logsumexp(input, dim, keepdim=False, out=None)

        Returns the log of summed exponentials of each row of the :attr:`input`
        tensor in the given dimension :attr:`dim`. The computation is numerically
        stabilized.

        For summation index :math:`j` given by `dim` and other indices :math:`i`, the result is

            .. math::
                \text{logsumexp}(x)_{i} = \log \sum_j \exp(x_{ij})


        If :attr:`keepdim` is ``True``, the output tensor is of the same size
        as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
        Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
        output tensor having 1 (or ``len(dim)``) fewer dimension(s).


        Args:
            input (Tensor): the input tensor.
            dim (int or tuple of ints): the dimension or dimensions to reduce.
            keepdim (bool): whether the output tensor has :attr:`dim` retained or not.
            out (Tensor, optional): the output tensor.


        Example::
            >>> a = torch.randn(3, 3)
            >>> torch.logsumexp(a, 1)
            tensor([ 0.8442,  1.4322,  0.8711])
        """
        return super().logsumexp(dim, keepdim=keepdim)

    @return_tensor_wrapper
    def long(self, memory_format=torch.preserve_format) -> 'Tensor':
        """
        long(memory_format=torch.preserve_format) -> Tensor

        ``self.long()`` is equivalent to ``self.to(torch.int64)``. See :func:`to`.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
        return super().long(memory_format=memory_format)

    @return_tensor_wrapper
    def lt(self, other) -> 'Tensor':
        """
        lt(input, other, out=None) -> Tensor

        Computes :math:`\text{input} < \text{other}` element-wise.

        The second argument can be a number or a tensor whose shape is
        :ref:`broadcastable <broadcasting-semantics>` with the first argument.

        Args:
            input (Tensor): the tensor to compare
            other (Tensor or float): the tensor or value to compare
            out (Tensor, optional): the output tensor that must be a `BoolTensor`

        Returns:
            Tensor: A `torch.BoolTensor` containing a True at each location where comparison is true

        Example::

            >>> torch.lt(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
            tensor([[False, False], [True, False]])
        """
        return super().lt(other)

    @return_tensor_wrapper
    def lt_(self, other) -> 'Tensor':
        """
        lt_(other) -> Tensor

        In-place version of :meth:`~Tensor.lt`
        """
        return super().lt_(other)

    @return_tensor_wrapper
    def lu_solve(self, LU_data, LU_pivots) -> 'Tensor':
        """
        lu_solve(input, LU_data, LU_pivots, out=None) -> Tensor

        Returns the LU solve of the linear system :math:`Ax = b` using the partially pivoted
        LU factorization of A from :meth:`torch.lu`.

        Arguments:
            b (Tensor): the RHS tensor of size :math:`(*, m, k)`, where :math:`*`
                        is zero or more batch dimensions.
            LU_data (Tensor): the pivoted LU factorization of A from :meth:`torch.lu` of size :math:`(*, m, m)`,
                               where :math:`*` is zero or more batch dimensions.
            LU_pivots (IntTensor): the pivots of the LU factorization from :meth:`torch.lu` of size :math:`(*, m)`,
                                   where :math:`*` is zero or more batch dimensions.
                                   The batch dimensions of :attr:`LU_pivots` must be equal to the batch dimensions of
                                   :attr:`LU_data`.
            out (Tensor, optional): the output tensor.

        Example::

            >>> A = torch.randn(2, 3, 3)
            >>> b = torch.randn(2, 3, 1)
            >>> A_LU = torch.lu(A)
            >>> x = torch.lu_solve(b, *A_LU)
            >>> torch.norm(torch.bmm(A, x) - b)
            tensor(1.00000e-07 *
                   2.8312)
        """
        return super().lu_solve(LU_data, LU_pivots)

    @return_tensor_wrapper
    def masked_fill(self, mask, value) -> 'Tensor':
        """
        masked_fill(mask, value) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.masked_fill_`
        """
        return super().masked_fill(mask, value)

    @return_tensor_wrapper
    def masked_scatter(self, mask, tensor) -> 'Tensor':
        """
        masked_scatter(mask, tensor) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.masked_scatter_`
        """
        return super().masked_scatter(mask, tensor)

    @return_tensor_wrapper
    def masked_select(self, mask) -> 'Tensor':
        """
        masked_select(input, mask, out=None) -> Tensor

        Returns a new 1-D tensor which indexes the :attr:`input` tensor according to
        the boolean mask :attr:`mask` which is a `BoolTensor`.

        The shapes of the :attr:`mask` tensor and the :attr:`input` tensor don't need
        to match, but they must be :ref:`broadcastable <broadcasting-semantics>`.

        .. note:: The returned tensor does **not** use the same storage
                  as the original tensor

        Args:
            input (Tensor): the input tensor.
            mask  (BoolTensor): the tensor containing the binary mask to index with
            out (Tensor, optional): the output tensor.

        Example::

            >>> x = torch.randn(3, 4)
            >>> x
            tensor([[ 0.3552, -2.3825, -0.8297,  0.3477],
                    [-1.2035,  1.2252,  0.5002,  0.6248],
                    [ 0.1307, -2.0608,  0.1244,  2.0139]])
            >>> mask = x.ge(0.5)
            >>> mask
            tensor([[False, False, False, False],
                    [False, True, True, True],
                    [False, False, False, True]])
            >>> torch.masked_select(x, mask)
            tensor([ 1.2252,  0.5002,  0.6248,  2.0139])
        """
        return super().masked_select(mask)

    @return_tensor_wrapper
    def matmul(self, tensor2) -> 'Tensor':
        """
        matmul(input, other, out=None) -> Tensor

        Matrix product of two tensors.

        The behavior depends on the dimensionality of the tensors as follows:

        - If both tensors are 1-dimensional, the dot product (scalar) is returned.
        - If both arguments are 2-dimensional, the matrix-matrix product is returned.
        - If the first argument is 1-dimensional and the second argument is 2-dimensional,
          a 1 is prepended to its dimension for the purpose of the matrix multiply.
          After the matrix multiply, the prepended dimension is removed.
        - If the first argument is 2-dimensional and the second argument is 1-dimensional,
          the matrix-vector product is returned.
        - If both arguments are at least 1-dimensional and at least one argument is
          N-dimensional (where N > 2), then a batched matrix multiply is returned.  If the first
          argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the
          batched matrix multiply and removed after.  If the second argument is 1-dimensional, a
          1 is appended to its dimension for the purpose of the batched matrix multiple and removed after.
          The non-matrix (i.e. batch) dimensions are :ref:`broadcasted <broadcasting-semantics>` (and thus
          must be broadcastable).  For example, if :attr:`input` is a
          :math:`(j \times 1 \times n \times m)` tensor and :attr:`other` is a :math:`(k \times m \times p)`
          tensor, :attr:`out` will be an :math:`(j \times k \times n \times p)` tensor.

        .. note::

            The 1-dimensional dot product version of this function does not support an :attr:`out` parameter.

        Arguments:
            input (Tensor): the first tensor to be multiplied
            other (Tensor): the second tensor to be multiplied
            out (Tensor, optional): the output tensor.

        Example::

            >>> # vector x vector
            >>> tensor1 = torch.randn(3)
            >>> tensor2 = torch.randn(3)
            >>> torch.matmul(tensor1, tensor2).size()
            torch.Size([])
            >>> # matrix x vector
            >>> tensor1 = torch.randn(3, 4)
            >>> tensor2 = torch.randn(4)
            >>> torch.matmul(tensor1, tensor2).size()
            torch.Size([3])
            >>> # batched matrix x broadcasted vector
            >>> tensor1 = torch.randn(10, 3, 4)
            >>> tensor2 = torch.randn(4)
            >>> torch.matmul(tensor1, tensor2).size()
            torch.Size([10, 3])
            >>> # batched matrix x batched matrix
            >>> tensor1 = torch.randn(10, 3, 4)
            >>> tensor2 = torch.randn(10, 4, 5)
            >>> torch.matmul(tensor1, tensor2).size()
            torch.Size([10, 3, 5])
            >>> # batched matrix x broadcasted matrix
            >>> tensor1 = torch.randn(10, 3, 4)
            >>> tensor2 = torch.randn(4, 5)
            >>> torch.matmul(tensor1, tensor2).size()
            torch.Size([10, 3, 5])
        """
        return super().matmul(tensor2)

    @return_tensor_wrapper
    def matrix_power(self, n) -> 'Tensor':
        """
        matrix_power(input, n) -> Tensor

        Returns the matrix raised to the power :attr:`n` for square matrices.
        For batch of matrices, each individual matrix is raised to the power :attr:`n`.

        If :attr:`n` is negative, then the inverse of the matrix (if invertible) is
        raised to the power :attr:`n`.  For a batch of matrices, the batched inverse
        (if invertible) is raised to the power :attr:`n`. If :attr:`n` is 0, then an identity matrix
        is returned.

        Args:
            input (Tensor): the input tensor.
            n (int): the power to raise the matrix to

        Example::

            >>> a = torch.randn(2, 2, 2)
            >>> a
            tensor([[[-1.9975, -1.9610],
                     [ 0.9592, -2.3364]],

                    [[-1.2534, -1.3429],
                     [ 0.4153, -1.4664]]])
            >>> torch.matrix_power(a, 3)
            tensor([[[  3.9392, -23.9916],
                     [ 11.7357,  -0.2070]],

                    [[  0.2468,  -6.7168],
                     [  2.0774,  -0.8187]]])
        """
        return super().matrix_power(n)

    @return_tensor_wrapper
    def mm(self, mat2) -> 'Tensor':
        """
        mm(input, mat2, out=None) -> Tensor

        Performs a matrix multiplication of the matrices :attr:`input` and :attr:`mat2`.

        If :attr:`input` is a :math:`(n \times m)` tensor, :attr:`mat2` is a
        :math:`(m \times p)` tensor, :attr:`out` will be a :math:`(n \times p)` tensor.

        .. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.
                  For broadcasting matrix products, see :func:`torch.matmul`.

        Args:
            input (Tensor): the first matrix to be multiplied
            mat2 (Tensor): the second matrix to be multiplied
            out (Tensor, optional): the output tensor.

        Example::

            >>> mat1 = torch.randn(2, 3)
            >>> mat2 = torch.randn(3, 3)
            >>> torch.mm(mat1, mat2)
            tensor([[ 0.4851,  0.5037, -0.3633],
                    [-0.0760, -3.6705,  2.4784]])
        """
        return super().mm(mat2)

    @return_tensor_wrapper
    def mul(self, value) -> 'Tensor':
        """
        mul(input, other, out=None)

        Multiplies each element of the input :attr:`input` with the scalar
        :attr:`other` and returns a new resulting tensor.

        .. math::
            \text{out}_i = \text{other} \times \text{input}_i

        If :attr:`input` is of type `FloatTensor` or `DoubleTensor`, :attr:`other`
        should be a real number, otherwise it should be an integer

        Args:
            {input}
            value (Number): the number to be multiplied to each element of :attr:`input`
            {out}

        Example::

            >>> a = torch.randn(3)
            >>> a
            tensor([ 0.2015, -0.4255,  2.6087])
            >>> torch.mul(a, 100)
            tensor([  20.1494,  -42.5491,  260.8663])

        .. function:: mul(input, other, out=None)

        Each element of the tensor :attr:`input` is multiplied by the corresponding
        element of the Tensor :attr:`other`. The resulting tensor is returned.

        The shapes of :attr:`input` and :attr:`other` must be
        :ref:`broadcastable <broadcasting-semantics>`.

        .. math::
            \text{out}_i = \text{input}_i \times \text{other}_i

        Args:
            input (Tensor): the first multiplicand tensor
            other (Tensor): the second multiplicand tensor
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4, 1)
            >>> a
            tensor([[ 1.1207],
                    [-0.3137],
                    [ 0.0700],
                    [ 0.8378]])
            >>> b = torch.randn(1, 4)
            >>> b
            tensor([[ 0.5146,  0.1216, -0.5244,  2.2382]])
            >>> torch.mul(a, b)
            tensor([[ 0.5767,  0.1363, -0.5877,  2.5083],
                    [-0.1614, -0.0382,  0.1645, -0.7021],
                    [ 0.0360,  0.0085, -0.0367,  0.1567],
                    [ 0.4312,  0.1019, -0.4394,  1.8753]])
        """
        return super().mul(value)

    @return_tensor_wrapper
    def multinomial(self, num_samples, replacement=False, *, generator=None) -> 'LongTensor':
        """
        multinomial(input, num_samples, replacement=False, *, generator=None, out=None) -> LongTensor

        Returns a tensor where each row contains :attr:`num_samples` indices sampled
        from the multinomial probability distribution located in the corresponding row
        of tensor :attr:`input`.

        .. note::
            The rows of :attr:`input` do not need to sum to one (in which case we use
            the values as weights), but must be non-negative, finite and have
            a non-zero sum.

        Indices are ordered from left to right according to when each was sampled
        (first samples are placed in first column).

        If :attr:`input` is a vector, :attr:`out` is a vector of size :attr:`num_samples`.

        If :attr:`input` is a matrix with `m` rows, :attr:`out` is an matrix of shape
        :math:`(m \times \text{num\_samples})`.

        If replacement is ``True``, samples are drawn with replacement.

        If not, they are drawn without replacement, which means that when a
        sample index is drawn for a row, it cannot be drawn again for that row.

        .. note::
            When drawn without replacement, :attr:`num_samples` must be lower than
            number of non-zero elements in :attr:`input` (or the min number of non-zero
            elements in each row of :attr:`input` if it is a matrix).

        Args:
            input (Tensor): the input tensor containing probabilities
            num_samples (int): number of samples to draw
            replacement (bool, optional): whether to draw with replacement or not
            generator (:class:`torch.Generator`, optional): a pseudorandom number generator for sampling
            out (Tensor, optional): the output tensor.

        Example::

            >>> weights = torch.tensor([0, 10, 3, 0], dtype=torch.float) # create a tensor of weights
            >>> torch.multinomial(weights, 2)
            tensor([1, 2])
            >>> torch.multinomial(weights, 4) # ERROR!
            RuntimeError: invalid argument 2: invalid multinomial distribution (with replacement=False,
            not enough non-negative category to sample) at ../aten/src/TH/generic/THTensorRandom.cpp:320
            >>> torch.multinomial(weights, 4, replacement=True)
            tensor([ 2,  1,  1,  1])
        """
        return super().multinomial(num_samples, replacement=replacement, generator=generator)

    @return_tensor_wrapper
    def mv(self, vec) -> 'Tensor':
        """
        mv(input, vec, out=None) -> Tensor

        Performs a matrix-vector product of the matrix :attr:`input` and the vector
        :attr:`vec`.

        If :attr:`input` is a :math:`(n \times m)` tensor, :attr:`vec` is a 1-D tensor of
        size :math:`m`, :attr:`out` will be 1-D of size :math:`n`.

        .. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.

        Args:
            input (Tensor): matrix to be multiplied
            vec (Tensor): vector to be multiplied
            out (Tensor, optional): the output tensor.

        Example::

            >>> mat = torch.randn(2, 3)
            >>> vec = torch.randn(3)
            >>> torch.mv(mat, vec)
            tensor([ 1.0404, -0.6361])
        """
        return super().mv(vec)

    @return_tensor_wrapper
    def mvlgamma(self, p) -> 'Tensor':
        """
        mvlgamma(input, p) -> Tensor

        Computes the `multivariate log-gamma function
        <https://en.wikipedia.org/wiki/Multivariate_gamma_function>`_) with dimension
        :math:`p` element-wise, given by

        .. math::
            \log(\Gamma_{p}(a)) = C + \displaystyle \sum_{i=1}^{p} \log\left(\Gamma\left(a - \frac{i - 1}{2}\right)\right)

        where :math:`C = \log(\pi) \times \frac{p (p - 1)}{4}` and :math:`\Gamma(\cdot)` is the Gamma function.

        All elements must be greater than :math:`\frac{p - 1}{2}`, otherwise an error would be thrown.

        Args:
            input (Tensor): the tensor to compute the multivariate log-gamma function
            p (int): the number of dimensions

        Example::

            >>> a = torch.empty(2, 3).uniform_(1, 2)
            >>> a
            tensor([[1.6835, 1.8474, 1.1929],
                    [1.0475, 1.7162, 1.4180]])
            >>> torch.mvlgamma(a, 2)
            tensor([[0.3928, 0.4007, 0.7586],
                    [1.0311, 0.3901, 0.5049]])
        """
        return super().mvlgamma(p)

    @return_tensor_wrapper
    def mvlgamma_(self, p) -> 'Tensor':
        """
        mvlgamma_(p) -> Tensor

        In-place version of :meth:`~Tensor.mvlgamma`
        """
        return super().mvlgamma_(p)

    @return_tensor_wrapper
    def narrow(self, dimension, start, length) -> 'Tensor':
        """
        narrow(dimension, start, length) -> Tensor

        See :func:`torch.narrow`

        Example::

            >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            >>> x.narrow(0, 0, 2)
            tensor([[ 1,  2,  3],
                    [ 4,  5,  6]])
            >>> x.narrow(1, 1, 2)
            tensor([[ 2,  3],
                    [ 5,  6],
                    [ 8,  9]])
        """
        return super().narrow(dimension, start, length)

    @return_tensor_wrapper
    def narrow_copy(self, dimension, start, length) -> 'Tensor':
        """
        narrow_copy(dimension, start, length) -> Tensor

        Same as :meth:`Tensor.narrow` except returning a copy rather
        than shared storage.  This is primarily for sparse tensors, which
        do not have a shared-storage narrow method.  Calling ```narrow_copy``
        with ```dimemsion > self.sparse_dim()``` will return a copy with the
        relevant dense dimension narrowed, and ```self.shape``` updated accordingly.
        """
        return super().narrow_copy(dimension, start, length)

    @return_tensor_wrapper
    def ne(self, other) -> 'Tensor':
        """
        ne(input, other, out=None) -> Tensor

        Computes :math:`input \neq other` element-wise.

        The second argument can be a number or a tensor whose shape is
        :ref:`broadcastable <broadcasting-semantics>` with the first argument.

        Args:
            input (Tensor): the tensor to compare
            other (Tensor or float): the tensor or value to compare
            out (Tensor, optional): the output tensor that must be a `BoolTensor`

        Returns:
            Tensor: A ``torch.BoolTensor`` containing a True at each location where comparison is true.

        Example::

            >>> torch.ne(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
            tensor([[False, True], [True, False]])
        """
        return super().ne(other)

    @return_tensor_wrapper
    def ne_(self, other) -> 'Tensor':
        """
        ne_(other) -> Tensor

        In-place version of :meth:`~Tensor.ne`
        """
        return super().ne_(other)

    @return_tensor_wrapper
    def neg(self) -> 'Tensor':
        """
        neg(input, out=None) -> Tensor

        Returns a new tensor with the negative of the elements of :attr:`input`.

        .. math::
            \text{out} = -1 \times \text{input}

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(5)
            >>> a
            tensor([ 0.0090, -0.2262, -0.0682, -0.2866,  0.3940])
            >>> torch.neg(a)
            tensor([-0.0090,  0.2262,  0.0682,  0.2866, -0.3940])
        """
        return super().neg()

    @return_tensor_wrapper
    def neg_(self) -> 'Tensor':
        """
        neg_() -> Tensor

        In-place version of :meth:`~Tensor.neg`
        """
        return super().neg_()

    @return_tensor_wrapper
    def new_empty(self, size, dtype=None, device=None, requires_grad=False) -> 'Tensor':
        """
        new_empty(size, dtype=None, device=None, requires_grad=False) -> Tensor

        Returns a Tensor of size :attr:`size` filled with uninitialized data.
        By default, the returned Tensor has the same :class:`torch.dtype` and
        :class:`torch.device` as this tensor.

        Args:
            dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
                Default: if None, same :class:`torch.dtype` as this tensor.
            device (:class:`torch.device`, optional): the desired device of returned tensor.
                Default: if None, same :class:`torch.device` as this tensor.
            requires_grad (bool, optional): If autograd should record operations on the
                returned tensor. Default: ``False``.

        Example::

            >>> tensor = torch.ones(())
            >>> tensor.new_empty((2, 3))
            tensor([[ 5.8182e-18,  4.5765e-41, -1.0545e+30],
                    [ 3.0949e-41,  4.4842e-44,  0.0000e+00]])
        """
        return super().new_empty(size, dtype=dtype, device=device, requires_grad=requires_grad)

    @return_tensor_wrapper
    def new_full(self, size, fill_value, dtype=None, device=None, requires_grad=False) -> 'Tensor':
        """
        new_full(size, fill_value, dtype=None, device=None, requires_grad=False) -> Tensor

        Returns a Tensor of size :attr:`size` filled with :attr:`fill_value`.
        By default, the returned Tensor has the same :class:`torch.dtype` and
        :class:`torch.device` as this tensor.

        Args:
            fill_value (scalar): the number to fill the output tensor with.
            dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
                Default: if None, same :class:`torch.dtype` as this tensor.
            device (:class:`torch.device`, optional): the desired device of returned tensor.
                Default: if None, same :class:`torch.device` as this tensor.
            requires_grad (bool, optional): If autograd should record operations on the
                returned tensor. Default: ``False``.

        Example::

            >>> tensor = torch.ones((2,), dtype=torch.float64)
            >>> tensor.new_full((3, 4), 3.141592)
            tensor([[ 3.1416,  3.1416,  3.1416,  3.1416],
                    [ 3.1416,  3.1416,  3.1416,  3.1416],
                    [ 3.1416,  3.1416,  3.1416,  3.1416]], dtype=torch.float64)
        """
        return super().new_full(size, fill_value, dtype=dtype, device=device, requires_grad=requires_grad)

    @return_tensor_wrapper
    def new_ones(self, size, dtype=None, device=None, requires_grad=False) -> 'Tensor':
        """
        new_ones(size, dtype=None, device=None, requires_grad=False) -> Tensor

        Returns a Tensor of size :attr:`size` filled with ``1``.
        By default, the returned Tensor has the same :class:`torch.dtype` and
        :class:`torch.device` as this tensor.

        Args:
            size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
                shape of the output tensor.
            dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
                Default: if None, same :class:`torch.dtype` as this tensor.
            device (:class:`torch.device`, optional): the desired device of returned tensor.
                Default: if None, same :class:`torch.device` as this tensor.
            requires_grad (bool, optional): If autograd should record operations on the
                returned tensor. Default: ``False``.

        Example::

            >>> tensor = torch.tensor((), dtype=torch.int32)
            >>> tensor.new_ones((2, 3))
            tensor([[ 1,  1,  1],
                    [ 1,  1,  1]], dtype=torch.int32)
        """
        return super().new_ones(size, dtype=dtype, device=device, requires_grad=requires_grad)

    @return_tensor_wrapper
    def new_tensor(self, data, dtype=None, device=None, requires_grad=False) -> 'Tensor':
        """
        new_tensor(data, dtype=None, device=None, requires_grad=False) -> Tensor

        Returns a new Tensor with :attr:`data` as the tensor data.
        By default, the returned Tensor has the same :class:`torch.dtype` and
        :class:`torch.device` as this tensor.

        .. warning::

            :func:`new_tensor` always copies :attr:`data`. If you have a Tensor
            ``data`` and want to avoid a copy, use :func:`torch.Tensor.requires_grad_`
            or :func:`torch.Tensor.detach`.
            If you have a numpy array and want to avoid a copy, use
            :func:`torch.from_numpy`.

        .. warning::

            When data is a tensor `x`, :func:`new_tensor()` reads out 'the data' from whatever it is passed,
            and constructs a leaf variable. Therefore ``tensor.new_tensor(x)`` is equivalent to ``x.clone().detach()``
            and ``tensor.new_tensor(x, requires_grad=True)`` is equivalent to ``x.clone().detach().requires_grad_(True)``.
            The equivalents using ``clone()`` and ``detach()`` are recommended.

        Args:
            data (array_like): The returned Tensor copies :attr:`data`.
            dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
                Default: if None, same :class:`torch.dtype` as this tensor.
            device (:class:`torch.device`, optional): the desired device of returned tensor.
                Default: if None, same :class:`torch.device` as this tensor.
            requires_grad (bool, optional): If autograd should record operations on the
                returned tensor. Default: ``False``.

        Example::

            >>> tensor = torch.ones((2,), dtype=torch.int8)
            >>> data = [[0, 1], [2, 3]]
            >>> tensor.new_tensor(data)
            tensor([[ 0,  1],
                    [ 2,  3]], dtype=torch.int8)
        """
        return super().new_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

    @return_tensor_wrapper
    def new_zeros(self, size, dtype=None, device=None, requires_grad=False) -> 'Tensor':
        """
        new_zeros(size, dtype=None, device=None, requires_grad=False) -> Tensor

        Returns a Tensor of size :attr:`size` filled with ``0``.
        By default, the returned Tensor has the same :class:`torch.dtype` and
        :class:`torch.device` as this tensor.

        Args:
            size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
                shape of the output tensor.
            dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
                Default: if None, same :class:`torch.dtype` as this tensor.
            device (:class:`torch.device`, optional): the desired device of returned tensor.
                Default: if None, same :class:`torch.device` as this tensor.
            requires_grad (bool, optional): If autograd should record operations on the
                returned tensor. Default: ``False``.

        Example::

            >>> tensor = torch.tensor((), dtype=torch.float64)
            >>> tensor.new_zeros((2, 3))
            tensor([[ 0.,  0.,  0.],
                    [ 0.,  0.,  0.]], dtype=torch.float64)
        """
        return super().new_zeros(size, dtype=dtype, device=device, requires_grad=requires_grad)

    @return_tensor_wrapper
    def nonzero(self) -> 'LongTensor':
        """
        nonzero(input, *, out=None, as_tuple=False) -> LongTensor or tuple of LongTensors

        .. note::
            :func:`torch.nonzero(..., as_tuple=False) <torch.nonzero>` (default) returns a
            2-D tensor where each row is the index for a nonzero value.

            :func:`torch.nonzero(..., as_tuple=True) <torch.nonzero>` returns a tuple of 1-D
            index tensors, allowing for advanced indexing, so ``x[x.nonzero(as_tuple=True)]``
            gives all nonzero values of tensor ``x``. Of the returned tuple, each index tensor
            contains nonzero indices for a certain dimension.

            See below for more details on the two behaviors.


        **When** :attr:`as_tuple` **is ``False`` (default)**:

        Returns a tensor containing the indices of all non-zero elements of
        :attr:`input`.  Each row in the result contains the indices of a non-zero
        element in :attr:`input`. The result is sorted lexicographically, with
        the last index changing the fastest (C-style).

        If :attr:`input` has :math:`n` dimensions, then the resulting indices tensor
        :attr:`out` is of size :math:`(z \times n)`, where :math:`z` is the total number of
        non-zero elements in the :attr:`input` tensor.

        **When** :attr:`as_tuple` **is ``True``**:

        Returns a tuple of 1-D tensors, one for each dimension in :attr:`input`,
        each containing the indices (in that dimension) of all non-zero elements of
        :attr:`input` .

        If :attr:`input` has :math:`n` dimensions, then the resulting tuple contains :math:`n`
        tensors of size :math:`z`, where :math:`z` is the total number of
        non-zero elements in the :attr:`input` tensor.

        As a special case, when :attr:`input` has zero dimensions and a nonzero scalar
        value, it is treated as a one-dimensional tensor with one element.

        Args:
            input (Tensor): the input tensor.
            out (LongTensor, optional): the output tensor containing indices

        Returns:
            LongTensor or tuple of LongTensor: If :attr:`as_tuple` is ``False``, the output
            tensor containing indices. If :attr:`as_tuple` is ``True``, one 1-D tensor for
            each dimension, containing the indices of each nonzero element along that
            dimension.

        Example::

            >>> torch.nonzero(torch.tensor([1, 1, 1, 0, 1]))
            tensor([[ 0],
                    [ 1],
                    [ 2],
                    [ 4]])
            >>> torch.nonzero(torch.tensor([[0.6, 0.0, 0.0, 0.0],
                                            [0.0, 0.4, 0.0, 0.0],
                                            [0.0, 0.0, 1.2, 0.0],
                                            [0.0, 0.0, 0.0,-0.4]]))
            tensor([[ 0,  0],
                    [ 1,  1],
                    [ 2,  2],
                    [ 3,  3]])
            >>> torch.nonzero(torch.tensor([1, 1, 1, 0, 1]), as_tuple=True)
            (tensor([0, 1, 2, 4]),)
            >>> torch.nonzero(torch.tensor([[0.6, 0.0, 0.0, 0.0],
                                            [0.0, 0.4, 0.0, 0.0],
                                            [0.0, 0.0, 1.2, 0.0],
                                            [0.0, 0.0, 0.0,-0.4]]), as_tuple=True)
            (tensor([0, 1, 2, 3]), tensor([0, 1, 2, 3]))
            >>> torch.nonzero(torch.tensor(5), as_tuple=True)
            (tensor([0]),)
        """
        return super().nonzero()

    @return_tensor_wrapper
    def normal_(self, mean=0, std=1, *, generator=None) -> 'Tensor':
        """
        normal_(mean=0, std=1, *, generator=None) -> Tensor

        Fills :attr:`self` tensor with elements samples from the normal distribution
        parameterized by :attr:`mean` and :attr:`std`.
        """
        return super().normal_(mean=mean, std=std, generator=generator)

    @return_tensor_wrapper
    def orgqr(self, input2) -> 'Tensor':
        """
        orgqr(input, input2) -> Tensor

        Computes the orthogonal matrix `Q` of a QR factorization, from the `(input, input2)`
        tuple returned by :func:`torch.geqrf`.

        This directly calls the underlying LAPACK function `?orgqr`.
        See `LAPACK documentation for orgqr`_ for further details.

        Args:
            input (Tensor): the `a` from :func:`torch.geqrf`.
            input2 (Tensor): the `tau` from :func:`torch.geqrf`.

        .. _LAPACK documentation for orgqr:
            https://software.intel.com/en-us/mkl-developer-reference-c-orgqr
        """
        return super().orgqr(input2)

    @return_tensor_wrapper
    def ormqr(self, input2, input3, left=True, transpose=False) -> 'Tensor':
        """
        ormqr(input, input2, input3, left=True, transpose=False) -> Tensor

        Multiplies `mat` (given by :attr:`input3`) by the orthogonal `Q` matrix of the QR factorization
        formed by :func:`torch.geqrf` that is represented by `(a, tau)` (given by (:attr:`input`, :attr:`input2`)).

        This directly calls the underlying LAPACK function `?ormqr`.
        See `LAPACK documentation for ormqr`_ for further details.

        Args:
            input (Tensor): the `a` from :func:`torch.geqrf`.
            input2 (Tensor): the `tau` from :func:`torch.geqrf`.
            input3 (Tensor): the matrix to be multiplied.

        .. _LAPACK documentation for ormqr:
            https://software.intel.com/en-us/mkl-developer-reference-c-ormqr
        """
        return super().ormqr(input2, input3, left=left, transpose=transpose)

    @return_tensor_wrapper
    def permute(self, *dims) -> 'Tensor':
        """
        permute(*dims) -> Tensor

        Returns a view of the original tensor with its dimensions permuted.

        Args:
            *dims (int...): The desired ordering of dimensions

        Example:
            >>> x = torch.randn(2, 3, 5)
            >>> x.size()
            torch.Size([2, 3, 5])
            >>> x.permute(2, 0, 1).size()
            torch.Size([5, 2, 3])
        """
        return super().permute(*dims)

    @return_tensor_wrapper
    def pin_memory(self) -> 'Tensor':
        """
        pin_memory() -> Tensor

        Copies the tensor to pinned memory, if it's not already pinned.
        """
        return super().pin_memory()

    @return_tensor_wrapper
    def pinverse(self) -> 'Tensor':
        """
        pinverse(input, rcond=1e-15) -> Tensor

        Calculates the pseudo-inverse (also known as the Moore-Penrose inverse) of a 2D tensor.
        Please look at `Moore-Penrose inverse`_ for more details

        .. note::
            This method is implemented using the Singular Value Decomposition.

        .. note::
            The pseudo-inverse is not necessarily a continuous function in the elements of the matrix `[1]`_.
            Therefore, derivatives are not always existent, and exist for a constant rank only `[2]`_.
            However, this method is backprop-able due to the implementation by using SVD results, and
            could be unstable. Double-backward will also be unstable due to the usage of SVD internally.
            See :meth:`~torch.svd` for more details.

        Arguments:
            input (Tensor): The input tensor of size :math:`(*, m, n)` where :math:`*` is zero or more batch dimensions
            rcond (float): A floating point value to determine the cutoff for small singular values.
                           Default: 1e-15

        Returns:
            The pseudo-inverse of :attr:`input` of dimensions :math:`(*, n, m)`

        Example::

            >>> input = torch.randn(3, 5)
            >>> input
            tensor([[ 0.5495,  0.0979, -1.4092, -0.1128,  0.4132],
                    [-1.1143, -0.3662,  0.3042,  1.6374, -0.9294],
                    [-0.3269, -0.5745, -0.0382, -0.5922, -0.6759]])
            >>> torch.pinverse(input)
            tensor([[ 0.0600, -0.1933, -0.2090],
                    [-0.0903, -0.0817, -0.4752],
                    [-0.7124, -0.1631, -0.2272],
                    [ 0.1356,  0.3933, -0.5023],
                    [-0.0308, -0.1725, -0.5216]])
            >>> # Batched pinverse example
            >>> a = torch.randn(2,6,3)
            >>> b = torch.pinverse(a)
            >>> torch.matmul(b, a)
            tensor([[[ 1.0000e+00,  1.6391e-07, -1.1548e-07],
                    [ 8.3121e-08,  1.0000e+00, -2.7567e-07],
                    [ 3.5390e-08,  1.4901e-08,  1.0000e+00]],

                    [[ 1.0000e+00, -8.9407e-08,  2.9802e-08],
                    [-2.2352e-07,  1.0000e+00,  1.1921e-07],
                    [ 0.0000e+00,  8.9407e-08,  1.0000e+00]]])

        .. _Moore-Penrose inverse: https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse

        .. _[1]: https://epubs.siam.org/doi/10.1137/0117004

        .. _[2]: https://www.jstor.org/stable/2156365
        """
        return super().pinverse()

    @return_tensor_wrapper
    def polygamma(self, n) -> 'Tensor':
        """
        polygamma(n, input, out=None) -> Tensor

        Computes the :math:`n^{th}` derivative of the digamma function on :attr:`input`.
        :math:`n \geq 0` is called the order of the polygamma function.

        .. math::
            \psi^{(n)}(x) = \frac{d^{(n)}}{dx^{(n)}} \psi(x)

        .. note::
            This function is not implemented for :math:`n \geq 2`.

        Args:
            n (int): the order of the polygamma function
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::
            >>> a = torch.tensor([1, 0.5])
            >>> torch.polygamma(1, a)
            tensor([1.64493, 4.9348])
        """
        return super().polygamma(n)

    @return_tensor_wrapper
    def polygamma_(self, n) -> 'Tensor':
        """
        polygamma_(n) -> Tensor

        In-place version of :meth:`~Tensor.polygamma`
        """
        return super().polygamma_(n)

    @return_tensor_wrapper
    def pow(self, exponent) -> 'Tensor':
        """
        pow(input, exponent, out=None) -> Tensor

        Takes the power of each element in :attr:`input` with :attr:`exponent` and
        returns a tensor with the result.

        :attr:`exponent` can be either a single ``float`` number or a `Tensor`
        with the same number of elements as :attr:`input`.

        When :attr:`exponent` is a scalar value, the operation applied is:

        .. math::
            \text{out}_i = x_i ^ \text{exponent}

        When :attr:`exponent` is a tensor, the operation applied is:

        .. math::
            \text{out}_i = x_i ^ {\text{exponent}_i}

        When :attr:`exponent` is a tensor, the shapes of :attr:`input`
        and :attr:`exponent` must be :ref:`broadcastable <broadcasting-semantics>`.

        Args:
            input (Tensor): the input tensor.
            exponent (float or tensor): the exponent value
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4)
            >>> a
            tensor([ 0.4331,  1.2475,  0.6834, -0.2791])
            >>> torch.pow(a, 2)
            tensor([ 0.1875,  1.5561,  0.4670,  0.0779])
            >>> exp = torch.arange(1., 5.)

            >>> a = torch.arange(1., 5.)
            >>> a
            tensor([ 1.,  2.,  3.,  4.])
            >>> exp
            tensor([ 1.,  2.,  3.,  4.])
            >>> torch.pow(a, exp)
            tensor([   1.,    4.,   27.,  256.])

        .. function:: pow(self, exponent, out=None) -> Tensor

        :attr:`self` is a scalar ``float`` value, and :attr:`exponent` is a tensor.
        The returned tensor :attr:`out` is of the same shape as :attr:`exponent`

        The operation applied is:

        .. math::
            \text{out}_i = \text{self} ^ {\text{exponent}_i}

        Args:
            self (float): the scalar base value for the power operation
            exponent (Tensor): the exponent tensor
            out (Tensor, optional): the output tensor.

        Example::

            >>> exp = torch.arange(1., 5.)
            >>> base = 2
            >>> torch.pow(base, exp)
            tensor([  2.,   4.,   8.,  16.])
        """
        return super().pow(exponent)

    @return_tensor_wrapper
    def pow_(self, exponent) -> 'Tensor':
        """
        pow_(exponent) -> Tensor

        In-place version of :meth:`~Tensor.pow`
        """
        return super().pow_(exponent)

    @return_tensor_wrapper
    def prod(self, dim=None, keepdim=False, dtype=None) -> 'Tensor':
        """
        prod(input, dtype=None) -> Tensor

        Returns the product of all elements in the :attr:`input` tensor.

        Args:
            input (Tensor): the input tensor.
            dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
                If specified, the input tensor is casted to :attr:`dtype` before the operation
                is performed. This is useful for preventing data type overflows. Default: None.

        Example::

            >>> a = torch.randn(1, 3)
            >>> a
            tensor([[-0.8020,  0.5428, -1.5854]])
            >>> torch.prod(a)
            tensor(0.6902)

        .. function:: prod(input, dim, keepdim=False, dtype=None) -> Tensor

        Returns the product of each row of the :attr:`input` tensor in the given
        dimension :attr:`dim`.

        If :attr:`keepdim` is ``True``, the output tensor is of the same size
        as :attr:`input` except in the dimension :attr:`dim` where it is of size 1.
        Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in
        the output tensor having 1 fewer dimension than :attr:`input`.

        Args:
            input (Tensor): the input tensor.
            dim (int): the dimension to reduce.
            keepdim (bool): whether the output tensor has :attr:`dim` retained or not.
            dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
                If specified, the input tensor is casted to :attr:`dtype` before the operation
                is performed. This is useful for preventing data type overflows. Default: None.

        Example::

            >>> a = torch.randn(4, 2)
            >>> a
            tensor([[ 0.5261, -0.3837],
                    [ 1.1857, -0.2498],
                    [-1.1646,  0.0705],
                    [ 1.1131, -1.0629]])
            >>> torch.prod(a, 1)
            tensor([-0.2018, -0.2962, -0.0821, -1.1831])
        """
        return super().prod(dim=dim, keepdim=keepdim, dtype=dtype)

    @return_tensor_wrapper
    def put_(self, indices, tensor, accumulate=False) -> 'Tensor':
        """
        put_(indices, tensor, accumulate=False) -> Tensor

        Copies the elements from :attr:`tensor` into the positions specified by
        indices. For the purpose of indexing, the :attr:`self` tensor is treated as if
        it were a 1-D tensor.

        If :attr:`accumulate` is ``True``, the elements in :attr:`tensor` are added to
        :attr:`self`. If accumulate is ``False``, the behavior is undefined if indices
        contain duplicate elements.

        Args:
            indices (LongTensor): the indices into self
            tensor (Tensor): the tensor containing values to copy from
            accumulate (bool): whether to accumulate into self

        Example::

            >>> src = torch.tensor([[4, 3, 5],
                                    [6, 7, 8]])
            >>> src.put_(torch.tensor([1, 3]), torch.tensor([9, 10]))
            tensor([[  4,   9,   5],
                    [ 10,   7,   8]])
        """
        return super().put_(indices, tensor, accumulate=accumulate)

    @return_tensor_wrapper
    def q_per_channel_scales(self) -> 'Tensor':
        """
        q_per_channel_scales() -> Tensor

        Given a Tensor quantized by linear (affine) per-channel quantization,
        returns a Tensor of scales of the underlying quantizer. It has the number of
        elements that matches the corresponding dimensions (from q_per_channel_axis) of
        the tensor.
        """
        return super().q_per_channel_scales()

    @return_tensor_wrapper
    def q_per_channel_zero_points(self) -> 'Tensor':
        """
        q_per_channel_zero_points() -> Tensor

        Given a Tensor quantized by linear (affine) per-channel quantization,
        returns a tensor of zero_points of the underlying quantizer. It has the number of
        elements that matches the corresponding dimensions (from q_per_channel_axis) of
        the tensor.
        """
        return super().q_per_channel_zero_points()

    @return_tensor_wrapper
    def rad2deg(self) -> 'Tensor':
        """
        rad2deg(input, out=None) -> Tensor

        Returns a new tensor with each of the elements of :attr:`input`
        converted from angles in radians to degrees.

        Args:
            input (Tensor): the input tensor.

        Keyword arguments:
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.tensor([[3.142, -3.142], [6.283, -6.283], [1.570, -1.570]])
            >>> torch.rad2deg(a)
            tensor([[ 180.0233, -180.0233],
                    [ 359.9894, -359.9894],
                    [  89.9544,  -89.9544]])
        """
        return super().rad2deg()

    @return_tensor_wrapper
    def rad2deg_(self) -> 'Tensor':
        """
        rad2deg_() -> Tensor

        In-place version of :meth:`~Tensor.rad2deg`
        """
        return super().rad2deg_()

    @return_tensor_wrapper
    def random_(self, start=0, to=None, *, generator=None) -> 'Tensor':
        """
        random_(from=0, to=None, *, generator=None) -> Tensor

        Fills :attr:`self` tensor with numbers sampled from the discrete uniform
        distribution over ``[from, to - 1]``. If not specified, the values are usually
        only bounded by :attr:`self` tensor's data type. However, for floating point
        types, if unspecified, range will be ``[0, 2^mantissa]`` to ensure that every
        value is representable. For example, `torch.tensor(1, dtype=torch.double).random_()`
        will be uniform in ``[0, 2^53]``.
        """
        return super().random_(start, to=to, generator=generator)

    @return_tensor_wrapper
    def reciprocal(self) -> 'Tensor':
        """
        reciprocal(input, out=None) -> Tensor

        Returns a new tensor with the reciprocal of the elements of :attr:`input`

        .. math::
            \text{out}_{i} = \frac{1}{\text{input}_{i}}

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4)
            >>> a
            tensor([-0.4595, -2.1219, -1.4314,  0.7298])
            >>> torch.reciprocal(a)
            tensor([-2.1763, -0.4713, -0.6986,  1.3702])
        """
        return super().reciprocal()

    @return_tensor_wrapper
    def reciprocal_(self) -> 'Tensor':
        """
        reciprocal_() -> Tensor

        In-place version of :meth:`~Tensor.reciprocal`
        """
        return super().reciprocal_()

    @return_tensor_wrapper
    def remainder(self, divisor) -> 'Tensor':
        """
        remainder(input, other, out=None) -> Tensor

        Computes the element-wise remainder of division.

        The dividend and divisor may contain both for integer and floating point
        numbers. The remainder has the same sign as the divisor :attr:`other`.

        When :attr:`other` is a tensor, the shapes of :attr:`input` and
        :attr:`other` must be :ref:`broadcastable <broadcasting-semantics>`.

        Args:
            input (Tensor): the dividend
            other (Tensor or float): the divisor that may be either a number or a
                                       Tensor of the same shape as the dividend
            out (Tensor, optional): the output tensor.

        Example::

            >>> torch.remainder(torch.tensor([-3., -2, -1, 1, 2, 3]), 2)
            tensor([ 1.,  0.,  1.,  1.,  0.,  1.])
            >>> torch.remainder(torch.tensor([1., 2, 3, 4, 5]), 1.5)
            tensor([ 1.0000,  0.5000,  0.0000,  1.0000,  0.5000])

        .. seealso::

                :func:`torch.fmod`, which computes the element-wise remainder of
                division equivalently to the C library function ``fmod()``.
        """
        return super().remainder(divisor)

    @return_tensor_wrapper
    def remainder_(self, divisor) -> 'Tensor':
        """
        remainder_(divisor) -> Tensor

        In-place version of :meth:`~Tensor.remainder`
        """
        return super().remainder_(divisor)

    @return_tensor_wrapper
    def renorm(self, p, dim, maxnorm) -> 'Tensor':
        """
        renorm(input, p, dim, maxnorm, out=None) -> Tensor

        Returns a tensor where each sub-tensor of :attr:`input` along dimension
        :attr:`dim` is normalized such that the `p`-norm of the sub-tensor is lower
        than the value :attr:`maxnorm`

        .. note:: If the norm of a row is lower than `maxnorm`, the row is unchanged

        Args:
            input (Tensor): the input tensor.
            p (float): the power for the norm computation
            dim (int): the dimension to slice over to get the sub-tensors
            maxnorm (float): the maximum norm to keep each sub-tensor under
            out (Tensor, optional): the output tensor.

        Example::

            >>> x = torch.ones(3, 3)
            >>> x[1].fill_(2)
            tensor([ 2.,  2.,  2.])
            >>> x[2].fill_(3)
            tensor([ 3.,  3.,  3.])
            >>> x
            tensor([[ 1.,  1.,  1.],
                    [ 2.,  2.,  2.],
                    [ 3.,  3.,  3.]])
            >>> torch.renorm(x, 1, 0, 5)
            tensor([[ 1.0000,  1.0000,  1.0000],
                    [ 1.6667,  1.6667,  1.6667],
                    [ 1.6667,  1.6667,  1.6667]])
        """
        return super().renorm(p, dim, maxnorm)

    @return_tensor_wrapper
    def renorm_(self, p, dim, maxnorm) -> 'Tensor':
        """
        renorm_(p, dim, maxnorm) -> Tensor

        In-place version of :meth:`~Tensor.renorm`
        """
        return super().renorm_(p, dim, maxnorm)

    @return_tensor_wrapper
    def repeat(self, *sizes) -> 'Tensor':
        """
        repeat(*sizes) -> Tensor

        Repeats this tensor along the specified dimensions.

        Unlike :meth:`~Tensor.expand`, this function copies the tensor's data.

        .. warning::

            :meth:`~Tensor.repeat` behaves differently from
            `numpy.repeat <https://docs.scipy.org/doc/numpy/reference/generated/numpy.repeat.html>`_,
            but is more similar to
            `numpy.tile <https://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html>`_.
            For the operator similar to `numpy.repeat`, see :func:`torch.repeat_interleave`.

        Args:
            sizes (torch.Size or int...): The number of times to repeat this tensor along each
                dimension

        Example::

            >>> x = torch.tensor([1, 2, 3])
            >>> x.repeat(4, 2)
            tensor([[ 1,  2,  3,  1,  2,  3],
                    [ 1,  2,  3,  1,  2,  3],
                    [ 1,  2,  3,  1,  2,  3],
                    [ 1,  2,  3,  1,  2,  3]])
            >>> x.repeat(4, 2, 1).size()
            torch.Size([4, 2, 3])
        """
        return super().repeat(*sizes)

    @return_tensor_wrapper
    def repeat_interleave(self, repeats, dim=None) -> 'Tensor':
        """
        repeat_interleave(input, repeats, dim=None) -> Tensor

        Repeat elements of a tensor.

        .. warning::

            This is different from :meth:`torch.Tensor.repeat` but similar to ``numpy.repeat``.

        Args:
            input (Tensor): the input tensor.
            repeats (Tensor or int): The number of repetitions for each element.
                repeats is broadcasted to fit the shape of the given axis.
            dim (int, optional): The dimension along which to repeat values.
                By default, use the flattened input array, and return a flat output
                array.

        Returns:
            Tensor: Repeated tensor which has the same shape as input, except along the
             given axis.

        Example::

            >>> x = torch.tensor([1, 2, 3])
            >>> x.repeat_interleave(2)
            tensor([1, 1, 2, 2, 3, 3])
            >>> y = torch.tensor([[1, 2], [3, 4]])
            >>> torch.repeat_interleave(y, 2)
            tensor([1, 1, 2, 2, 3, 3, 4, 4])
            >>> torch.repeat_interleave(y, 3, dim=1)
            tensor([[1, 1, 1, 2, 2, 2],
                    [3, 3, 3, 4, 4, 4]])
            >>> torch.repeat_interleave(y, torch.tensor([1, 2]), dim=0)
            tensor([[1, 2],
                    [3, 4],
                    [3, 4]])

        .. function:: repeat_interleave(repeats) -> Tensor

        If the `repeats` is `tensor([n1, n2, n3, ...])`, then the output will be
        `tensor([0, 0, ..., 1, 1, ..., 2, 2, ..., ...])` where `0` appears `n1` times,
        `1` appears `n2` times, `2` appears `n3` times, etc.
        """
        return super().repeat_interleave(repeats, dim=dim)

    @return_tensor_wrapper
    def requires_grad_(self, requires_grad=True) -> 'Tensor':
        """
        requires_grad_(requires_grad=True) -> Tensor

        Change if autograd should record operations on this tensor: sets this tensor's
        :attr:`requires_grad` attribute in-place. Returns this tensor.

        :func:`requires_grad_`'s main use case is to tell autograd to begin recording
        operations on a Tensor ``tensor``. If ``tensor`` has ``requires_grad=False``
        (because it was obtained through a DataLoader, or required preprocessing or
        initialization), ``tensor.requires_grad_()`` makes it so that autograd will
        begin to record operations on ``tensor``.

        Args:
            requires_grad (bool): If autograd should record operations on this tensor.
                Default: ``True``.

        Example::

            >>> # Let's say we want to preprocess some saved weights and use
            >>> # the result as new weights.
            >>> saved_weights = [0.1, 0.2, 0.3, 0.25]
            >>> loaded_weights = torch.tensor(saved_weights)
            >>> weights = preprocess(loaded_weights)  # some function
            >>> weights
            tensor([-0.5503,  0.4926, -2.1158, -0.8303])

            >>> # Now, start to record operations done to weights
            >>> weights.requires_grad_()
            >>> out = weights.pow(2).sum()
            >>> out.backward()
            >>> weights.grad
            tensor([-1.1007,  0.9853, -4.2316, -1.6606])
        """
        return super().requires_grad_(requires_grad=requires_grad)

    @return_tensor_wrapper
    def reshape(self, *shape) -> 'Tensor':
        """
        reshape(*shape) -> Tensor

        Returns a tensor with the same data and number of elements as :attr:`self`
        but with the specified shape. This method returns a view if :attr:`shape` is
        compatible with the current shape. See :meth:`torch.Tensor.view` on when it is
        possible to return a view.

        See :func:`torch.reshape`

        Args:
            shape (tuple of ints or int...): the desired shape
        """
        return super().reshape(*shape)

    @return_tensor_wrapper
    def reshape_as(self, other) -> 'Tensor':
        """
        reshape_as(other) -> Tensor

        Returns this tensor as the same shape as :attr:`other`.
        ``self.reshape_as(other)`` is equivalent to ``self.reshape(other.sizes())``.
        This method returns a view if ``other.sizes()`` is compatible with the current
        shape. See :meth:`torch.Tensor.view` on when it is possible to return a view.

        Please see :meth:`reshape` for more information about ``reshape``.

        Args:
            other (:class:`torch.Tensor`): The result tensor has the same shape
                as :attr:`other`.
        """
        return super().reshape_as(other)

    @return_tensor_wrapper
    def resize_(self, *sizes, memory_format=torch.contiguous_format) -> 'Tensor':
        """
        resize_(*sizes, memory_format=torch.contiguous_format) -> Tensor

        Resizes :attr:`self` tensor to the specified size. If the number of elements is
        larger than the current storage size, then the underlying storage is resized
        to fit the new number of elements. If the number of elements is smaller, the
        underlying storage is not changed. Existing elements are preserved but any new
        memory is uninitialized.

        .. warning::

            This is a low-level method. The storage is reinterpreted as C-contiguous,
            ignoring the current strides (unless the target size equals the current
            size, in which case the tensor is left unchanged). For most purposes, you
            will instead want to use :meth:`~Tensor.view()`, which checks for
            contiguity, or :meth:`~Tensor.reshape()`, which copies data if needed. To
            change the size in-place with custom strides, see :meth:`~Tensor.set_()`.

        Args:
            sizes (torch.Size or int...): the desired size
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                Tensor. Default: ``torch.contiguous_format``. Note that memory format of
                :attr:`self` is going to be unaffected if ``self.size()`` matches ``sizes``.

        Example::

            >>> x = torch.tensor([[1, 2], [3, 4], [5, 6]])
            >>> x.resize_(2, 2)
            tensor([[ 1,  2],
                    [ 3,  4]])
        """
        return super().resize_(*sizes, memory_format=memory_format)

    @return_tensor_wrapper
    def resize_as_(self, tensor, memory_format=torch.contiguous_format) -> 'Tensor':
        """
        resize_as_(tensor, memory_format=torch.contiguous_format) -> Tensor

        Resizes the :attr:`self` tensor to be the same size as the specified
        :attr:`tensor`. This is equivalent to ``self.resize_(tensor.size())``.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                Tensor. Default: ``torch.contiguous_format``. Note that memory format of
                :attr:`self` is going to be unaffected if ``self.size()`` matches ``tensor.size()``.
        """
        return super().resize_as_(tensor, memory_format=memory_format)

    @return_tensor_wrapper
    def rfft(self, signal_ndim, normalized=False, onesided=True) -> 'Tensor':
        """
        rfft(input, signal_ndim, normalized=False, onesided=True) -> Tensor

        Real-to-complex Discrete Fourier Transform

        This method computes the real-to-complex discrete Fourier transform. It is
        mathematically equivalent with :func:`~torch.fft` with differences only in
        formats of the input and output.

        This method supports 1D, 2D and 3D real-to-complex transforms, indicated
        by :attr:`signal_ndim`. :attr:`input` must be a tensor with at least
        ``signal_ndim`` dimensions with optionally arbitrary number of leading batch
        dimensions. If :attr:`normalized` is set to ``True``, this normalizes the result
        by dividing it with :math:`\sqrt{\prod_{i=1}^K N_i}` so that the operator is
        unitary, where :math:`N_i` is the size of signal dimension :math:`i`.

        The real-to-complex Fourier transform results follow conjugate symmetry:

        .. math::
            X[\omega_1, \dots, \omega_d] = X^*[N_1 - \omega_1, \dots, N_d - \omega_d],

        where the index arithmetic is computed modulus the size of the corresponding
        dimension, :math:`\ ^*` is the conjugate operator, and
        :math:`d` = :attr:`signal_ndim`. :attr:`onesided` flag controls whether to avoid
        redundancy in the output results. If set to ``True`` (default), the output will
        not be full complex result of shape :math:`(*, 2)`, where :math:`*` is the shape
        of :attr:`input`, but instead the last dimension will be halfed as of size
        :math:`\lfloor \frac{N_d}{2} \rfloor + 1`.

        The inverse of this function is :func:`~torch.irfft`.

        .. note::
            For CUDA tensors, an LRU cache is used for cuFFT plans to speed up
            repeatedly running FFT methods on tensors of same geometry with same
            configuration. See :ref:`cufft-plan-cache` for more details on how to
            monitor and control the cache.

        .. warning::
            For CPU tensors, this method is currently only available with MKL. Use
            :func:`torch.backends.mkl.is_available` to check if MKL is installed.

        Arguments:
            input (Tensor): the input tensor of at least :attr:`signal_ndim` dimensions
            signal_ndim (int): the number of dimensions in each signal.
                :attr:`signal_ndim` can only be 1, 2 or 3
            normalized (bool, optional): controls whether to return normalized results.
                Default: ``False``
            onesided (bool, optional): controls whether to return half of results to
                avoid redundancy. Default: ``True``

        Returns:
            Tensor: A tensor containing the real-to-complex Fourier transform result

        Example::

            >>> x = torch.randn(5, 5)
            >>> torch.rfft(x, 2).shape
            torch.Size([5, 3, 2])
            >>> torch.rfft(x, 2, onesided=False).shape
            torch.Size([5, 5, 2])
        """
        return super().rfft(signal_ndim, normalized=normalized, onesided=onesided)

    @return_tensor_wrapper
    def roll(self, shifts, dims) -> 'Tensor':
        """
        roll(input, shifts, dims=None) -> Tensor

        Roll the tensor along the given dimension(s). Elements that are shifted beyond the
        last position are re-introduced at the first position. If a dimension is not
        specified, the tensor will be flattened before rolling and then restored
        to the original shape.

        Args:
            input (Tensor): the input tensor.
            shifts (int or tuple of ints): The number of places by which the elements
                of the tensor are shifted. If shifts is a tuple, dims must be a tuple of
                the same size, and each dimension will be rolled by the corresponding
                value
            dims (int or tuple of ints): Axis along which to roll

        Example::

            >>> x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).view(4, 2)
            >>> x
            tensor([[1, 2],
                    [3, 4],
                    [5, 6],
                    [7, 8]])
            >>> torch.roll(x, 1, 0)
            tensor([[7, 8],
                    [1, 2],
                    [3, 4],
                    [5, 6]])
            >>> torch.roll(x, -1, 0)
            tensor([[3, 4],
                    [5, 6],
                    [7, 8],
                    [1, 2]])
            >>> torch.roll(x, shifts=(2, 1), dims=(0, 1))
            tensor([[6, 5],
                    [8, 7],
                    [2, 1],
                    [4, 3]])
        """
        return super().roll(shifts, dims)

    @return_tensor_wrapper
    def rot90(self, k, dims) -> 'Tensor':
        """
        rot90(input, k, dims) -> Tensor

        Rotate a n-D tensor by 90 degrees in the plane specified by dims axis.
        Rotation direction is from the first towards the second axis if k > 0, and from the second towards the first for k < 0.

        Args:
            input (Tensor): the input tensor.
            k (int): number of times to rotate
            dims (a list or tuple): axis to rotate

        Example::

            >>> x = torch.arange(4).view(2, 2)
            >>> x
            tensor([[0, 1],
                    [2, 3]])
            >>> torch.rot90(x, 1, [0, 1])
            tensor([[1, 3],
                    [0, 2]])

            >>> x = torch.arange(8).view(2, 2, 2)
            >>> x
            tensor([[[0, 1],
                     [2, 3]],

                    [[4, 5],
                     [6, 7]]])
            >>> torch.rot90(x, 1, [1, 2])
            tensor([[[1, 3],
                     [0, 2]],

                    [[5, 7],
                     [4, 6]]])
        """
        return super().rot90(k, dims)

    @return_tensor_wrapper
    def round(self) -> 'Tensor':
        """
        round(input, out=None) -> Tensor

        Returns a new tensor with each of the elements of :attr:`input` rounded
        to the closest integer.

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4)
            >>> a
            tensor([ 0.9920,  0.6077,  0.9734, -1.0362])
            >>> torch.round(a)
            tensor([ 1.,  1.,  1., -1.])
        """
        return super().round()

    @return_tensor_wrapper
    def round_(self) -> 'Tensor':
        """
        round_() -> Tensor

        In-place version of :meth:`~Tensor.round`
        """
        return super().round_()

    @return_tensor_wrapper
    def rsqrt(self) -> 'Tensor':
        """
        rsqrt(input, out=None) -> Tensor

        Returns a new tensor with the reciprocal of the square-root of each of
        the elements of :attr:`input`.

        .. math::
            \text{out}_{i} = \frac{1}{\sqrt{\text{input}_{i}}}

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4)
            >>> a
            tensor([-0.0370,  0.2970,  1.5420, -0.9105])
            >>> torch.rsqrt(a)
            tensor([    nan,  1.8351,  0.8053,     nan])
        """
        return super().rsqrt()

    @return_tensor_wrapper
    def rsqrt_(self) -> 'Tensor':
        """
        rsqrt_() -> Tensor

        In-place version of :meth:`~Tensor.rsqrt`
        """
        return super().rsqrt_()

    @return_tensor_wrapper
    def scatter(self, dim, index, src) -> 'Tensor':
        """
        scatter(dim, index, src) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.scatter_`
        """
        return super().scatter(dim, index, src)

    @return_tensor_wrapper
    def scatter_(self, dim, index, src) -> 'Tensor':
        """
        scatter_(dim, index, src) -> Tensor

        Writes all values from the tensor :attr:`src` into :attr:`self` at the indices
        specified in the :attr:`index` tensor. For each value in :attr:`src`, its output
        index is specified by its index in :attr:`src` for ``dimension != dim`` and by
        the corresponding value in :attr:`index` for ``dimension = dim``.

        For a 3-D tensor, :attr:`self` is updated as::

            self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
            self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
            self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

        This is the reverse operation of the manner described in :meth:`~Tensor.gather`.

        :attr:`self`, :attr:`index` and :attr:`src` (if it is a Tensor) should have same
        number of dimensions. It is also required that ``index.size(d) <= src.size(d)``
        for all dimensions ``d``, and that ``index.size(d) <= self.size(d)`` for all
        dimensions ``d != dim``.

        Moreover, as for :meth:`~Tensor.gather`, the values of :attr:`index` must be
        between ``0`` and ``self.size(dim) - 1`` inclusive, and all values in a row
        along the specified dimension :attr:`dim` must be unique.

        Args:
            dim (int): the axis along which to index
            index (LongTensor): the indices of elements to scatter,
              can be either empty or the same size of src.
              When empty, the operation returns identity
            src (Tensor): the source element(s) to scatter,
              incase `value` is not specified
            value (float): the source element(s) to scatter,
              incase `src` is not specified

        Example::

            >>> x = torch.rand(2, 5)
            >>> x
            tensor([[ 0.3992,  0.2908,  0.9044,  0.4850,  0.6004],
                    [ 0.5735,  0.9006,  0.6797,  0.4152,  0.1732]])
            >>> torch.zeros(3, 5).scatter_(0, torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x)
            tensor([[ 0.3992,  0.9006,  0.6797,  0.4850,  0.6004],
                    [ 0.0000,  0.2908,  0.0000,  0.4152,  0.0000],
                    [ 0.5735,  0.0000,  0.9044,  0.0000,  0.1732]])

            >>> z = torch.zeros(2, 4).scatter_(1, torch.tensor([[2], [3]]), 1.23)
            >>> z
            tensor([[ 0.0000,  0.0000,  1.2300,  0.0000],
                    [ 0.0000,  0.0000,  0.0000,  1.2300]])
        """
        return super().scatter_(dim, index, src)

    @return_tensor_wrapper
    def scatter_add(self, dim, index, src) -> 'Tensor':
        """
        scatter_add(dim, index, src) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.scatter_add_`
        """
        return super().scatter_add(dim, index, src)

    @return_tensor_wrapper
    def scatter_add_(self, dim, index, src) -> 'Tensor':
        """
        scatter_add_(dim, index, src) -> Tensor

        Adds all values from the tensor :attr:`other` into :attr:`self` at the indices
        specified in the :attr:`index` tensor in a similar fashion as
        :meth:`~torch.Tensor.scatter_`. For each value in :attr:`src`, it is added to
        an index in :attr:`self` which is specified by its index in :attr:`src`
        for ``dimension != dim`` and by the corresponding value in :attr:`index` for
        ``dimension = dim``.

        For a 3-D tensor, :attr:`self` is updated as::

            self[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0
            self[i][index[i][j][k]][k] += src[i][j][k]  # if dim == 1
            self[i][j][index[i][j][k]] += src[i][j][k]  # if dim == 2

        :attr:`self`, :attr:`index` and :attr:`src` should have same number of
        dimensions. It is also required that ``index.size(d) <= src.size(d)`` for all
        dimensions ``d``, and that ``index.size(d) <= self.size(d)`` for all dimensions
        ``d != dim``.

        Note:
            In some circumstances when using the CUDA backend with CuDNN, this operator
            may select a nondeterministic algorithm to increase performance. If this is
            undesirable, you can try to make the operation deterministic (potentially at
            a performance cost) by setting ``torch.backends.cudnn.deterministic =
            True``.
            Please see the notes on :doc:`/notes/randomness` for background.

        Args:
            dim (int): the axis along which to index
            index (LongTensor): the indices of elements to scatter and add,
              can be either empty or the same size of src.
              When empty, the operation returns identity.
            src (Tensor): the source elements to scatter and add

        Example::

            >>> x = torch.rand(2, 5)
            >>> x
            tensor([[0.7404, 0.0427, 0.6480, 0.3806, 0.8328],
                    [0.7953, 0.2009, 0.9154, 0.6782, 0.9620]])
            >>> torch.ones(3, 5).scatter_add_(0, torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x)
            tensor([[1.7404, 1.2009, 1.9154, 1.3806, 1.8328],
                    [1.0000, 1.0427, 1.0000, 1.6782, 1.0000],
                    [1.7953, 1.0000, 1.6480, 1.0000, 1.9620]])
        """
        return super().scatter_add_(dim, index, src)

    @return_tensor_wrapper
    def select(self, dim, index) -> 'Tensor':
        """
        select(dim, index) -> Tensor

        Slices the :attr:`self` tensor along the selected dimension at the given index.
        This function returns a view of the original tensor with the given dimension removed.

        Args:
            dim (int): the dimension to slice
            index (int): the index to select with

        .. note::

            :meth:`select` is equivalent to slicing. For example,
            ``tensor.select(0, index)`` is equivalent to ``tensor[index]`` and
            ``tensor.select(2, index)`` is equivalent to ``tensor[:,:,index]``.
        """
        return super().select(dim, index)

    @return_tensor_wrapper
    def set_(self, source=None, storage_offset=0, size=None, stride=None) -> 'Tensor':
        """
        set_(source=None, storage_offset=0, size=None, stride=None) -> Tensor

        Sets the underlying storage, size, and strides. If :attr:`source` is a tensor,
        :attr:`self` tensor will share the same storage and have the same size and
        strides as :attr:`source`. Changes to elements in one tensor will be reflected
        in the other.

        If :attr:`source` is a :class:`~torch.Storage`, the method sets the underlying
        storage, offset, size, and stride.

        Args:
            source (Tensor or Storage): the tensor or storage to use
            storage_offset (int, optional): the offset in the storage
            size (torch.Size, optional): the desired size. Defaults to the size of the source.
            stride (tuple, optional): the desired stride. Defaults to C-contiguous strides.
        """
        return super().set_(source=source, storage_offset=storage_offset, size=size, stride=stride)

    @return_tensor_wrapper
    def short(self, memory_format=torch.preserve_format) -> 'Tensor':
        """
        short(memory_format=torch.preserve_format) -> Tensor

        ``self.short()`` is equivalent to ``self.to(torch.int16)``. See :func:`to`.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
        return super().short(memory_format=memory_format)

    @return_tensor_wrapper
    def sigmoid(self) -> 'Tensor':
        """
        sigmoid(input, out=None) -> Tensor

        Returns a new tensor with the sigmoid of the elements of :attr:`input`.

        .. math::
            \text{out}_{i} = \frac{1}{1 + e^{-\text{input}_{i}}}

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4)
            >>> a
            tensor([ 0.9213,  1.0887, -0.8858, -1.7683])
            >>> torch.sigmoid(a)
            tensor([ 0.7153,  0.7481,  0.2920,  0.1458])
        """
        return super().sigmoid()

    @return_tensor_wrapper
    def sigmoid_(self) -> 'Tensor':
        """
        sigmoid_() -> Tensor

        In-place version of :meth:`~Tensor.sigmoid`
        """
        return super().sigmoid_()

    @return_tensor_wrapper
    def sign(self) -> 'Tensor':
        """
        sign(input, out=None) -> Tensor

        Returns a new tensor with the signs of the elements of :attr:`input`.

        .. math::
            \text{out}_{i} = \operatorname{sgn}(\text{input}_{i})

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.tensor([0.7, -1.2, 0., 2.3])
            >>> a
            tensor([ 0.7000, -1.2000,  0.0000,  2.3000])
            >>> torch.sign(a)
            tensor([ 1., -1.,  0.,  1.])
        """
        return super().sign()

    @return_tensor_wrapper
    def sign_(self) -> 'Tensor':
        """
        sign_() -> Tensor

        In-place version of :meth:`~Tensor.sign`
        """
        return super().sign_()

    @return_tensor_wrapper
    def sin(self) -> 'Tensor':
        """
        sin(input, out=None) -> Tensor

        Returns a new tensor with the sine of the elements of :attr:`input`.

        .. math::
            \text{out}_{i} = \sin(\text{input}_{i})

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4)
            >>> a
            tensor([-0.5461,  0.1347, -2.7266, -0.2746])
            >>> torch.sin(a)
            tensor([-0.5194,  0.1343, -0.4032, -0.2711])
        """
        return super().sin()

    @return_tensor_wrapper
    def sin_(self) -> 'Tensor':
        """
        sin_() -> Tensor

        In-place version of :meth:`~Tensor.sin`
        """
        return super().sin_()

    @return_tensor_wrapper
    def sinh(self) -> 'Tensor':
        """
        sinh(input, out=None) -> Tensor

        Returns a new tensor with the hyperbolic sine of the elements of
        :attr:`input`.

        .. math::
            \text{out}_{i} = \sinh(\text{input}_{i})

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4)
            >>> a
            tensor([ 0.5380, -0.8632, -0.1265,  0.9399])
            >>> torch.sinh(a)
            tensor([ 0.5644, -0.9744, -0.1268,  1.0845])
        """
        return super().sinh()

    @return_tensor_wrapper
    def sinh_(self) -> 'Tensor':
        """
        sinh_() -> Tensor

        In-place version of :meth:`~Tensor.sinh`
        """
        return super().sinh_()

    @return_tensor_wrapper
    def solve(self, A) -> Tuple[['Tensor', 'Tensor']]:
        """
        torch.solve(input, A, out=None) -> (Tensor, Tensor)

        This function returns the solution to the system of linear
        equations represented by :math:`AX = B` and the LU factorization of
        A, in order as a namedtuple `solution, LU`.

        `LU` contains `L` and `U` factors for LU factorization of `A`.

        `torch.solve(B, A)` can take in 2D inputs `B, A` or inputs that are
        batches of 2D matrices. If the inputs are batches, then returns
        batched outputs `solution, LU`.

        .. note::

            Irrespective of the original strides, the returned matrices
            `solution` and `LU` will be transposed, i.e. with strides like
            `B.contiguous().transpose(-1, -2).stride()` and
            `A.contiguous().transpose(-1, -2).stride()` respectively.

        Args:
            input (Tensor): input matrix :math:`B` of size :math:`(*, m, k)` , where :math:`*`
                        is zero or more batch dimensions.
            A (Tensor): input square matrix of size :math:`(*, m, m)`, where
                        :math:`*` is zero or more batch dimensions.
            out ((Tensor, Tensor), optional): optional output tuple.

        Example::

            >>> A = torch.tensor([[6.80, -2.11,  5.66,  5.97,  8.23],
                                  [-6.05, -3.30,  5.36, -4.44,  1.08],
                                  [-0.45,  2.58, -2.70,  0.27,  9.04],
                                  [8.32,  2.71,  4.35,  -7.17,  2.14],
                                  [-9.67, -5.14, -7.26,  6.08, -6.87]]).t()
            >>> B = torch.tensor([[4.02,  6.19, -8.22, -7.57, -3.03],
                                  [-1.56,  4.00, -8.67,  1.75,  2.86],
                                  [9.81, -4.09, -4.57, -8.61,  8.99]]).t()
            >>> X, LU = torch.solve(B, A)
            >>> torch.dist(B, torch.mm(A, X))
            tensor(1.00000e-06 *
                   7.0977)

            >>> # Batched solver example
            >>> A = torch.randn(2, 3, 1, 4, 4)
            >>> B = torch.randn(2, 3, 1, 4, 6)
            >>> X, LU = torch.solve(B, A)
            >>> torch.dist(B, A.matmul(X))
            tensor(1.00000e-06 *
               3.6386)
        """
        return super().solve(A)

    @return_tensor_wrapper
    def sparse_mask(self, input, mask) -> 'Tensor':
        """
        sparse_mask(input, mask) -> Tensor

        Returns a new SparseTensor with values from Tensor :attr:`input` filtered
        by indices of :attr:`mask` and values are ignored. :attr:`input` and :attr:`mask`
        must have the same shape.

        Args:
            input (Tensor): an input Tensor
            mask (SparseTensor): a SparseTensor which we filter :attr:`input` based on its indices

        Example::

            >>> nnz = 5
            >>> dims = [5, 5, 2, 2]
            >>> I = torch.cat([torch.randint(0, dims[0], size=(nnz,)),
                               torch.randint(0, dims[1], size=(nnz,))], 0).reshape(2, nnz)
            >>> V = torch.randn(nnz, dims[2], dims[3])
            >>> size = torch.Size(dims)
            >>> S = torch.sparse_coo_tensor(I, V, size).coalesce()
            >>> D = torch.randn(dims)
            >>> D.sparse_mask(S)
            tensor(indices=tensor([[0, 0, 0, 2],
                                   [0, 1, 4, 3]]),
                   values=tensor([[[ 1.6550,  0.2397],
                                   [-0.1611, -0.0779]],

                                  [[ 0.2326, -1.0558],
                                   [ 1.4711,  1.9678]],

                                  [[-0.5138, -0.0411],
                                   [ 1.9417,  0.5158]],

                                  [[ 0.0793,  0.0036],
                                   [-0.2569, -0.1055]]]),
                   size=(5, 5, 2, 2), nnz=4, layout=torch.sparse_coo)
        """
        return super().sparse_mask(input, mask)

    @return_tensor_wrapper
    def sqrt(self) -> 'Tensor':
        """
        sqrt(input, out=None) -> Tensor

        Returns a new tensor with the square-root of the elements of :attr:`input`.

        .. math::
            \text{out}_{i} = \sqrt{\text{input}_{i}}

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4)
            >>> a
            tensor([-2.0755,  1.0226,  0.0831,  0.4806])
            >>> torch.sqrt(a)
            tensor([    nan,  1.0112,  0.2883,  0.6933])
        """
        return super().sqrt()

    @return_tensor_wrapper
    def sqrt_(self) -> 'Tensor':
        """
        sqrt_() -> Tensor

        In-place version of :meth:`~Tensor.sqrt`
        """
        return super().sqrt_()

    @return_tensor_wrapper
    def square(self) -> 'Tensor':
        """
        square(input, out=None) -> Tensor

        Returns a new tensor with the square of the elements of :attr:`input`.

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4)
            >>> a
            tensor([-2.0755,  1.0226,  0.0831,  0.4806])
            >>> torch.square(a)
            tensor([ 4.3077,  1.0457,  0.0069,  0.2310])
        """
        return super().square()

    @return_tensor_wrapper
    def square_(self) -> 'Tensor':
        """
        square_() -> Tensor

        In-place version of :meth:`~Tensor.square`
        """
        return super().square_()

    @return_tensor_wrapper
    def squeeze(self, dim=None) -> 'Tensor':
        """
        squeeze(input, dim=None, out=None) -> Tensor

        Returns a tensor with all the dimensions of :attr:`input` of size `1` removed.

        For example, if `input` is of shape:
        :math:`(A \times 1 \times B \times C \times 1 \times D)` then the `out` tensor
        will be of shape: :math:`(A \times B \times C \times D)`.

        When :attr:`dim` is given, a squeeze operation is done only in the given
        dimension. If `input` is of shape: :math:`(A \times 1 \times B)`,
        ``squeeze(input, 0)`` leaves the tensor unchanged, but ``squeeze(input, 1)``
        will squeeze the tensor to the shape :math:`(A \times B)`.

        .. note:: The returned tensor shares the storage with the input tensor,
                  so changing the contents of one will change the contents of the other.

        .. warning:: If the tensor has a batch dimension of size 1, then `squeeze(input)`
                  will also remove the batch dimension, which can lead to unexpected
                  errors.

        Args:
            input (Tensor): the input tensor.
            dim (int, optional): if given, the input will be squeezed only in
                   this dimension
            out (Tensor, optional): the output tensor.

        Example::

            >>> x = torch.zeros(2, 1, 2, 1, 2)
            >>> x.size()
            torch.Size([2, 1, 2, 1, 2])
            >>> y = torch.squeeze(x)
            >>> y.size()
            torch.Size([2, 2, 2])
            >>> y = torch.squeeze(x, 0)
            >>> y.size()
            torch.Size([2, 1, 2, 1, 2])
            >>> y = torch.squeeze(x, 1)
            >>> y.size()
            torch.Size([2, 2, 1, 2])
        """
        if dim is None:
            return super().squeeze()
        return super().squeeze(dim=dim)

    @return_tensor_wrapper
    def squeeze_(self, dim=None) -> 'Tensor':
        """
        squeeze_(dim=None) -> Tensor

        In-place version of :meth:`~Tensor.squeeze`
        """
        return super().squeeze_(dim=dim)

    @return_tensor_wrapper
    def std(self, dim=None, unbiased=True, keepdim=False) -> 'Tensor':
        """
        std(input, unbiased=True) -> Tensor

        Returns the standard-deviation of all elements in the :attr:`input` tensor.

        If :attr:`unbiased` is ``False``, then the standard-deviation will be calculated
        via the biased estimator. Otherwise, Bessel's correction will be used.

        Args:
            input (Tensor): the input tensor.
            unbiased (bool): whether to use the unbiased estimation or not

        Example::

            >>> a = torch.randn(1, 3)
            >>> a
            tensor([[-0.8166, -1.3802, -0.3560]])
            >>> torch.std(a)
            tensor(0.5130)

        .. function:: std(input, dim, unbiased=True, keepdim=False, out=None) -> Tensor

        Returns the standard-deviation of each row of the :attr:`input` tensor in the
        dimension :attr:`dim`. If :attr:`dim` is a list of dimensions,
        reduce over all of them.


        If :attr:`keepdim` is ``True``, the output tensor is of the same size
        as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
        Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
        output tensor having 1 (or ``len(dim)``) fewer dimension(s).


        If :attr:`unbiased` is ``False``, then the standard-deviation will be calculated
        via the biased estimator. Otherwise, Bessel's correction will be used.

        Args:
            input (Tensor): the input tensor.
            dim (int or tuple of ints): the dimension or dimensions to reduce.
            unbiased (bool): whether to use the unbiased estimation or not
            keepdim (bool): whether the output tensor has :attr:`dim` retained or not.
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4, 4)
            >>> a
            tensor([[ 0.2035,  1.2959,  1.8101, -0.4644],
                    [ 1.5027, -0.3270,  0.5905,  0.6538],
                    [-1.5745,  1.3330, -0.5596, -0.6548],
                    [ 0.1264, -0.5080,  1.6420,  0.1992]])
            >>> torch.std(a, dim=1)
            tensor([ 1.0311,  0.7477,  1.2204,  0.9087])
        """
        return super().std(dim=dim, unbiased=unbiased, keepdim=keepdim)

    @return_tensor_wrapper
    def sub(self, other, *, alpha=1) -> 'Tensor':
        """
        sub(other, *, alpha=1) -> Tensor

        Subtracts a scalar or tensor from :attr:`self` tensor. If both :attr:`alpha`
        and :attr:`other` are specified, each element of :attr:`other` is scaled by
        :attr:`alpha` before being used.

        When :attr:`other` is a tensor, the shape of :attr:`other` must be
        :ref:`broadcastable <broadcasting-semantics>` with the shape of the underlying
        tensor.
        """
        return super().sub(other, alpha=alpha)

    @return_tensor_wrapper
    def sub_(self, other, *, alpha=1) -> 'Tensor':
        """
        sub_(other, *, alpha=1) -> Tensor

        In-place version of :meth:`~Tensor.sub`
        """
        return super().sub_(other, alpha=alpha)

    @return_tensor_wrapper
    def sum(self, *args, **kwargs) -> 'Tensor':
        """
        sum(input, dtype=None) -> Tensor

        Returns the sum of all elements in the :attr:`input` tensor.

        Args:
            input (Tensor): the input tensor.
            dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
                If specified, the input tensor is casted to :attr:`dtype` before the operation
                is performed. This is useful for preventing data type overflows. Default: None.

        Example::

            >>> a = torch.randn(1, 3)
            >>> a
            tensor([[ 0.1133, -0.9567,  0.2958]])
            >>> torch.sum(a)
            tensor(-0.5475)

        .. function:: sum(input, dim, keepdim=False, dtype=None) -> Tensor

        Returns the sum of each row of the :attr:`input` tensor in the given
        dimension :attr:`dim`. If :attr:`dim` is a list of dimensions,
        reduce over all of them.


        If :attr:`keepdim` is ``True``, the output tensor is of the same size
        as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
        Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
        output tensor having 1 (or ``len(dim)``) fewer dimension(s).


        Args:
            input (Tensor): the input tensor.
            dim (int or tuple of ints): the dimension or dimensions to reduce.
            keepdim (bool): whether the output tensor has :attr:`dim` retained or not.
            dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
                If specified, the input tensor is casted to :attr:`dtype` before the operation
                is performed. This is useful for preventing data type overflows. Default: None.

        Example::

            >>> a = torch.randn(4, 4)
            >>> a
            tensor([[ 0.0569, -0.2475,  0.0737, -0.3429],
                    [-0.2993,  0.9138,  0.9337, -1.6864],
                    [ 0.1132,  0.7892, -0.1003,  0.5688],
                    [ 0.3637, -0.9906, -0.4752, -1.5197]])
            >>> torch.sum(a, 1)
            tensor([-0.4598, -0.1381,  1.3708, -2.6217])
            >>> b = torch.arange(4 * 5 * 6).view(4, 5, 6)
            >>> torch.sum(b, (2, 1))
            tensor([  435.,  1335.,  2235.,  3135.])
        """
        return super().sum(*args, **kwargs)

    @return_tensor_wrapper
    def sum_to_size(self, *size) -> 'Tensor':
        """
        sum_to_size(*size) -> Tensor

        Sum ``this`` tensor to :attr:`size`.
        :attr:`size` must be broadcastable to ``this`` tensor size.

        Args:
            size (int...): a sequence of integers defining the shape of the output tensor.
        """
        return super().sum_to_size(*size)

    @return_tensor_wrapper
    def t(self) -> 'Tensor':
        """
        t(input) -> Tensor

        Expects :attr:`input` to be <= 2-D tensor and transposes dimensions 0
        and 1.

        0-D and 1-D tensors are returned as is. When input is a 2-D tensor this
        is equivalent to ``transpose(input, 0, 1)``.

        Args:
            input (Tensor): the input tensor.

        Example::

            >>> x = torch.randn(())
            >>> x
            tensor(0.1995)
            >>> torch.t(x)
            tensor(0.1995)
            >>> x = torch.randn(3)
            >>> x
            tensor([ 2.4320, -0.4608,  0.7702])
            >>> torch.t(x)
            tensor([ 2.4320, -0.4608,  0.7702])
            >>> x = torch.randn(2, 3)
            >>> x
            tensor([[ 0.4875,  0.9158, -0.5872],
                    [ 0.3938, -0.6929,  0.6932]])
            >>> torch.t(x)
            tensor([[ 0.4875,  0.3938],
                    [ 0.9158, -0.6929],
                    [-0.5872,  0.6932]])
        """
        return self.T

    @return_tensor_wrapper
    def t_(self) -> 'Tensor':
        """
        t_() -> Tensor

        In-place version of :meth:`~Tensor.t`
        """
        if not self.has_special: return super().t_()
        s = self.shape.special
        if len(s) == 1:
            for i in range(s[0] // 2):
                self.transpose_(i, s[0] - i - 1)
            for i in range((self.ndim - s[0] - 1) // 2):
                self.transpose_(s[0] + i + 1, self.ndim - i - 1)
        elif len(s) == 2:
            for i in range(s[0] // 2):
                self.transpose_(i, s[0] - i - 1)
            for i in range((s[1] - s[0] - 1) // 2):
                self.transpose_(s[0] + i + 1, s[1] - i - 1)
            for i in range((self.ndim - s[1] - 1) // 2):
                self.transpose_(s[1] + i + 1, self.ndim - i - 1)
        return self

    @return_tensor_wrapper
    def take(self, indices) -> 'Tensor':
        """
        take(input, index) -> Tensor

        Returns a new tensor with the elements of :attr:`input` at the given indices.
        The input tensor is treated as if it were viewed as a 1-D tensor. The result
        takes the same shape as the indices.

        Args:
            input (Tensor): the input tensor.
            indices (LongTensor): the indices into tensor

        Example::

            >>> src = torch.tensor([[4, 3, 5],
                                    [6, 7, 8]])
            >>> torch.take(src, torch.tensor([0, 2, 5]))
            tensor([ 4,  5,  8])
        """
        return super().take(indices)

    @return_tensor_wrapper
    def tan(self) -> 'Tensor':
        """
        tan(input, out=None) -> Tensor

        Returns a new tensor with the tangent of the elements of :attr:`input`.

        .. math::
            \text{out}_{i} = \tan(\text{input}_{i})

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4)
            >>> a
            tensor([-1.2027, -1.7687,  0.4412, -1.3856])
            >>> torch.tan(a)
            tensor([-2.5930,  4.9859,  0.4722, -5.3366])
        """
        return super().tan()

    @return_tensor_wrapper
    def tan_(self) -> 'Tensor':
        """
        tan_() -> Tensor

        In-place version of :meth:`~Tensor.tan`
        """
        return super().tan_()

    @return_tensor_wrapper
    def tanh(self) -> 'Tensor':
        """
        tanh(input, out=None) -> Tensor

        Returns a new tensor with the hyperbolic tangent of the elements
        of :attr:`input`.

        .. math::
            \text{out}_{i} = \tanh(\text{input}_{i})

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4)
            >>> a
            tensor([ 0.8986, -0.7279,  1.1745,  0.2611])
            >>> torch.tanh(a)
            tensor([ 0.7156, -0.6218,  0.8257,  0.2553])
        """
        return super().tanh()

    @return_tensor_wrapper
    def tanh_(self) -> 'Tensor':
        """
        tanh_() -> Tensor

        In-place version of :meth:`~Tensor.tanh`
        """
        return super().tanh_()

    def _to(self, *args, **kwargs) -> 'Tensor':
        return super().to(*args, **kwargs)

    @return_tensor_wrapper(False)
    def to(self, *args, **kwargs) -> 'Tensor':
        """
        to(*args, **kwargs) -> Tensor

        Performs Tensor dtype and/or device conversion. A :class:`torch.dtype` and :class:`torch.device` are
        inferred from the arguments of ``self.to(*args, **kwargs)``.

        .. note::

            If the ``self`` Tensor already
            has the correct :class:`torch.dtype` and :class:`torch.device`, then ``self`` is returned.
            Otherwise, the returned tensor is a copy of ``self`` with the desired
            :class:`torch.dtype` and :class:`torch.device`.

        Here are the ways to call ``to``:

        .. function:: to(dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format) -> Tensor

            Returns a Tensor with the specified :attr:`dtype`

            Args:
                memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.

        .. function:: to(device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format) -> Tensor

            Returns a Tensor with the specified :attr:`device` and (optional)
            :attr:`dtype`. If :attr:`dtype` is ``None`` it is inferred to be ``self.dtype``.
            When :attr:`non_blocking`, tries to convert asynchronously with respect to
            the host if possible, e.g., converting a CPU Tensor with pinned memory to a
            CUDA Tensor.
            When :attr:`copy` is set, a new Tensor is created even when the Tensor
            already matches the desired conversion.

            Args:
                memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.

        .. function:: to(other, non_blocking=False, copy=False) -> Tensor

            Returns a Tensor with same :class:`torch.dtype` and :class:`torch.device` as
            the Tensor :attr:`other`. When :attr:`non_blocking`, tries to convert
            asynchronously with respect to the host if possible, e.g., converting a CPU
            Tensor with pinned memory to a CUDA Tensor.
            When :attr:`copy` is set, a new Tensor is created even when the Tensor
            already matches the desired conversion.

        Example::

            >>> tensor = torch.randn(2, 2)  # Initially dtype=float32, device=cpu
            >>> tensor.to(torch.float64)
            tensor([[-0.5044,  0.0005],
                    [ 0.3310, -0.0584]], dtype=torch.float64)

            >>> cuda0 = torch.device('cuda:0')
            >>> tensor.to(cuda0)
            tensor([[-0.5044,  0.0005],
                    [ 0.3310, -0.0584]], device='cuda:0')

            >>> tensor.to(cuda0, dtype=torch.float64)
            tensor([[-0.5044,  0.0005],
                    [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')

            >>> other = torch.randn((), dtype=torch.float64, device=cuda0)
            >>> tensor.to(other, non_blocking=True)
            tensor([[-0.5044,  0.0005],
                    [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')
        """
        return super().to(*args, **kwargs)

    @return_tensor_wrapper
    def to_mkldnn(self) -> 'Tensor':
        """
        to_mkldnn() -> Tensor
        Returns a copy of the tensor in ``torch.mkldnn`` layout.
        """
        return super().to_mkldnn()

    @return_tensor_wrapper
    def to_sparse(self, sparseDims) -> 'Tensor':
        """
        to_sparse(sparseDims) -> Tensor
        Returns a sparse copy of the tensor.  PyTorch supports sparse tensors in
        :ref:`coordinate format <sparse-docs>`.

        Args:
            sparseDims (int, optional): the number of sparse dimensions to include in the new sparse tensor

        Example::

            >>> d = torch.tensor([[0, 0, 0], [9, 0, 10], [0, 0, 0]])
            >>> d
            tensor([[ 0,  0,  0],
                    [ 9,  0, 10],
                    [ 0,  0,  0]])
            >>> d.to_sparse()
            tensor(indices=tensor([[1, 1],
                                   [0, 2]]),
                   values=tensor([ 9, 10]),
                   size=(3, 3), nnz=2, layout=torch.sparse_coo)
            >>> d.to_sparse(1)
            tensor(indices=tensor([[1]]),
                   values=tensor([[ 9,  0, 10]]),
                   size=(3, 3), nnz=1, layout=torch.sparse_coo)
        """
        return super().to_sparse(sparseDims)

    @return_tensor_wrapper
    def trace(self) -> 'Tensor':
        """
        trace(input) -> Tensor

        Returns the sum of the elements of the diagonal of the input 2-D matrix.

        Example::

            >>> x = torch.arange(1., 10.).view(3, 3)
            >>> x
            tensor([[ 1.,  2.,  3.],
                    [ 4.,  5.,  6.],
                    [ 7.,  8.,  9.]])
            >>> torch.trace(x)
            tensor(15.)
        """
        return super().trace()

    @return_tensor_wrapper
    def transpose(self, dim0, dim1) -> 'Tensor':
        """
        transpose(input, dim0, dim1) -> Tensor

        Returns a tensor that is a transposed version of :attr:`input`.
        The given dimensions :attr:`dim0` and :attr:`dim1` are swapped.

        The resulting :attr:`out` tensor shares it's underlying storage with the
        :attr:`input` tensor, so changing the content of one would change the content
        of the other.

        Args:
            input (Tensor): the input tensor.
            dim0 (int): the first dimension to be transposed
            dim1 (int): the second dimension to be transposed

        Example::

            >>> x = torch.randn(2, 3)
            >>> x
            tensor([[ 1.0028, -0.9893,  0.5809],
                    [-0.1669,  0.7299,  0.4942]])
            >>> torch.transpose(x, 0, 1)
            tensor([[ 1.0028, -0.1669],
                    [-0.9893,  0.7299],
                    [ 0.5809,  0.4942]])
        """
        return super().transpose(dim0, dim1)

    @return_tensor_wrapper
    def transpose_(self, dim0, dim1) -> 'Tensor':
        """
        transpose_(dim0, dim1) -> Tensor

        In-place version of :meth:`~Tensor.transpose`
        """
        return super().transpose_(dim0, dim1)

    @return_tensor_wrapper
    def tril(self, k=0) -> 'Tensor':
        """
        tril(input, diagonal=0, out=None) -> Tensor

        Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices
        :attr:`input`, the other elements of the result tensor :attr:`out` are set to 0.

        The lower triangular part of the matrix is defined as the elements on and
        below the diagonal.

        The argument :attr:`diagonal` controls which diagonal to consider. If
        :attr:`diagonal` = 0, all elements on and below the main diagonal are
        retained. A positive value includes just as many diagonals above the main
        diagonal, and similarly a negative value excludes just as many diagonals below
        the main diagonal. The main diagonal are the set of indices
        :math:`\lbrace (i, i) \rbrace` for :math:`i \in [0, \min\{d_{1}, d_{2}\} - 1]` where
        :math:`d_{1}, d_{2}` are the dimensions of the matrix.

        Args:
            input (Tensor): the input tensor.
            diagonal (int, optional): the diagonal to consider
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(3, 3)
            >>> a
            tensor([[-1.0813, -0.8619,  0.7105],
                    [ 0.0935,  0.1380,  2.2112],
                    [-0.3409, -0.9828,  0.0289]])
            >>> torch.tril(a)
            tensor([[-1.0813,  0.0000,  0.0000],
                    [ 0.0935,  0.1380,  0.0000],
                    [-0.3409, -0.9828,  0.0289]])

            >>> b = torch.randn(4, 6)
            >>> b
            tensor([[ 1.2219,  0.5653, -0.2521, -0.2345,  1.2544,  0.3461],
                    [ 0.4785, -0.4477,  0.6049,  0.6368,  0.8775,  0.7145],
                    [ 1.1502,  3.2716, -1.1243, -0.5413,  0.3615,  0.6864],
                    [-0.0614, -0.7344, -1.3164, -0.7648, -1.4024,  0.0978]])
            >>> torch.tril(b, diagonal=1)
            tensor([[ 1.2219,  0.5653,  0.0000,  0.0000,  0.0000,  0.0000],
                    [ 0.4785, -0.4477,  0.6049,  0.0000,  0.0000,  0.0000],
                    [ 1.1502,  3.2716, -1.1243, -0.5413,  0.0000,  0.0000],
                    [-0.0614, -0.7344, -1.3164, -0.7648, -1.4024,  0.0000]])
            >>> torch.tril(b, diagonal=-1)
            tensor([[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                    [ 0.4785,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                    [ 1.1502,  3.2716,  0.0000,  0.0000,  0.0000,  0.0000],
                    [-0.0614, -0.7344, -1.3164,  0.0000,  0.0000,  0.0000]])
        """
        return super().tril(k=k)

    @return_tensor_wrapper
    def tril_(self, k=0) -> 'Tensor':
        """
        tril_(k=0) -> Tensor

        In-place version of :meth:`~Tensor.tril`
        """
        return super().tril_(k=k)

    @return_tensor_wrapper
    def triu(self, k=0) -> 'Tensor':
        """
        triu(input, diagonal=0, out=None) -> Tensor

        Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices
        :attr:`input`, the other elements of the result tensor :attr:`out` are set to 0.

        The upper triangular part of the matrix is defined as the elements on and
        above the diagonal.

        The argument :attr:`diagonal` controls which diagonal to consider. If
        :attr:`diagonal` = 0, all elements on and above the main diagonal are
        retained. A positive value excludes just as many diagonals above the main
        diagonal, and similarly a negative value includes just as many diagonals below
        the main diagonal. The main diagonal are the set of indices
        :math:`\lbrace (i, i) \rbrace` for :math:`i \in [0, \min\{d_{1}, d_{2}\} - 1]` where
        :math:`d_{1}, d_{2}` are the dimensions of the matrix.

        Args:
            input (Tensor): the input tensor.
            diagonal (int, optional): the diagonal to consider
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(3, 3)
            >>> a
            tensor([[ 0.2309,  0.5207,  2.0049],
                    [ 0.2072, -1.0680,  0.6602],
                    [ 0.3480, -0.5211, -0.4573]])
            >>> torch.triu(a)
            tensor([[ 0.2309,  0.5207,  2.0049],
                    [ 0.0000, -1.0680,  0.6602],
                    [ 0.0000,  0.0000, -0.4573]])
            >>> torch.triu(a, diagonal=1)
            tensor([[ 0.0000,  0.5207,  2.0049],
                    [ 0.0000,  0.0000,  0.6602],
                    [ 0.0000,  0.0000,  0.0000]])
            >>> torch.triu(a, diagonal=-1)
            tensor([[ 0.2309,  0.5207,  2.0049],
                    [ 0.2072, -1.0680,  0.6602],
                    [ 0.0000, -0.5211, -0.4573]])

            >>> b = torch.randn(4, 6)
            >>> b
            tensor([[ 0.5876, -0.0794, -1.8373,  0.6654,  0.2604,  1.5235],
                    [-0.2447,  0.9556, -1.2919,  1.3378, -0.1768, -1.0857],
                    [ 0.4333,  0.3146,  0.6576, -1.0432,  0.9348, -0.4410],
                    [-0.9888,  1.0679, -1.3337, -1.6556,  0.4798,  0.2830]])
            >>> torch.triu(b, diagonal=1)
            tensor([[ 0.0000, -0.0794, -1.8373,  0.6654,  0.2604,  1.5235],
                    [ 0.0000,  0.0000, -1.2919,  1.3378, -0.1768, -1.0857],
                    [ 0.0000,  0.0000,  0.0000, -1.0432,  0.9348, -0.4410],
                    [ 0.0000,  0.0000,  0.0000,  0.0000,  0.4798,  0.2830]])
            >>> torch.triu(b, diagonal=-1)
            tensor([[ 0.5876, -0.0794, -1.8373,  0.6654,  0.2604,  1.5235],
                    [-0.2447,  0.9556, -1.2919,  1.3378, -0.1768, -1.0857],
                    [ 0.0000,  0.3146,  0.6576, -1.0432,  0.9348, -0.4410],
                    [ 0.0000,  0.0000, -1.3337, -1.6556,  0.4798,  0.2830]])
        """
        return super().triu(k=k)

    @return_tensor_wrapper
    def triu_(self, k=0) -> 'Tensor':
        """
        triu_(k=0) -> Tensor

        In-place version of :meth:`~Tensor.triu`
        """
        return super().triu_(k=k)

    @return_tensor_wrapper
    def true_divide(self, value) -> 'Tensor':
        """
        true_divide(dividend, divisor) -> Tensor

        Performs "true division" that always computes the division
        in floating point. Analogous to division in Python 3 and equivalent to
        :func:`torch.div` except when both inputs have bool or integer scalar types,
        in which case they are cast to the default (floating) scalar type before the division.

        .. math::
            \text{out}_i = \frac{\text{dividend}_i}{\text{divisor}}

        Args:
            dividend (Tensor): the dividend
            divisor (Tensor or Scalar): the divisor

        Keyword args:
            out (Tensor, optional): the output tensor.

        Example::

            >>> dividend = torch.tensor([5, 3], dtype=torch.int)
            >>> divisor = torch.tensor([3, 2], dtype=torch.int)
            >>> torch.true_divide(dividend, divisor)
            tensor([1.6667, 1.5000])
            >>> torch.true_divide(dividend, 2)
            tensor([2.5000, 1.5000])
        """
        return super().true_divide(value)

    @return_tensor_wrapper
    def true_divide_(self, value) -> 'Tensor':
        """
        true_divide_(value) -> Tensor

        In-place version of :meth:`~Tensor.true_divide_`
        """
        return super().true_divide_(value)

    @return_tensor_wrapper
    def trunc(self) -> 'Tensor':
        """
        trunc(input, out=None) -> Tensor

        Returns a new tensor with the truncated integer values of
        the elements of :attr:`input`.

        Args:
            input (Tensor): the input tensor.
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4)
            >>> a
            tensor([ 3.4742,  0.5466, -0.8008, -0.9079])
            >>> torch.trunc(a)
            tensor([ 3.,  0., -0., -0.])
        """
        return super().trunc()

    @return_tensor_wrapper
    def trunc_(self) -> 'Tensor':
        """
        trunc_() -> Tensor

        In-place version of :meth:`~Tensor.trunc`
        """
        return super().trunc_()

    def _type(self, dtype=None, non_blocking=False, **kwargs) -> str or 'Tensor':
        """
        type(dtype=None, non_blocking=False, **kwargs) -> str or Tensor
        Returns the type if `dtype` is not provided, else casts this object to
        the specified type.

        If this is already of the correct type, no copy is performed and the
        original object is returned.

        Args:
            dtype (type or string): The desired type
            non_blocking (bool): If ``True``, and the source is in pinned memory
                and destination is on the GPU or vice versa, the copy is performed
                asynchronously with respect to the host. Otherwise, the argument
                has no effect.
            **kwargs: For compatibility, may contain the key ``async`` in place of
                the ``non_blocking`` argument. The ``async`` arg is deprecated.
        """
        return super().type(dtype=dtype, non_blocking=non_blocking, **kwargs)

    @return_tensor_wrapper
    def type(self, dtype=None, non_blocking=False, **kwargs) -> str or 'Tensor':
        """
        type(dtype=None, non_blocking=False, **kwargs) -> str or Tensor
        Returns the type if `dtype` is not provided, else casts this object to
        the specified type.

        If this is already of the correct type, no copy is performed and the
        original object is returned.

        Args:
            dtype (type or string): The desired type
            non_blocking (bool): If ``True``, and the source is in pinned memory
                and destination is on the GPU or vice versa, the copy is performed
                asynchronously with respect to the host. Otherwise, the argument
                has no effect.
            **kwargs: For compatibility, may contain the key ``async`` in place of
                the ``non_blocking`` argument. The ``async`` arg is deprecated.
        """
        return super().type(dtype=dtype, non_blocking=non_blocking, **kwargs)

    @return_tensor_wrapper
    def type_as(self, tensor) -> 'Tensor':
        """
        type_as(tensor) -> Tensor

        Returns this tensor cast to the type of the given tensor.

        This is a no-op if the tensor is already of the correct type. This is
        equivalent to ``self.type(tensor.type())``

        Args:
            tensor (Tensor): the tensor which has the desired type
        """
        return super().type_as(tensor)

    @return_tensor_wrapper
    def unfold(self, dimension, size, step) -> 'Tensor':
        """
        unfold(dimension, size, step) -> Tensor

        Returns a view of the original tensor which contains all slices of size :attr:`size` from
        :attr:`self` tensor in the dimension :attr:`dimension`.

        Step between two slices is given by :attr:`step`.

        If `sizedim` is the size of dimension :attr:`dimension` for :attr:`self`, the size of
        dimension :attr:`dimension` in the returned tensor will be
        `(sizedim - size) / step + 1`.

        An additional dimension of size :attr:`size` is appended in the returned tensor.

        Args:
            dimension (int): dimension in which unfolding happens
            size (int): the size of each slice that is unfolded
            step (int): the step between each slice

        Example::

            >>> x = torch.arange(1., 8)
            >>> x
            tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.])
            >>> x.unfold(0, 2, 1)
            tensor([[ 1.,  2.],
                    [ 2.,  3.],
                    [ 3.,  4.],
                    [ 4.,  5.],
                    [ 5.,  6.],
                    [ 6.,  7.]])
            >>> x.unfold(0, 2, 2)
            tensor([[ 1.,  2.],
                    [ 3.,  4.],
                    [ 5.,  6.]])
        """
        return super().unfold(dimension, size, step)

    @return_tensor_wrapper
    def uniform_(self, start=0, to=1) -> 'Tensor':
        """
        uniform_(from=0, to=1) -> Tensor

        Fills :attr:`self` tensor with numbers sampled from the continuous uniform
        distribution:

        .. math::
            P(x) = \dfrac{1}{\text{to} - \text{from}}
        """
        return super().uniform_(start, to=to)

    @return_tensor_wrapper
    def unsqueeze(self, dim) -> 'Tensor':
        """
        unsqueeze(input, dim) -> Tensor

        Returns a new tensor with a dimension of size one inserted at the
        specified position.

        The returned tensor shares the same underlying data with this tensor.

        A :attr:`dim` value within the range ``[-input.dim() - 1, input.dim() + 1)``
        can be used. Negative :attr:`dim` will correspond to :meth:`unsqueeze`
        applied at :attr:`dim` = ``dim + input.dim() + 1``.

        Args:
            input (Tensor): the input tensor.
            dim (int): the index at which to insert the singleton dimension

        Example::

            >>> x = torch.tensor([1, 2, 3, 4])
            >>> torch.unsqueeze(x, 0)
            tensor([[ 1,  2,  3,  4]])
            >>> torch.unsqueeze(x, 1)
            tensor([[ 1],
                    [ 2],
                    [ 3],
                    [ 4]])
        """
        return super().unsqueeze(dim)

    @return_tensor_wrapper
    def unsqueeze_(self, dim) -> 'Tensor':
        """
        unsqueeze_(dim) -> Tensor

        In-place version of :meth:`~Tensor.unsqueeze`
        """
        return super().unsqueeze_(dim)

    @return_tensor_wrapper
    def values(self) -> 'Tensor':
        """
        values() -> Tensor

        If :attr:`self` is a sparse COO tensor (i.e., with ``torch.sparse_coo`` layout),
        this returns a view of the contained values tensor. Otherwise, this throws an
        error.

        See also :meth:`Tensor.indices`.

        .. note::
          This method can only be called on a coalesced sparse tensor. See
          :meth:`Tensor.coalesce` for details.
        """
        return super().values()

    @return_tensor_wrapper
    def var(self, dim=None, unbiased=True, keepdim=False) -> 'Tensor':
        """
        var(input, unbiased=True) -> Tensor

        Returns the variance of all elements in the :attr:`input` tensor.

        If :attr:`unbiased` is ``False``, then the variance will be calculated via the
        biased estimator. Otherwise, Bessel's correction will be used.

        Args:
            input (Tensor): the input tensor.
            unbiased (bool): whether to use the unbiased estimation or not

        Example::

            >>> a = torch.randn(1, 3)
            >>> a
            tensor([[-0.3425, -1.2636, -0.4864]])
            >>> torch.var(a)
            tensor(0.2455)


        .. function:: var(input, dim, keepdim=False, unbiased=True, out=None) -> Tensor

        Returns the variance of each row of the :attr:`input` tensor in the given
        dimension :attr:`dim`.


        If :attr:`keepdim` is ``True``, the output tensor is of the same size
        as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
        Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
        output tensor having 1 (or ``len(dim)``) fewer dimension(s).


        If :attr:`unbiased` is ``False``, then the variance will be calculated via the
        biased estimator. Otherwise, Bessel's correction will be used.

        Args:
            input (Tensor): the input tensor.
            dim (int or tuple of ints): the dimension or dimensions to reduce.
            keepdim (bool): whether the output tensor has :attr:`dim` retained or not.
            unbiased (bool): whether to use the unbiased estimation or not
            out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4, 4)
            >>> a
            tensor([[-0.3567,  1.7385, -1.3042,  0.7423],
                    [ 1.3436, -0.1015, -0.9834, -0.8438],
                    [ 0.6056,  0.1089, -0.3112, -1.4085],
                    [-0.7700,  0.6074, -0.1469,  0.7777]])
            >>> torch.var(a, 1)
            tensor([ 1.7444,  1.1363,  0.7356,  0.5112])
        """
        return super().var(dim=dim, unbiased=unbiased, keepdim=keepdim)

    @return_tensor_wrapper
    def view(self, *shape) -> 'Tensor':
        """
        view(*shape) -> Tensor

        Returns a new tensor with the same data as the :attr:`self` tensor but of a
        different :attr:`shape`.

        The returned tensor shares the same data and must have the same number
        of elements, but may have a different size. For a tensor to be viewed, the new
        view size must be compatible with its original size and stride, i.e., each new
        view dimension must either be a subspace of an original dimension, or only span
        across original dimensions :math:`d, d+1, \dots, d+k` that satisfy the following
        contiguity-like condition that :math:`\forall i = d, \dots, d+k-1`,

        .. math::

          \text{stride}[i] = \text{stride}[i+1] \times \text{size}[i+1]

        Otherwise, it will not be possible to view :attr:`self` tensor as :attr:`shape`
        without copying it (e.g., via :meth:`contiguous`). When it is unclear whether a
        :meth:`view` can be performed, it is advisable to use :meth:`reshape`, which
        returns a view if the shapes are compatible, and copies (equivalent to calling
        :meth:`contiguous`) otherwise.

        Args:
            shape (torch.Size or int...): the desired size

        Example::

            >>> x = torch.randn(4, 4)
            >>> x.size()
            torch.Size([4, 4])
            >>> y = x.view(16)
            >>> y.size()
            torch.Size([16])
            >>> z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
            >>> z.size()
            torch.Size([2, 8])

            >>> a = torch.randn(1, 2, 3, 4)
            >>> a.size()
            torch.Size([1, 2, 3, 4])
            >>> b = a.transpose(1, 2)  # Swaps 2nd and 3rd dimension
            >>> b.size()
            torch.Size([1, 3, 2, 4])
            >>> c = a.view(1, 3, 2, 4)  # Does not change tensor layout in memory
            >>> c.size()
            torch.Size([1, 3, 2, 4])
            >>> torch.equal(b, c)
            False
        """
        return super().view(*shape)

    @return_tensor_wrapper
    def view_as(self, other) -> 'Tensor':
        """
        view_as(other) -> Tensor

        View this tensor as the same size as :attr:`other`.
        ``self.view_as(other)`` is equivalent to ``self.view(other.size())``.

        Please see :meth:`~Tensor.view` for more information about ``view``.

        Args:
            other (:class:`torch.Tensor`): The result tensor has the same size
                as :attr:`other`.
        """
        return super().view_as(other)

    @return_tensor_wrapper
    def where(self, condition, y) -> 'Tensor':
        """
        where(condition, x, y) -> Tensor

        Return a tensor of elements selected from either :attr:`x` or :attr:`y`, depending on :attr:`condition`.

        The operation is defined as:

        .. math::
            \text{out}_i = \begin{cases}
                \text{x}_i & \text{if } \text{condition}_i \\
                \text{y}_i & \text{otherwise} \\
            \end{cases}

        .. note::
            The tensors :attr:`condition`, :attr:`x`, :attr:`y` must be :ref:`broadcastable <broadcasting-semantics>`.

        Arguments:
            condition (BoolTensor): When True (nonzero), yield x, otherwise yield y
            x (Tensor): values selected at indices where :attr:`condition` is ``True``
            y (Tensor): values selected at indices where :attr:`condition` is ``False``

        Returns:
            Tensor: A tensor of shape equal to the broadcasted shape of :attr:`condition`, :attr:`x`, :attr:`y`

        Example::

            >>> x = torch.randn(3, 2)
            >>> y = torch.ones(3, 2)
            >>> x
            tensor([[-0.4620,  0.3139],
                    [ 0.3898, -0.7197],
                    [ 0.0478, -0.1657]])
            >>> torch.where(x > 0, x, y)
            tensor([[ 1.0000,  0.3139],
                    [ 0.3898,  1.0000],
                    [ 0.0478,  1.0000]])

        .. function:: where(condition) -> tuple of LongTensor

        ``torch.where(condition)`` is identical to
        ``torch.nonzero(condition, as_tuple=True)``.

        .. note::
            See also :func:`torch.nonzero`.
        """
        return super().where(condition, y)

    @return_tensor_wrapper
    def zero_(self) -> 'Tensor':
        """
        zero_() -> Tensor

        Fills :attr:`self` tensor with zeros.
        """
        return super().zero_()

    @return_tensor_wrapper
    def as_strided_(self, *args, **kwargs):
        return super().as_strided_(*args, **kwargs)

    @return_tensor_wrapper
    def clamp_max(self, *args, **kwargs):
        return super().clamp_max(*args, **kwargs)

    @return_tensor_wrapper
    def clamp_max_(self, *args, **kwargs):
        return super().clamp_max_(*args, **kwargs)

    @return_tensor_wrapper
    def clamp_min(self, *args, **kwargs):
        return super().clamp_min(*args, **kwargs)

    @return_tensor_wrapper
    def clamp_min_(self, *args, **kwargs):
        return super().clamp_min_(*args, **kwargs)

    @return_tensor_wrapper
    def coalesce(self, *args, **kwargs):
        return super().coalesce(*args, **kwargs)

    @return_tensor_wrapper
    def is_coalesced(self, *args, **kwargs):
        return super().is_coalesced(*args, **kwargs)

    @return_tensor_wrapper
    def is_distributed(self, *args, **kwargs):
        return super().is_distributed(*args, **kwargs)

    @return_tensor_wrapper
    def is_nonzero(self, *args, **kwargs):
        return super().is_nonzero(*args, **kwargs)

    @return_tensor_wrapper
    def is_same_size(self, *args, **kwargs):
        return super().is_same_size(*args, **kwargs)

    @return_tensor_wrapper
    def log_softmax(self, *args, **kwargs):
        return super().log_softmax(*args, **kwargs)

    @return_tensor_wrapper
    def map2_(self, *args, **kwargs):
        return super().map2_(*args, **kwargs)

    @return_tensor_wrapper
    def new(self, *args, **kwargs):
        return super().new(*args, **kwargs)

    @return_tensor_wrapper
    def prelu(self, *args, **kwargs):
        return super().prelu(*args, **kwargs)

    @return_tensor_wrapper
    def reinforce(self, *args, **kwargs):
        return super().reinforce(*args, **kwargs)

    @return_tensor_wrapper
    def relu(self, *args, **kwargs):
        return super().relu(*args, **kwargs)

    @return_tensor_wrapper
    def relu_(self, *args, **kwargs):
        return super().relu_(*args, **kwargs)

    @return_tensor_wrapper
    def resize(self, *args, **kwargs):
        return super().resize(*args, **kwargs)

    @return_tensor_wrapper
    def resize_as(self, *args, **kwargs):
        return super().resize_as(*args, **kwargs)

    @return_tensor_wrapper
    def smm(self, *args, **kwargs):
        return super().smm(*args, **kwargs)

    @return_tensor_wrapper
    def softmax(self, *args, **kwargs):
        return super().softmax(*args, **kwargs)

    @return_tensor_wrapper
    def sparse_resize_(self, *args, **kwargs):
        return super().sparse_resize_(*args, **kwargs)

    @return_tensor_wrapper
    def sparse_resize_and_clear_(self, *args, **kwargs):
        return super().sparse_resize_and_clear_(*args, **kwargs)

    @return_tensor_wrapper
    def split_with_sizes(self, *args, **kwargs):
        return super().split_with_sizes(*args, **kwargs)

    @return_tensor_wrapper
    def sspaddmm(self, *args, **kwargs):
        return super().sspaddmm(*args, **kwargs)

    @return_tensor_wrapper
    def to_dense(self, *args, **kwargs):
        return super().to_dense(*args, **kwargs)

    @return_tensor_wrapper
    def __abs__(self, *args, **kwargs):
        return super().__abs__(*args, **kwargs)

    @return_tensor_wrapper
    def __add__(self, other):
        # return self.__op__('__add__', *args, **kwargs)
       if isinstance(other, torch.Tensor):
           other = Tensor(other)
           if self.dim() == other.dim():
               return super().__add__(other)
           elif self.dim() < other.dim():
               return self.expand_as(other).__add__(other)
           else:
               return super().__add__(other.expand_as(self))
       return super().__add__(other)

    @return_tensor_wrapper
    def __and__(self, other):
        # return self.__op__('__add__', *args, **kwargs)
        return super().__and__(other)

    @return_tensor_wrapper
    def __array__(self, *args, **kwargs):
        return super().__array__(*args, **kwargs)

    @return_tensor_wrapper
    def __array_wrap__(self, *args, **kwargs):
        return super().__array_wrap__(*args, **kwargs)

    @return_tensor_wrapper
    def __bool__(self, *args, **kwargs):
        return super().__bool__(*args, **kwargs)

    @return_tensor_wrapper
    def __class__(self, *args, **kwargs):
        return super().__class__(*args, **kwargs)

    @return_tensor_wrapper
    def __contains__(self, *args, **kwargs):
        return super().__contains__(*args, **kwargs)

    @return_tensor_wrapper
    def __deepcopy__(self, *args, **kwargs):
        return super().__deepcopy__(*args, **kwargs)

    @return_tensor_wrapper
    def __delattr__(self, *args, **kwargs):
        return super().__delattr__(*args, **kwargs)

    @return_tensor_wrapper
    def __delitem__(self, *args, **kwargs):
        return super().__delitem__(*args, **kwargs)

    def __dir__(self, *args, **kwargs):
        result = dir(torch.Tensor)
        result.remove("volatile")
        result.remove("__cuda_array_interface__")
        result = result + ['get_default_tensor_type', 'batch_dimension', '__grad_fn', 'batch_size']
        return result

    @return_tensor_wrapper
    def __div__(self, other):
        # return self.__op__('__div__', *args, **kwargs)
        return super().__div__(other)

    @return_tensor_wrapper
    def __eq__(self, other):
        # return self.__op__('__eq__', *args, **kwargs)
        return super().__eq__(other)

    @return_tensor_wrapper
    def __float__(self, *args, **kwargs):
        return super().__float__(*args, **kwargs)

    @return_tensor_wrapper
    def __floordiv__(self, other):
        # return self.__op__('__floordiv__', *args, **kwargs)
        return super().__floordiv__(other)

    # @return_tensor_wrapper
    def __format__(self, *args, **kwargs):
        return super().__format__(*args, **kwargs)

    @return_tensor_wrapper
    def __ge__(self, other):
        # return self.__op__('__ge__', *args, **kwargs)
        return super().__ge__(other)

    def __getattribute__(self, *args, **kwargs):
        return super().__getattribute__(*args, **kwargs)

    @return_tensor_wrapper
    def __getitem__(self, *args, **kwargs):
        return super().__getitem__(*args, **kwargs)

    @return_tensor_wrapper
    def __gt__(self, other):
        # return self.__op__('__gt__', *args, **kwargs)
        return super().__gt__(other)

    @return_tensor_wrapper
    def __hash__(self, *args, **kwargs):
        return super().__hash__(*args, **kwargs)

    @return_tensor_wrapper
    def __iadd__(self, other):
        # return self.__op__('__iadd__', *args, **kwargs)
        return super().__iadd__(other)

    @return_tensor_wrapper
    def __iand__(self, other):
        # return self.__op__('__iand__', *args, **kwargs)
        return super().__iand__(other)

    @return_tensor_wrapper
    def __idiv__(self, other):
        # return self.__op__('__idiv__', *args, **kwargs)
        return super().__idiv__(other)

    @return_tensor_wrapper
    def __ifloordiv__(self, other):
        # return self.__op__('__ifloordiv__', *args, **kwargs)
        return super().__ifloordiv__(other)

    @return_tensor_wrapper
    def __ilshift__(self, *args, **kwargs):
        return super().__ilshift__(*args, **kwargs)

    @return_tensor_wrapper
    def __imul__(self, other):
        # return self.__op__('__imul__', *args, **kwargs)
        return super().__imul__(other)

    @return_tensor_wrapper
    def __index__(self, *args, **kwargs):
        return super().__index__(*args, **kwargs)

    # @return_tensor_wrapper
    # def __init_subclass__(self, *args, **kwargs):
    #     return super().__init_subclass__(*args, **kwargs)

    @return_tensor_wrapper
    def __int__(self, *args, **kwargs):
        return super().__int__(*args, **kwargs)

    @return_tensor_wrapper
    def __invert__(self, *args, **kwargs):
        return super().__invert__(*args, **kwargs)

    @return_tensor_wrapper
    def __ior__(self, other):
        # return self.__op__('__ior__', *args, **kwargs)
        return super().__ior__(other)

    @return_tensor_wrapper
    def __ipow__(self, other):
        # return self.__op__('__ipow__', *args, **kwargs)
        return super().__ipow__(other)

    @return_tensor_wrapper
    def __irshift__(self, *args, **kwargs):
        return super().__irshift__(*args, **kwargs)

    @return_tensor_wrapper
    def __isub__(self, other):
        # return self.__op__('__isub__', *args, **kwargs)
        return super().__isub__(other)

    @return_tensor_wrapper
    def __iter__(self, *args, **kwargs):
        return super().__iter__(*args, **kwargs)

    @return_tensor_wrapper
    def __itruediv__(self, other):
        # return self.__op__('__itruediv__', *args, **kwargs)
        return super().__itruediv__(other)

    @return_tensor_wrapper
    def __ixor__(self, other):
        # return self.__op__('__ixor__', *args, **kwargs)
        return super().__ixor__(other)

    @return_tensor_wrapper
    def __le__(self, other):
        # return self.__op__('__ile__', *args, **kwargs)
        return super().__le__(other)

    @return_tensor_wrapper
    def __len__(self, *args, **kwargs):
        return super().__len__(*args, **kwargs)

    @return_tensor_wrapper
    def __long__(self, *args, **kwargs):
        return super().__long__(*args, **kwargs)

    @return_tensor_wrapper
    def __lshift__(self, *args, **kwargs):
        return super().__lshift__(*args, **kwargs)

    @return_tensor_wrapper
    def __lt__(self, other):
        # return self.__op__('__lt__', *args, **kwargs)
        return super().__lt__(other)

    @return_tensor_wrapper
    def __matmul__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], torch.Tensor):
            other = Tensor(args[0])
            if self.shape[-2:].has_special or other.shape[-2:].has_special:
                raise RuntimeError("'matmul' cannot operate for special dimensions. Please make sure that the last two dimension of both tensors are not batch/channel dimensions. ")
            new_size = self.shape[:-2] @ other.shape[:-2]
            return (self.expand_to(new_size + self.shape[-2:])) @ (other.expand_to(new_size + other.shape[-2:]))
        return super().__matmul__(*args, **kwargs)

    @return_tensor_wrapper
    def __mod__(self, other):
        # return self.__op__('__mod__', *args, **kwargs)
        return super().__mod__(other)

    @return_tensor_wrapper
    def __mul__(self, other):
        # return self.__op__('__mul__', *args, **kwargs)
        return super().__mul__(other)

    @return_tensor_wrapper
    def __ne__(self, other):
        # return self.__op__('__ne__', *args, **kwargs)
        return super().__ne__(other)

    @return_tensor_wrapper
    def __neg__(self, *args, **kwargs):
        return super().__neg__(*args, **kwargs)

    @return_tensor_wrapper
    def __nonzero__(self, *args, **kwargs):
        return super().__nonzero__(*args, **kwargs)

    # @return_tensor_wrapper
    # def __op__(self, opname, *args, **kwargs):
    #     if len(args) == 1 and isinstance(args[0], torch.Tensor):
    #         other = Tensor(args[0])
    #         new_size = self.shape @ other.shape
    #         return getattr(super(Tensor, self.expand_to(new_size)), opname)(other.expand_to(new_size))
    #     return getattr(super(), opname)(*args, **kwargs)

    @return_tensor_wrapper
    def __or__(self, other):
        # return self.__op__('__or__', *args, **kwargs)
        return super().__or__(other)

    @return_tensor_wrapper
    def __pow__(self, other):
        # return self.__op__('__pow__', *args, **kwargs)
        return super().__pow__(other)

    @return_tensor_wrapper
    def __radd__(self, other):
        # return self.__op__('__radd__', *args, **kwargs)
        return super().__radd__(other)

    @return_tensor_wrapper
    def __rdiv__(self, other):
        # return self.__op__('__rdiv__', *args, **kwargs)
        return super().__rdiv__(other)

    @return_tensor_wrapper
    def __reduce__(self, *args, **kwargs):
        return super().__reduce__(*args, **kwargs)

    @return_tensor_wrapper
    def __reduce_ex__(self, *args, **kwargs):
        return super().__reduce_ex__(*args, **kwargs)

    # @return_tensor_wrapper
    def __repr__(self, *args, **kwargs):
        string = self.tensor().__repr__(*args, **kwargs)
        if 'shape=' not in string:
            string = string.rstrip(')') + f', shape={self.shape})'
        return string.replace("tensor", "Tensor")

    @return_tensor_wrapper
    def __reversed__(self, *args, **kwargs):
        return super().__reversed__(*args, **kwargs)

    @return_tensor_wrapper
    def __rfloordiv__(self, other):
        # return self.__op__('__rfloordiv__', *args, **kwargs)
        return super().__rfloordiv__(other)

    @return_tensor_wrapper
    def __rmul__(self, other):
        # return self.__op__('__rmul__', *args, **kwargs)
        return super().__rmul__(other)

    @return_tensor_wrapper
    def __rpow__(self, other):
        # return self.__op__('__rpow__', *args, **kwargs)
        return super().__rpow__(other)

    @return_tensor_wrapper
    def __rshift__(self, *args, **kwargs):
        return super().__rshift__(*args, **kwargs)

    @return_tensor_wrapper
    def __rsub__(self, other):
        # return self.__op__('__rsub__', *args, **kwargs)
        return super().__rsub__(other)

    @return_tensor_wrapper
    def __rtruediv__(self, other):
        # return self.__op__('__rtruediv__', *args, **kwargs)
        return super().__rtruediv__(other)

    # @return_tensor_wrapper
    # def __setattr__(self, *args, **kwargs):
    #     return super().__setattr__(*args, **kwargs)

    @return_tensor_wrapper
    def __setitem__(self, *args, **kwargs):
        return super().__setitem__(*args, **kwargs)

    @return_tensor_wrapper
    def __setstate__(self, *args, **kwargs):
        return super().__setstate__(*args, **kwargs)

    @return_tensor_wrapper
    def __sizeof__(self, *args, **kwargs):
        return super().__sizeof__(*args, **kwargs)

    # @return_tensor_wrapper
    def __str__(self, *args, **kwargs):
        string = self.tensor().__str__(*args, **kwargs)
        if 'shape=' not in string:
            string = string.rstrip(')') + f', shape={self.shape})'
        return string.replace("tensor", "Tensor")

    @return_tensor_wrapper
    def __sub__(self, other):
        # return self.__op__('__sub__', *args, **kwargs)
        return super().__sub__(other)

    @return_tensor_wrapper
    def __subclasshook__(self, *args, **kwargs):
        return super().__subclasshook__(*args, **kwargs)

    @return_tensor_wrapper
    def __truediv__(self, other):
        # return self.__op__('__truediv__', *args, **kwargs)
        return super().__truediv__(other)

    @return_tensor_wrapper
    def __xor__(self, other):
        # return self.__op__('__xor__', *args, **kwargs)
        return super().__xor__(other)

    @return_tensor_wrapper
    def _coalesced_(self, *args, **kwargs):
        return super()._coalesced_(*args, **kwargs)

    @return_tensor_wrapper
    def _dimI(self, *args, **kwargs):
        return super()._dimI(*args, **kwargs)

    @return_tensor_wrapper
    def _dimV(self, *args, **kwargs):
        return super()._dimV(*args, **kwargs)

    @return_tensor_wrapper
    def _indices(self, *args, **kwargs):
        return super()._indices(*args, **kwargs)

    @return_tensor_wrapper
    def _is_view(self, *args, **kwargs):
        return super()._is_view(*args, **kwargs)

    # @return_tensor_wrapper
    # def _make_subclass(self, *args, **kwargs):
    #     return super()._make_subclass(*args, **kwargs)

    @return_tensor_wrapper
    def _nnz(self, *args, **kwargs):
        return super()._nnz(*args, **kwargs)

    @return_tensor_wrapper
    def _update_names(self, *args, **kwargs):
        return super()._update_names(*args, **kwargs)

    @return_tensor_wrapper
    def _values(self, *args, **kwargs):
        return super()._values(*args, **kwargs)

    @return_tensor_wrapper
    def norm(self, *args, **kwargs):
        return super().norm(*args, **kwargs)

    @property
    def grad_fn(self):
        return touch(lambda: self.__grad_fn)

__all__.extend(["zeros", "ones", "zeros_like", "ones_like", "tensor", "t", "eye"])

# @overload
# def zeros(*size: SizeRep.itemtypes, **kwargs):
#     return zeros(size, **kwargs)

# @overload
# def zeros(tensor: Array.Torch | Tensor, **kwargs):
#     out = Tensor(torch.zeros_like(tensor, **kwargs), **kwargs)
#     out.batch_dimension = tensor.batch_dimension
#     out.channel_dimension = tensor.channel_dimension
#     return out

# @overload
# def zeros(size: SizeRep | Size, **kwargs):
#     size = Size(size)
#     out = Tensor(torch.zeros(size, **kwargs), **kwargs)
#     out.batch_dimension = size.batch_dimension
#     out.channel_dimension = size.channel_dimension
#     return out

# @overload
# def zeros__default__(*args, **kwargs):
#     return Tensor(torch.zeros(*args, **kwargs), **kwargs)

# @overload
# def zeros_like(tensor: Array.Torch, **kwargs):
#     return zeros(tensor, **kwargs)

# @overload
# def ones(*size: SizeRep.itemtypes, **kwargs):
#     return ones(size, **kwargs)

# @overload
# def ones(tensor: Array.Torch | Tensor, **kwargs):
#     out = Tensor(torch.ones_like(tensor, **kwargs), **kwargs)
#     out.batch_dimension = tensor.batch_dimension
#     out.channel_dimension = tensor.channel_dimension
#     return out

# @overload
# def ones(size: SizeRep | Size, **kwargs):
#     size = Size(size)
#     out = Tensor(torch.ones(size, **kwargs), **kwargs)
#     out.batch_dimension = size.batch_dimension
#     out.channel_dimension = size.channel_dimension
#     return out

# @overload
# def ones__default__(*args, **kwargs):
#     return Tensor(torch.ones(*args, **kwargs), **kwargs)

# @overload
# def ones_as(tensor: Array.Torch, **kwargs):
#     return ones(tensor, **kwargs)

# @overload
# def eye(*size: SizeRep.itemtypes):
#     return eye(size)

# @overload
# def eye(size: SizeRep | Size):
#     size = Size(size)
#     if size.nspace < 1: raise TypeError("Empty size not valid for 'eye'. ")
#     if size.nspace == 1: size = size + (size.space[0],)
#     if size.nspace > 2: raise TypeError("No more than 2-D is allowed for 'eye'. ")
#     n = builtins.min(*size.space)
#     s = [slice(None)] * size.ndim
#     for i in builtins.range(size.ndim):
#         if i not in size.special:
#             s[i] = torch.arange(n)
#     out = zeros(size)
#     out[tuple(s)] = 1
#     return out

# @overload
# def t(tensor: Array.Torch):
#     return Tensor(tensor).T

# tensor = Tensor
@return_tensor_wrapper(False)
def tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False):
    if device is None and _auto_device is True:
        device = Device
    return torch.tensor(data, dtype=dtype, device=device, requires_grad=requires_grad, pin_memory=pin_memory)

template = "@return_tensor_wrapper\ndef {key}(*args, **kwargs): return torch.{key}(*args, **kwargs)"
for key in dir(torch):
    if key.startswith("_"):
        continue
    if inspect.isclass(eval("torch.{}".format(key))):
        continue
    if isinstance(eval("torch.{}".format(key)), torch.dtype):
        exec("{} = torch.{}".format(key, key))
        __all__.append(key)
    if key not in globals() and key not in {"optim", "random"}:
        exec(template.format(key=key))
        __all__.append(key)

no_grad = torch.no_grad
__all__.append("no_grad")
optim = torch.optim
__all__.append("optim")
random = torch.random
__all__.append("random")

    # @property
    # def dtype(self):
    #     return self._dtype

# import torch
# x = torch.Tensor([1, 2, 3])

# from pyctlib import touch, vector

# def getdoc(func):
#     assert func in x.__dir__()
#     class_func = touch(lambda: x.__getattribute__(func).__doc__)
#     if touch(lambda:class_func.split("\n")[-2].startswith("See :func:")):
#         new_func = re.match(r"See :func:`([\w.]*)`", class_func.split("\n")[-2]).group(1)
#         torch_func = eval(new_func + ".__doc__")
#     elif touch(lambda:class_func.split("\n")[-1].startswith("See :func:")):
#         new_func = re.match(r"See :func:`([\w.]*)`", class_func.split("\n")[-1]).group(1)
#         torch_func = eval(new_func + ".__doc__")
#     else:
#         torch_func = None
#     return torch_func, class_func, func

# import re

# def getgooddoc(func):
#     torch_func, class_func, func = getdoc(func)
#     if not torch_func and not class_func:
#         return "", -1
#     if class_func:
#         if torch_func:
#             return torch_func, 1
#         else:
#             return class_func, 0
#     return torch_func, 2

# def getfuncdef(func):
#     torch_func, class_func, func = getdoc(func)
#     for index in [0, 1, -1]:
#         if touch(lambda: class_func.split("\n")[index].strip().startswith(func)):
#             return "def " + class_func.split("\n")[index].strip()
#     return None

# def generate_def(func):
#     funcdef = getfuncdef(func)
#     parameters = vector(re.split(r"[, ]", re.match(r".*\((.*)\)", funcdef).group(1))).filter(lambda x: x and x != "*")
#     basic = parameters.filter(lambda x: "=" not in x and "*" not in x)
#     args = parameters.filter(lambda x: x.startswith("*") and not x.startswith("**"))
#     kwargs = parameters.filter(lambda x: "**" in x)
#     keyword_args = parameters.filter(lambda x: "=" in x)
#     return_parameter = ", ".join(basic + args + keyword_args.map(lambda x: x.split("=")[0]).map(lambda x: x + "=" + x) + kwargs)
#     return (funcdef.split("(")[0] + "(self" + (", " if parameters else "") + funcdef.split("(")[1].replace(" Tensor", " 'Tensor'").replace("LongTensor", "'LongTensor'") + ":", "    return super().%s(%s)" % (func, return_parameter))

# def simple_def(func):
#     return ("def %s(self, *args, **kwargs):" % func, "    return super().%s(*args, **kwargs)" % func)

# def returnTensor(func):
#     deff = getfuncdef(func)
#     if deff and deff.endswith("Tensor"):
#         return True
#     return False

# p = vector(x.__dir__()).filter(lambda t: touch(lambda: callable(x.__getattribute__(t)))).filter(lambda x: x[0] != "_")
# p_ = vector(x.__dir__()).filter(lambda t: touch(lambda: callable(x.__getattribute__(t)))).filter(lambda x: x[0] == "_")
# error_p = p.filter(lambda x: getgooddoc(x)[1] == -1)
# p = p.filter(lambda x: getgooddoc(x)[1] != -1)
# rt_p = p.filter(returnTensor)

# from pyctlib.filemanager import *
# f = file(path(".").abs() / "Torch.py")

# content = vector()
# for (d, r, doc) in rt_p.map(generate_def) * rt_p.map(lambda x: getgooddoc(x)[0]):
#     content.append("@return_tensor_wrapper")
#     content.append(d)
#     content.append('    """')
#     doc = vector(doc.strip("\n").split("\n"))
#     content = content + doc.map(lambda x: "    " + x).map(lambda x: x.rstrip())
#     content.append('    """')
#     content.append("    " + r.strip())
#     content.append("")

# f.writelines(content)


# from pyctlib.filemanager import *
# f = file(path(".").abs() / "error_Torch.py")

# content = vector()
# for (d, r) in p_.map(simple_def):
#     content.append("@return_tensor_wrapper")
#     content.append(d)
#     content.append("    " + r.strip())
#     content.append("")

# f.writelines(content)