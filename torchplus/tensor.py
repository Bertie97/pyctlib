#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package torchplus
##############################

__all__ = """
    Device
    DeviceCPU
    Tensor
    Size
    set_autodevice
    unset_autodevice
    is_autodevice
""".split()

try:
    import torch
except ImportError:
    raise ImportError("'pyctlib.torchplus' cannot be used without dependency 'torch'.")
import typing
import inspect
import builtins
import torchplus as tp
from .tensorfunc import __all__ as tf_list
# from pyoverload import overload, override, Tuple, List, Set, params, null, Array, isarray, isoftype, isofsubclass, isint, isdtype, isitertype, isclassmethod
from pyoverload import *
from pyctlib import raw_function, return_type_wrapper, touch
#from pyctlib.visual.debugger import profile
from functools import wraps
from types import GeneratorType, MethodWrapperType
from collections import OrderedDict
from .torch_namespace import *
from .device import AutoDevice as Device, DeviceCPU

"""
from torchplus import Tensor
import torchplus as tp
"""

# Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_auto_device = True

def set_autodevice(flag=True):
    global _auto_device
    _auto_device = flag

def unset_autodevice():
    global _auto_device
    _auto_device = False

def is_autodevice():
    global _auto_device
    return _auto_device

INT = builtins.int
MIN = builtins.min
MAX = builtins.max
ANY = builtins.any
ALL = builtins.all
RANGE = builtins.range
FLOAT = builtins.float
NUM = (INT, FLOAT)

def kwget(kwargs, key, default):
    if kwargs is None: return default
    else: return kwargs.get(key, default)

class Size(tuple):

    NegSizeError = TypeError("Size cannot have negative values except -1 indicating arbitrary number. ")

    def __new__(cls, *args, **kwargs):

        if len(args) == 1:
            arg = args[0]
            if hasattr(arg, 'shape'): arg = arg.shape
            if isinstance(arg, (INT, set)): pass
            elif isinstance(arg, tuple): args = arg
            elif isinstance(arg, GeneratorType):
                self = super().__new__(cls, arg)
                self.ndim = len(tuple(arg))
                return self.set_special_(kwargs.get('batch_dim', None), kwargs.get('channel_dim', None))
            elif isinstance(arg, list):
                l = len(arg)
                if l != 1:
                    self = super().__new__(cls, arg)
                    self.ndim = l
                    return self.set_special_(kwargs.get('batch_dim', None), kwargs.get('channel_dim', None))
            elif isinstance(arg, Size):
                self = super().__new__(cls, arg.tuple())
                self.ndim = arg.ndim
                self._special = arg._special
                self._batch_first = arg._batch_first
                return self.set_special_(kwargs.get('batch_dim', None), kwargs.get('channel_dim', None))
            else:
                raise TypeError("'Size' object only takes tensors, tuples, lists and generators as initialization. ")

        # args is now a tuple of * or [*], {*} where * is an integer.
        ndim = len(args)
        batch_dim = ndim
        channel_dim = ndim
        raw_args = []
        for i, a in enumerate(args):

            if isinstance(a, list):
                if batch_dim == ndim:
                    batch_dim = i
                else:
                    raise TypeError("Only one batch dimension is allowed.")
                raw_args.append(a[0])

            elif isinstance(a, set):
                if channel_dim == ndim:
                    channel_dim = i
                else:
                    raise TypeError("Only one channel dimension is allowed.")
                raw_args.append(a.pop())

            else: raw_args.append(a)

        self = super().__new__(cls, raw_args)
        self.ndim = ndim
        return self.set_special_(kwargs.get('batch_dim', batch_dim), kwargs.get('channel_dim', channel_dim))

    def tuple(self): return super().__new__(tuple, self)

    @property
    def batch_dimension(self): return self._special[0 if self._batch_first else 1]

    @batch_dimension.setter
    def batch_dimension(self, batch_dim):
        self.set_special_(batch_dim, None)

    def batch_dimension_(self, value):
        self.batch_dimension = value
        return self

    @property
    def batch_size(self):
        batch_dim = self.batch_dimension
        if batch_dim == self.ndim:
            raise ValueError("There is no batch dimension provided. ")
        return self[batch_dim]

    @property
    def channel_dimension(self): return self._special[1 if self._batch_first else 0]

    @channel_dimension.setter
    def channel_dimension(self, channel_dim):
        self.set_special_(None, channel_dim)

    def channel_dimension_(self, value):
        self.channel_dimension = value
        return self

    @property
    def channel_size(self):
        channel_dim = self.channel_dimension
        if channel_dim is None:
            raise ValueError("There is no channel dimension provided. ")
        return self[channel_dim]

    @property
    def special(self): return [x for x in self._special if x < self.ndim]

    def special_from_(self, other=None):

        if isinstance(other, Size) or isinstance(other, Tensor) and other.init:
            self._special = [self.ndim if x == other.ndim else x for x in other._special]
            self._batch_first = other._batch_first
        else:
            self._special = [self.ndim, self.ndim]
            self._batch_first = True

        return self

    def add_special_from_(self, other=None):

        if isinstance(other, Size):

            doit = False
            batch_dim = None
            channel_dim = None

            if self.batch_dimension == self.ndim:
                batch_dim = other.batch_dimension
                if batch_dim != self.ndim:
                    doit = True
            if self.channel_dimension == self.ndim:
                channel_dim = other.channel_dimension
                if channel_dim != self.ndim:
                    doit = True
            if doit: self.set_special_(batch_dim, channel_dim)

        return self

    def set_special_(self, batch_dim=None, channel_dim=None):

        if batch_dim is not None:
            if not isinstance(batch_dim, INT): batch_dim = self.ndim
            if batch_dim < 0: batch_dim = batch_dim + self.ndim
            if not 0 <= batch_dim <= self.ndim:
                raise TypeError(f"batch_dimension should be a dimension index which is smaller than {self.ndim}. ")
        else: batch_dim = self.ndim

        if channel_dim is not None:
            if not isinstance(channel_dim, INT): channel_dim = self.ndim
            if channel_dim < 0: channel_dim = channel_dim + self.ndim
            if not 0 <= channel_dim <= self.ndim:
                raise TypeError(f"channel_dimension should be a dimension index which is smaller than {self.ndim}. ")
        else: channel_dim = self.ndim

        if batch_dim is None and channel_dim is None: return self

        if batch_dim < channel_dim:
            self._batch_first = True
            self._special = [batch_dim, channel_dim]
        elif channel_dim < batch_dim:
            self._batch_first = False
            self._special = [channel_dim, batch_dim]
        elif batch_dim < self.ndim:
            raise ValueError(f"special dimensions can not be the same: {batch_dim} and {channel_dim}. ")
        else:
            self._batch_first = True
            self._special = [channel_dim, channel_dim]

        return self

    def insert_special_to_tuple(self, target, value):
        s = self._special
        t = tuple(target)
        res = t[:s[0]]
        if s[0] < self.ndim: res += (value,)
        else: return Size(res).special_from_(self)
        res += t[s[0]:s[1]-1]
        if s[1] < self.ndim: res += (value,)
        else: return Size(res).special_from_(self)
        res += t[s[1]-1:]
        return Size(res).special_from_(self)

    def replace_special(self, value):
        s = self._special
        t = tuple(self)
        res = t[:s[0]]
        if s[0] < self.ndim: res += (value,)
        else: return Size(res).special_from_(self)
        res += t[s[0]+1:s[1]]
        if s[1] < self.ndim: res += (value,)
        else: return Size(res).special_from_(self)
        res += t[s[1]+1:]
        return Size(res).special_from_(self)

    @property
    def space(self):
        s = self._special
        t = tuple(self)
        return t[:s[0]] + t[s[0]+1:s[1]] + t[s[1]+1:]

    @property
    def nele(self):
        p = 1
        for i in self:
            if i == -1: return -1
            p *= i
        return p

    def __len__(self): return self.ndim

    @property
    def n_dim(self): return self.ndim

    @property
    def nspace(self): return self.ndim - (self._special[0] < self.ndim) - (self._special[1] < self.ndim)

    @property
    def nspecial(self): return (self._special[0] < self.ndim) + (self._special[1] < self.ndim)

    nbatch = batch_size
    nchannel = channel_size
    n_space = nspace
    n_special = nspecial

    @property
    def has_batch(self): return self._special[0 if self._batch_first else 1] < self.ndim

    @property
    def has_channel(self): return self._special[1 if self._batch_first else 0] < self.ndim

    @property
    def has_special(self): return self._special != [self.ndim, self.ndim]

    def remove_special_(self):
        self._special = [self.ndim, self.ndim]
        self._batch_first = True
        return self

    def copy(self): return Size(self)

    def __add__(self, other: [tuple, 'Size']):
        if not isinstance(other, tuple):
            raise TypeError("Only Size+tuple is available for Size as a python tuple, "
                            "please use size << 1 to increase the size numerically. ")
        if not isinstance(other, Size):
            return Size(tuple(self) + other).special_from_(self)

        ndim = self.ndim + other.ndim
        batch_dim = channel_dim = ndim
        if self.has_batch: batch_dim = self.batch_dimension
        if other.has_batch:
            if batch_dim < ndim: raise TypeError("Batch dimension conflict in addition. ")
            batch_dim = self.ndim + other.batch_dimension
        if self.has_channel: channel_dim = self.channel_dimension
        if other.has_channel:
            if channel_dim < ndim: raise TypeError("Channel dimension conflict in addition. ")
            channel_dim = self.ndim + other.channel_dimension

        return Size(tuple(self) + tuple(other), batch_dim=batch_dim, channel_dim=channel_dim)

    def __radd__(self, other: [tuple, 'Size']):
        if not isinstance(other, tuple):
            raise TypeError("Only Size+tuple is available for Size as a python tuple, "
                            "please use size << 1 to increase the size numerically. ")
        if not isinstance(other, Size):
            other = Size(other)
        return other + self
    __iadd__ = __add__

    def __mul__(self, value: INT):
        if not isinstance(value, INT):
            raise TypeError("Only Size*int is available for Size as a python tuple, "
                            "please use size ** 1 to do the multiply numerically. ")
        return Size(tuple(self) * value).special_from_(self)
    __imul__ = __rmul__ = __mul__

    @staticmethod
    def __op__(a: [INT, tuple, 'Size'], b: [INT, tuple, 'Size'], *, op):

        def getvalue(x, y, r):
            if x == -2: return y
            if y == -2: return x
            if x == -1 or y == -1: return -1
            if r: z = op(y, x)
            else: z = op(x, y)
            if z < 0: raise Size.NegSizeError
            return INT(z)

        LengthError = TypeError("Operation only apply for Sizes of the same length, "
                                "please consider to identify the batch/channel dimension or use + to concatenate. ")

        rev = False
        if isinstance(a, NUM):
            a, b = b, a
            rev = True
        if isinstance(a, NUM): raise TypeError("'__op__' for Size do not take two numbers. ")
        if isinstance(b, NUM): return Size(getvalue(x, b, rev) for x in a).special_from_(a)
        if isinstance(a, tuple):
            a, b = b, a
            rev = True
        if isinstance(a, tuple):
            if len(a) == len(b): return Size(getvalue(x, y, rev) for x, y in zip(a, b))
            raise LengthError
        if isinstance(b, tuple):
            if len(a) == len(b): return Size(getvalue(x, y, rev) for x, y in zip(a, b)).special_from_(a)
            if a.space == len(b): b = a.insert_special_to_tuple(b, -2)
        
        # Now deal with two Sizes
        if a.ndim == b.nspace:
            a, b = b, a
            rev = True
        if a.ndim == b.nspace:
            if len(a) == len(b): return Size(getvalue(x, y, rev) for x, y in zip(a, b))
            raise LengthError
        if a.nspace == b.ndim: b = a.insert_special_to_tuple(b, -2)
        if a.ndim == b.ndim:
            if a._special == b._special: return Size(getvalue(x, y, rev) for x, y in zip(a, b)).add_special_from_(a).add_special_from_(b)
            raise TypeError("Only Sizes with same batch/channel dimensions can be operated. ")
        else: raise LengthError

    def __lshift__(self, other: [INT, tuple, 'Size']): return self.__op__(self, other, op=lambda x, y: x + y)
    __ilshift__ = __rlshift__ = __lshift__

    def __rshift__(self, other: [INT, tuple, 'Size']): return Size.__op__(self, other, op=lambda x, y: x - y)

    def __rrshift__(self, other: [INT, tuple, 'Size']): return Size.__op__(other, self, op=lambda x, y: x - y)
    __irshift__ = __rshift__

    def __pow__(self, other: [INT, tuple, 'Size']): return Size.__op__(self, other, op=lambda x, y: x * y)
    __ipow__ = __rpow__ = __pow__

    def __floordiv__(self, other: [INT, tuple, 'Size']): return Size.__op__(self, other, op=lambda x, y: x // y)

    def __rfloordiv__(self, other: [INT, tuple, 'Size']): return Size.__op__(other, self, op=lambda x, y: x // y)
    __ifloordiv__ = __floordiv__

    def __xor__(self, other: [tuple, 'Size']):

        if not isinstance(other, Size): other = Size(other)

        if self.ndim == 0: return (Size(1) * other.ndim).special_from_(other), other
        elif other.ndim == 0: return self, (Size(1) * self.ndim).special_from_(self)

        if self.ndim == other.ndim:
            if self.nspecial == 0: return self.special_from_(other), other
            elif other.nspecial == 0: return self, other.special_from_(self)

        if self.nspace == other.nspace:
            if self.nspecial == 0: self = other.insert_special_to_tuple(self, 1)
            elif other.nspecial == 0: other = self.insert_special_to_tuple(other, 1)

        if self.nspecial == other.nspecial:
            if self._batch_first != other._batch_first:
                raise RuntimeError(f"Sizes {self} and {other} with opposite batch-channel order cannot be expand together.")

            self_len1 = self._special[0]
            other_len1 = other._special[0]
            self_len2 = self._special[1] - self._special[0] - 1
            other_len2 = other._special[1] - other._special[0] - 1
            self_len3 = self.ndim - self._special[1] - 1
            other_len3 = other.ndim - other._special[1] - 1
            len1 = MAX(self_len1, other_len1)
            len2 = MAX(self_len2, other_len2)
            len3 = MAX(self_len3, other_len3)
            tup_self = self.tuple()
            tup_other = other.tuple()
            exp_self = (
                (1,) * (len1 - self_len1) + tup_self[:self_len1+1] + 
                (1,) * (len2 - self_len2) + tup_self[self_len1+1:self_len1+self_len2+2] + 
                (1,) * (len3 - self_len3) + tup_self[self_len1+self_len2+2:]
            )
            exp_other = (
                (1,) * (len1 - other_len1) + tup_other[:other_len1+1] + 
                (1,) * (len2 - other_len2) + tup_other[other_len1+1:other_len1+other_len2+2] + 
                (1,) * (len3 - other_len3) + tup_other[other_len1+other_len2+2:]
            )
            if self._batch_first: batch_dim, channel_dim = len1, len1 + len2 + 1
            else: batch_dim, channel_dim = len1 + len2 + 1, len1
            exp_self = Size(exp_self, batch_dim=batch_dim, channel_dim=channel_dim)
            exp_other = Size(exp_other, batch_dim=batch_dim, channel_dim=channel_dim)
            return exp_self, exp_other

        if self.has_batch and other.has_batch:
            lp_self, lp_other = self[:self.batch_dimension] ^ other[:other.batch_dimension]
            rp_self, rp_other = self[self.batch_dimension+1:] ^ other[other.batch_dimension+1:]
            return (
                lp_self + Size([self.batch_size]) + rp_self,
                lp_other + Size([other.batch_size]) + rp_other
            )

        if self.has_channel and other.has_channel:
            lp_self, lp_other = self[:self.channel_dimension] ^ other[:other.channel_dimension]
            rp_self, rp_other = self[self.channel_dimension+1:] ^ other[other.channel_dimension+1:]
            return (
                lp_self + Size({self.channel_size}) + rp_self,
                lp_other + Size({other.channel_size}) + rp_other
            )

        raise RuntimeError("Unexpected error occurs in sizes expanding, please contact developers for more information. ")
    __ixor__ = __rxor__ = __xor__

    def __getitem__(self, k):
        if isinstance(k, INT): return super().__getitem__(k)
        return Size(*self.python_repr[k])

    @property
    def python_repr(self):
        args = list(self)
        batch_dim = self.batch_dimension
        if batch_dim < self.ndim: args[batch_dim] = [args[batch_dim]]
        channel_dim = self.channel_dimension
        if channel_dim < self.ndim: args[channel_dim] = {args[channel_dim]}
        return tuple(args)

    def __str__(self):
        rep = self.python_repr
        if len(rep) == 1: rep = str(rep).replace(',', '')
        return f"torchplus.Size{rep}"
    __repr__ = __str__

class Tensor(torch.Tensor):

    init = False

    @staticmethod
    def _make_subclass(cls, data, auto_device=_auto_device, requires_grad=False, device=None):
        to_device = None
        if device is None and auto_device: to_device = Device
        if device is not None: to_device = device
        if to_device is not None and to_device != data.device: data = data.to(to_device)
        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        if isinstance(data, Tensor): return self.special_from_(data)
        return self

    def __new__(cls, *args, **kwargs):

        to_device = kwargs.get('device', Device if kwargs.get('auto_device', _auto_device) else None)

        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, INT): pass
            elif isinstance(arg, torch.Tensor) and (to_device is None or arg.device == to_device):
                self = arg.as_subclass(Tensor)
                self.requires_grad = kwargs.get('requires_grad', arg.requires_grad)
                return self.set_special_(kwargs.get('batch_dim', None), kwargs.get('channel_dim', None))
            elif hasattr(arg, 'shape') or isinstance(arg, list):
                if not isinstance(arg, torch.Tensor):
                    if to_device is None: arg = torch.as_tensor(arg)
                    else: arg = torch.as_tensor(arg, device=to_device)
                rg = kwargs.get('requires_grad', arg.requires_grad)
                if to_device is None or arg.device == to_device:
                    self = super()._make_subclass(cls, arg, rg)
                else:
                    self = super()._make_subclass(cls, arg.to(to_device), rg)
                self.special_from_(arg)
                return self.set_special_(kwargs.get('batch_dim', None), kwargs.get('channel_dim', None))
            elif isinstance(arg, tuple): args = arg
            else: raise TypeError("Unrecognized initialization of 'Tensor'. ")

        # args is a tuple of integers (or a Size)
        if to_device is None: self = super().__new__(cls, *args, **kwargs)
        else: self = super().__new__(cls, *args, device=to_device, **kwargs)
        self.special_from_(args)
        return self.set_special_(kwargs.get('batch_dim', None), kwargs.get('channel_dim', None))

    def requires_grad_(self, mode=True):
        self.requires_grad = mode
        return self

    def refine_names(self, *args, kwargs):
        self.has_names = True
        return super().refine_names(*args, **kwargs)

    @property
    def ishape(self): return super().shape

    @property
    def shape(self):
        shape = Size(super().shape)
        shape.special_from_(self)
        if self.has_names():
            if not shape.has_batch:
                isbatch = [('batch' in x.lower()) if x else x for x in self.names]
                if ANY(isbatch):
                    ibatch = isbatch.index(True)
            if not shape.has_channel:
                ischannel = [('channel' in x.lower()) if x else x for x in self.names]
                if ANY(ischannel):
                    ichannel = ischannel.index(True)
        return shape

    def rename(self, *args, **kwargs):
        ibatch = ichannel = None
        for i, n in enumerate(args):
            if not isinstance(n, str): continue
            elif 'batch' in x.lower():
                if ibatch is not None:
                    raise TypeError("Multiple batch dimensions not supported. ")
                ibatch = i
            elif 'channel' in x.lower():
                if ichannel is not None:
                    raise TypeError("Multiple channel dimensions not supported. ")
                ichannel = i
            if ibatch is not None and ichannel is not None: break
        return super().rename(*args, **kwargs)

    def refine_names(self, *args):
        ibatch = ichannel = None
        for i, n in enumerate(args):
            if not isinstance(n, str): continue
            elif 'batch' in x.lower():
                if ibatch is not None:
                    raise TypeError("Multiple batch dimensions not supported. ")
                ibatch = i
            elif 'channel' in x.lower():
                if ichannel is not None:
                    raise TypeError("Multiple channel dimensions not supported. ")
                ichannel = i
            if ibatch is not None and ichannel is not None: break
        return super().refine_names(*args)
        
    @property
    def batch_dimension(self): return self._special[0 if self._batch_first else 1]

    @batch_dimension.setter
    def batch_dimension(self, batch_dim):
        self.set_special_(batch_dim, None)

    def batch_dimension_(self, value):
        self.batch_dimension = value
        return self

    @property
    def batch_size(self): return self.shape.batch_size

    @property
    def channel_dimension(self): return self._special[1 if self._batch_first else 0]

    @channel_dimension.setter
    def channel_dimension(self, channel_dim):
        self.set_special_(None, channel_dim)

    def channel_dimension_(self, value):
        self.channel_dimension = value
        return self

    @property
    def channel_size(self): return self.shape.channel_size

    @property
    def space(self):
        s = self._special
        t = tuple(self.ishape)
        return t[:s[0]] + t[s[0]+1:s[1]] + t[s[1]+1:]

    @property
    def nele(self): return super().numel()
    def numel(self): return super().numel()

    @property
    def n_dim(self): return self.ndim

    @property
    def nspace(self): return self.ndim if not self.init else (self.ndim - (self._special[0] < self.ndim) - (self._special[1] < self.ndim))

    @property
    def nspecial(self): return 0 if not self.init else ((self._special[0] < self.ndim) + (self._special[1] < self.ndim))

    nbatch = batch_size
    nchannel = channel_size
    n_ele = nele
    n_space = nspace
    n_special = nspecial

    @property
    def has_batch(self): return self.init and self._special[0 if self._batch_first else 1] < self.ndim

    @property
    def has_channel(self): return self.init and self._special[1 if self._batch_first else 0] < self.ndim

    @property
    def has_special(self): return self.init and self._special != [self.ndim, self.ndim]

    @property
    def special(self): return None if not self.init else [x for x in self._special if x < self.ndim]

    def special_from_(self, other=None):

        if isinstance(other, Size) or isinstance(other, Tensor) and other.init:
            self._special = [self.ndim if x == other.ndim else x for x in other._special]
            self._batch_first = other._batch_first
        else:
            self._special = [self.ndim, self.ndim]
            self._batch_first = True

        self.init = True
        return self

    def add_special_from_(self, other=None):

        if isinstance(other, Size):

            doit = False
            batch_dim = None
            channel_dim = None

            if self.batch_dimension == self.ndim:
                batch_dim = other.batch_dimension
                if batch_dim != self.ndim:
                    doit = True
            if self.channel_dimension == self.ndim:
                channel_dim = other.channel_dimension
                if channel_dim != self.ndim:
                    doit = True
            if doit: self.set_special_(batch_dim, channel_dim)

        self.init = True
        return self

    def set_special_(self, batch_dim=None, channel_dim=None, *, special=[], bf=True):

        if batch_dim is None and channel_dim is None:
            if not special:
                if not self.init: return self.special_from_()
                else: return self
            a, b = special
            if not isinstance(a, INT): a = self.ndim
            if not isinstance(b, INT): b = self.ndim
            if a < 0: a += self.ndim
            if b < 0: b += self.ndim
            if not 0 <= a <= self.ndim or not 0 <= b <= self.ndim:
                raise TypeError(f"Special dimension should be a dimension index which is smaller than {self.ndim}. ")
            if not self.init: self._batch_first = bf
            if a < b:
                self._special = [a, b]
            else:
                self._special = [b, a]
                self._batch_first = not self._batch_first
            self.init = True
            return self

        if batch_dim is not None:
            if not isinstance(batch_dim, INT): batch_dim = self.ndim
            if batch_dim < 0: batch_dim = batch_dim + self.ndim
            if not 0 <= batch_dim <= self.ndim:
                raise TypeError(f"batch_dimension should be a dimension index which is smaller than {self.ndim}. ")
        elif self.init:
            batch_dim = MIN(self.batch_dimension, self.ndim)
        else:
            batch_dim = self.ndim

        if channel_dim is not None:
            if not isinstance(channel_dim, INT): channel_dim = self.ndim
            if channel_dim < 0: channel_dim = channel_dim + self.ndim
            if not 0 <= channel_dim <= self.ndim:
                raise TypeError(f"channel_dimension should be a dimension index which is smaller than {self.ndim}. ")
        elif self.init:
            channel_dim = MIN(self.channel_dimension, self.ndim)
        else:
            channel_dim = self.ndim

        if batch_dim < channel_dim:
            self._batch_first = True
            self._special = [batch_dim, channel_dim]
        elif channel_dim < batch_dim:
            self._batch_first = False
            self._special = [channel_dim, batch_dim]
        elif batch_dim < self.ndim:
            raise ValueError(f"special dimensions can not be the same: {batch_dim} and {channel_dim}. ")
        else:
            self._batch_first = True
            self._special = [channel_dim, channel_dim]

        self.init = True
        return self

    def remove_special_(self):
        self._special = [self.ndim, self.ndim]
        self._batch_first = True
        return self

    def remove_batch_(self):
        return self.set_special_(batch_dim=self.ndim)

    def remove_channel_(self):
        return self.set_special_(channel=self.ndim)

    def normalize_(self):
        m, M = self.min(), self.max()
        if m == M:
            if M >= 1: return self.one_()
            if m <= 0: return self.zero_()
            return self
        self.sub_(m)
        self.div_(M-m)
        return self

    def normalize(self):
        m, M = self.min(), self.max()
        if m == M:
            if M >= 1: return ones_like(self)
            if m <= 0: return zeros_like(self)
            return self
        return (self - m) / (M - m)

    for func in tf_list:
        exec(f"def {func}(*args, **kwargs): return tp.{func}(*args, **kwargs)")

    def tensor(self): return super()._make_subclass(torch.Tensor, self, self.requires_grad)
    def numpy(self): return super(torch.Tensor, self.cpu().detach()).numpy()

    def one_(self): return self.zero_().add_(1)
    
    def autodevice_(self):
        if _auto_device and self.device != Device: return self.to(Device)
        return self

    def device_(self, device):
        if device != self.device: return self.to(device)
        return self

    def size(self, *k: [INT, str]):
        if len(k) == 0:
            return self.shape
        i = [(self.names.index(x) if x in self.names else None) if isoftype(x, str) else x for x in k]
        if None in i:
            return super().size(k[i.index(None)])
        if len(i) == 1:
            return self.ishape[i[0]]
        return tuple(self.ishape[x] for x in i)

    def squeeze(self, *dims: int, dim=None):
        if len(dims) > 0 and dim is not None:
            raise TypeError("squeeze function only accept either argument or positional argument. But both are given")
        if len(dims) == 1 and isinstance(dims[0], tuple):
            dims = dims[0]
        if dim is None:
            dim = dims
        if not isinstance(dim, tuple):
            dim = (dim,)
        if len(dim) == 0: dim = tuple(i for i, x in enumerate(self.ishape) if x == 1)[::-1]
        a, b = self._special
        ndim = self.ndim
        bf = self._batch_first
        for d in dim:
            if isinstance(d, list): d = d[0]
            if isinstance(d, set): d = d.pop()
            self = super(Tensor, self).squeeze(d)
            if a == d: a = ndim
            if 0 <= d <= a or d + ndim <= a: a -= 1
            if b == d: b = ndim
            if 0 <= d <= b or d + ndim <= b: b -= 1
            ndim -= 1
        return self.as_subclass(Tensor).set_special_(special=[a, b], bf=bf)

    def squeeze_(self, *dims:int, dim=None):
        if len(dims) > 0 and dim is not None:
            raise TypeError("squeeze_ function only accept either argument or positional argument. But both are given")
        if len(dims) == 1 and isinstance(dims[0], tuple):
            dims = dims[0]
        if dim is None:
            dim = dims
        if not isinstance(dim, tuple):
            dim = (dim,)
        if len(dim) == 0: dim = tuple(i for i, x in enumerate(self.ishape) if x == 1)
        a, b = self._special
        ndim = self.ndim
        bf = self._batch_first
        for d in dim:
            if isinstance(d, list): d = d[0]
            if isinstance(d, set): d = d.pop()
            self = super(Tensor, self).squeeze_(d)
            if a == d: a = ndim
            if 0 <= d <= a or d + ndim <= a: a -= 1
            if b == d: b = ndim
            if 0 <= d <= b or d + ndim <= b: b -= 1
            ndim -= 1
        return self.as_subclass(Tensor).set_special_(special=[a, b], bf=bf)

    def unsqueeze(self, *dims: int, dim=None):
        if len(dims) > 0 and dim is not None:
            raise TypeError("unsqueeze function only accept either argument or positional argument. But both are given")
        if len(dims) == 1 and isinstance(dims[0], tuple):
            dims = dims[0]
        if dim is None:
            dim = dims
        if not isinstance(dim, tuple):
            dim = (dim,)
        if len(dim) == 0: dim = (0,)
        a, b = self._special
        ndim = self.ndim
        bf = self._batch_first
        ibatch = ichannel = None
        for d in dim:
            if isinstance(d, list): d = d[0]; ibatch = d
            if isinstance(d, set): d = d.pop(); ichannel = d
            self = super(Tensor, self).unsqueeze(d)
            if 0 <= d <= a or d + ndim <= a: a += 1
            if 0 <= d <= b or d + ndim <= b: b += 1
            ndim += 1

        return self.as_subclass(Tensor).set_special_(special=[a, b], bf=bf).set_special_(batch_dim=ibatch, channel_dim=ichannel)

    def unsqueeze_(self, *dims:int, dim=None):
        if len(dims) > 0 and dim is not None:
            raise TypeError("unsqueeze_ function only accept either argument or positional argument. But both are given")
        if len(dims) == 1 and isinstance(dims[0], tuple):
            dims = dims[0]
        if dim is None:
            dim = dims
        if not isinstance(dim, tuple):
            dim = (dim,)
        if len(dim) == 0: dim = (0,)
        a, b = self._special
        ndim = self.ndim
        bf = self._batch_first
        ibatch = ichannel = None
        for d in dim:
            if isinstance(d, list): d = d[0]; ibatch = d
            if isinstance(d, set): d = d.pop(); ichannel = d
            self = super(Tensor, self).unsqueeze_(d)
            if 0 <= d <= a or d + ndim <= a: a += 1
            if 0 <= d <= b or d + ndim <= b: b += 1
            ndim += 1
        return self.as_subclass(Tensor).set_special_(special=[a, b], bf=bf).set_special_(batch_dim=ibatch, channel_dim=ichannel)

    def expand_to(self, *target):
        with torch._C.DisableTorchFunction():

            if len(target) == 1 and not isinstance(target[0], INT): target = target[0]
            if isinstance(target, torch.Tensor): target = target.shape
            if not isinstance(target, Size): target = Size(target)
            if self.init and self.nspecial == target.nspecial and target._batch_first != self._batch_first:
                if self.nspace == 0 and self.ndim == 2: self = self[::-1]
                else: raise TypeError(f"Batch and channel order not matched for {self.shape} and {target}")
                
            axis_map = list(RANGE(self.ndim))
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
                    while p < target.ndim and target[p] not in (s, -1): p += 1
                    axis_map[i] = p
                    p += 1
                if p >= target.ndim  + 1: raise TypeError(f"Unable to expand sizes {self.shape} to {target}. ")
            res = self.unsqueeze_to(target, axis=axis_map).repeat(tuple(1 if i in axis_map else (x if x >= 0 else 1) for i, x in enumerate(target)))

        return res.as_subclass(Tensor).special_from_(target)

    def unsqueeze_to(self, *target, axis: list):
        with torch._C.DisableTorchFunction():

            if len(target) == 1 and not isinstance(target[0], INT): target = target[0]
            if isinstance(target, torch.Tensor): target = target.shape
            if not isinstance(target, Size): target = Size(target)

            new_size = list(target)
            for i in RANGE(len(new_size)):
                if i not in axis or self.ishape[axis.index(i)] in (1, -1):
                    new_size[i] = 1
            res = self.view(*new_size)

        return res.as_subclass(Tensor).special_from_(target)

    def multiple(self, num, dim=0):
        if isinstance(dim, list): d = dim[0]
        elif isinstance(dim, set): d = dim.pop(); dim = {d}
        else: d = dim
        return self.unsqueeze(dim).repeat((1,) * d + (num,) + (1,) * (self.ndim - d))

    def sample(self, dim: INT = None, number: INT = 1, random: bool = True) -> 'Tensor':
        """
        sample(self, dim: int = self.batch_dimension, numbder: int = 1, random: bool = True) -> Tensor

        Sample a few subspaces from a given dimension.
        data.sample(2, 1, random=False) is equivalant to data[:, :, 0, ...].
        """
        if dim is None or isinstance(dim, list) and dim == []: dim = self.batch_dimension
        if dim is None or isinstance(dim, dict) and dim == {}: dim = self.channel_dimension
        if dim < 0: dim += self.ndim
        if dim is None: raise TypeError("Argument 'dim' needed for sampling Tensors with no special dimensions. ")
        if number < 1: raise TypeError("Argument 'number' for sampling Tensors can not be smaller than 1. ")
        sample_indices = [slice(None)] * self.dim()
        if self.shape[dim] < number: raise TypeError(f"Too many elements needed to be sampled from dimension {dim}")
        if random:
            import random
            samples = random.sample(RANGE(self.shape[dim]), k = number)
        else: samples = list(RANGE(number))
        if len(samples) != 1:
            sample_indices[dim] = samples
            output_tensor = self[tuple(sample_indices)].as_subclass(Tensor).special_from_(self)
            output_tensor.indices = samples
            return output_tensor
        else:
            sample_indices[dim] = samples[0]
            output_tensor = self[tuple(sample_indices)].as_subclass(Tensor).set_special_(special=[x - 1 if x > dim else (self.ndim-1 if x == dim else x) for x in self._special], bf=self._batch_first)
            output_tensor.indices = samples
            return output_tensor

    def pick(self, index: INT, dim: INT):
        """
        pick(self, dim, index) -> Tensor

        pick one of the item on dimension `dim` for big tensors. 
        data.pick(4, 2) is equivalent to data[:, :, 4]
        """
        if dim < 0: dim += self.ndim
        return self[(slice(None),) * dim + (index,)]

    def split(self, sec=1, dim=None, squeeze=False):
        """
        pick(self, dim, index) -> Tensor

        pick one of the item on dimension `dim` for big tensors. 
        data.pick(4, 2) is equivalent to data[:, :, 4]
        """
        if dim is None: dim = self.channel_dimension
        if sec == 1 or isinstance(sec, (tuple, list)) and all(x == 1 for x in sec):
            if squeeze: return tuple(x.as_subclass(Tensor).special_from_(self).squeeze(dim) for x in super().split(sec, dim))
        return tuple(x.as_subclass(Tensor).special_from_(self) for x in super().split(sec, dim))

    def mvdim(self, dim1: INT, dim2: INT):
        """
        mvdim(self, dim1, dim2) -> Tensor

        move dim1 to dim2(specified in the targeting size)
        data of size (2, 3, 4, 5) can be transform to (2, 4, 5, 3) by data.mvdim(1, -1) or data.mvdim(1, 3)
        """
        if dim1 < 0: dim1 += self.ndim
        if dim2 < 0: dim2 += self.ndim

        with torch._C.DisableTorchFunction():
            if dim1 == dim2: res = self
            elif dim1 < dim2: res = self.unsqueeze(dim2+1).transpose(dim1, dim2+1).squeeze(dim1)
            else: res = self.unsqueeze(dim2).transpose(dim1+1, dim2).squeeze(dim1+1)

        return res.as_subclass(Tensor).set_special_(special=[dim2 if x == dim1 else (
            x if x > dim2 and x > dim1 or x < dim1 and x < dim2 
            else (x + 1 if dim1 > dim2 else x - 1)) for x in self._special])

    def mvdim_(self, dim1: INT, dim2: INT):
        """
        In-place operation for mvdim
        """
        if dim1 < 0: dim1 += self.ndim
        if dim2 < 0: dim2 += self.ndim

        with torch._C.DisableTorchFunction():
            if dim1 == dim2: res = self
            elif dim1 < dim2: res = self.unsqueeze_(dim2+1).transpose_(dim1, dim2+1).squeeze_(dim1)
            else: res = self.unsqueeze_(dim2).transpose_(dim1+1, dim2).squeeze_(dim1+1)

        return res.as_subclass(Tensor).special_from_(special=[dim2 if x == dim1 else (
            x if x > dim2 and x > dim1 or x < dim1 and x < dim2 
            else (x + 1 if dim1 > dim2 else x - 1)) for x in self._special])

    def tobytes(self): return self.numpy().tobytes()

    def cat(self, other, dim=0):
        return torch.cat((self, other), dim=dim)

    def stack(self, other, dim=0):
        return torch.stack((self, other), dim=dim)

    @property
    def T(self: 'Tensor') -> 'Tensor':
        if not self.has_special: return super().T
        s = self._special

        with torch._C.DisableTorchFunction():
            permute_dim = tuple(RANGE(s[0]))[::-1] + (s[0],) + tuple(RANGE(s[0]+1, s[1]))[::-1] + (s[1],) * (s[1] != self.ndim) + tuple(RANGE(s[1]+1, self.ndim))[::-1]
        
        return self.permute(*permute_dim)

    def t(self) -> 'Tensor':
        return self.T

    def t_(self) -> 'Tensor':
        """
        t_() -> Tensor

        In-place version of :meth:`~Tensor.t`
        """
        if not self.has_special: return super().t_()
        s = self._special
        with torch._C.DisableTorchFunction():
            for i in RANGE(s[0] // 2):
                self.transpose_(i, s[0] - i - 1)
            for i in RANGE((s[1] - s[0] - 1) // 2):
                self.transpose_(s[0] + i + 1, s[1] - i - 1)
            for i in RANGE((self.ndim - s[1] - 1) // 2):
                self.transpose_(s[1] + i + 1, self.ndim - i - 1)
        return self

    def __getattr__(self, key):
        if not self.init:
            if key == '_special': return [self.ndim, self.ndim]
            elif key == '_batch_first': return True
        return super().__getattr__(key)

    def __matmul__(self, other, **kwargs):
        if isinstance(other, torch.Tensor) and self.has_special or isinstance(other, Tensor) and other.has_special:
            a, b = self.shape[:-2] ^ other.shape[:-2]
            with torch._C.DisableTorchFunction():
                res = super(torch.Tensor, self.view(a + tuple(self.shape[-2:]))).__matmul__(other.view(b + tuple(other.shape[-2:])))
            return res.as_subclass(Tensor).special_from_(self.shape[:-2])
        return super().__matmul__(other, **kwargs).as_subclass(Tensor).special_from_()

    def __op__(self, opname, other, **kwargs):
        if self.has_special or isinstance(other, Tensor) and other.has_special:
            if not isinstance(other, Tensor): other = tensor(other)
            a, b = self.shape ^ other.shape
            with torch._C.DisableTorchFunction():
                res = getattr(super(Tensor, self.view(a).as_subclass(Tensor)), opname)(other.view(b))
            return res.as_subclass(Tensor).special_from_(a)
        return getattr(super(), opname)(other, **kwargs).as_subclass(Tensor).special_from_()

    for op in '''
    add iadd radd
    sub isub rsub
    mul imul rmul
    div idiv rdiv
    pow ipow rpow
    mod imod rmod
    truediv itruediv rtruediv
    floordiv ifloordiv rfloordiv
    eq ieq req
    ne ine rne
    or ior ror
    and iand rand
    xor ixor rxor
    lt le gt ge
    '''.split():
        exec(f"def __{op}__(self, *args, **kwargs): return self.__op__('__{op}__', *args, **kwargs)")

    ###### old operation code ######
    #    if isinstance(other, torch.Tensor):
    #        other = Tensor(other)
    #        if self.dim() == other.dim():
    #            return super().__add__(other)
    #        elif self.dim() < other.dim():
    #            return self.expand_as(other).__add__(other)
    #        else:
    #            return super().__add__(other.expand_as(self))
    #    return super().__add__(other)
    #################################

    def __repr__(self, *args, **kwargs):
        string = super().__repr__(*args, **kwargs)
        if 'shape=' not in string:
            string = string.rstrip(')') + f', shape={self.shape})'
        return string.replace("tensor", "Tensor")

    def __str__(self, *args, **kwargs):
        string = super().__str__(*args, **kwargs)
        if 'shape=' not in string:
            string = string.rstrip(')') + f', shape={self.shape})'
        return string.replace("tensor", "Tensor")

    def __hash__(self): return super().__hash__()

    @staticmethod
    def __torch_function_convert__(ret, cls):
        with torch._C.DisableTorchFunction():
            if isinstance(ret, torch.Tensor):
                ret = ret.as_subclass(cls)
            if isinstance(ret, (tuple, list)):
                # Also handles things like namedtuples
                ret = type(ret)(Tensor.__torch_function_convert__(r, cls) for r in ret)
            return ret

    @staticmethod
    def __torch_function_collect__(r, c):
        if isinstance(r, (tuple, list)):
            for x in r: Tensor.__torch_function_collect__(x, c)
        if isinstance(r, Tensor) and not r.init: c.append(r)

    @staticmethod
    def __torch_function_convert_apply__(ret, apply, cls):

        with torch._C.DisableTorchFunction():

            if isinstance(ret, Tensor) and not ret.init:
                apply(ret)
                if cls != Tensor: ret = ret.as_subclass(cls)
                return ret

            if isinstance(ret, torch.Tensor):
                ret = ret.as_subclass(cls)
                return ret

            if isinstance(ret, (tuple, list)):
                # Also handles things like namedtuples
                return type(ret)(Tensor.__torch_function_convert_apply__(r, apply, cls) for r in ret)

            if 'tuple_iterator' in str(type(ret)):
                # And things iterator like Generators
                def out():
                    for r in ret:
                        if isinstance(r, torch.Tensor):
                            r = r.as_subclass(cls)
                            apply(r)
                            yield r
                return out()

            return ret

    @classmethod
    def __torch_function_ele_wise_func__(cls, func, types, args=(), kwargs=None):
        """
        FOR: tensor where
        __add__ __iadd__ __radd__
        __sub__ __isub__ __rsub__
        __mul__ __imul__ __rmul__
        __div__ __idiv__ __rdiv__
        __pow__ __ipow__ __rpow__
        __mod__ __imod__ __rmod__
        __truediv__ __itruediv__ __rtruediv__
        __floordiv__ __ifloordiv__ __rfloordiv__
        __eq__ __ieq__ __req__
        __ne__ __ine__ __rne__
        __or__ __ior__ __ror__
        __and__ __iand__ __rand__
        __xor__ __ixor__ __rxor__
        __lt__ __le__ __gt__ __ge__
        """
        self = args[0]

        with torch._C.DisableTorchFunction():
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, type(NotImplemented)):
                raise NotImplementedError(f"{func} for {args} is not implemented. ")

        def apply(r):
            r.special_from_(self)

        return Tensor.__torch_function_convert_apply__(ret, apply, cls)

    @classmethod
    def __torch_function_resizing_func__(cls, func, types, args=(), kwargs=None):
        "FOR: reshape view zeros ones rand randn"
        dims = args[1:]
        if len(dims) == 1: dims = dims[0]
        if not isinstance(dims, Size): dims = Size(dims)
        args = args[:1] + tuple(dims)

        with torch._C.DisableTorchFunction():
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, type(NotImplemented)):
                raise NotImplementedError(f"{func} for {args} is not implemented. ")

        def apply(r):
            r.special_from_(dims)

        return Tensor.__torch_function_convert_apply__(ret, apply, cls)

    @classmethod
    def __torch_function_resizing_as_func__(cls, func, types, args=(), kwargs=None):
        "FOR: reshape_as view_as unsqueeze_to expand_to"
        dims = args[-1].shape

        with torch._C.DisableTorchFunction():
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, type(NotImplemented)):
                raise NotImplementedError(f"{func} for {args} is not implemented. ")

        def apply(r):
            r.special_from_(dims)

        return Tensor.__torch_function_convert_apply__(ret, apply, cls)

    @classmethod
    def __torch_function_randint_func__(cls, func, types, args=(), kwargs=None):
        "FOR: randint"
        dims = Size(args[3])
        args = args[:3] + (dims,)

        with torch._C.DisableTorchFunction():
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, type(NotImplemented)):
                raise NotImplementedError(f"{func} for {args} is not implemented. ")

        def apply(r):
            r.special_from_(dims)

        return Tensor.__torch_function_convert_apply__(ret, apply, cls)

    @classmethod
    def __torch_function_transpose_func__(cls, func, types, args=(), kwargs=None):
        "FOR: transpose transpose_"
        self = args[0]

        with torch._C.DisableTorchFunction():
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, type(NotImplemented)):
                raise NotImplementedError(f"{func} for {args} is not implemented. ")

        def apply(r):
            dim1 = kwget(kwargs, 'dim0', args[1])
            dim2 = kwget(kwargs, 'dim1', args[2])
            a, b = self._special
            if a == dim1: a = dim2
            elif a == dim2: a = dim1
            if b == dim1: b = dim2
            elif b == dim2: b = dim1
            r.set_special_(special=[a, b], bf=self._batch_first)

        return Tensor.__torch_function_convert_apply__(ret, apply, cls)

    @classmethod
    def __torch_function_permute_func__(cls, func, types, args=(), kwargs=None):
        "FOR: permute"
        self = args[0]
        dims = args[1:]
        if len(dims) == 1 and isinstance(dims[0] (list, tuple)): dims = dims[0]

        with torch._C.DisableTorchFunction():
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, type(NotImplemented)):
                raise NotImplementedError(f"{func} for {args} is not implemented. ")

        d = lambda i: dims[i] if i < len(dims) else self.ndim
        def apply(r):
            r.set_special_(special=[d(self._special[0]), d(self._special[1])], bf=self._batch_first)

        return Tensor.__torch_function_convert_apply__(ret, apply, cls)

    @classmethod
    def __torch_function_matmul_func__(cls, func, types, args=(), kwargs=None):
        "FOR: mm bmm smm matmul __matmul__"
        self, other = args[:2]

        with torch._C.DisableTorchFunction():
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, type(NotImplemented)):
                raise NotImplementedError(f"{func} for {args} is not implemented. ")

        def apply(r):
            r.special_from_(self.shape[:-1] + other.shape[-1:])

        return Tensor.__torch_function_convert_apply__(ret, apply, cls)

    @classmethod
    def __torch_function_addmm_func__(cls, func, types, args=(), kwargs=None):
        "FOR: addmm addbmm saddmm"
        _, self, weight = args[:3]

        with torch._C.DisableTorchFunction():
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, type(NotImplemented)):
                raise NotImplementedError(f"{func} for {args} is not implemented. ")

        def apply(r):
            r.special_from_(self.shape[:-1] + other.shape[-1:])

        return Tensor.__torch_function_convert_apply__(ret, apply, cls)

    @classmethod
    def __torch_function_reducing_func__(cls, func, types, args=(), kwargs=None):
        "FOR: cummin cummax cumsum cumprod sum prod min max mean std argmin argmax"
        func_name = str(func).split(' of ')[0].split()[-1].strip("'")
        mkwargs = kwargs if kwargs is not None else {}
        if len(args) == 1 and 'dim' not in mkwargs and func_name in ('sum', 'prod', 'mean', 'std', 'cumsum', 'cumprod'):
            self = args[0]
            s = self._special
            dims = list(RANGE(s[0])) + list(RANGE(s[0]+1, s[1])) + list(RANGE(s[1]+1, self.ndim))
            args += (dims,)

        with torch._C.DisableTorchFunction():
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, type(NotImplemented)):
                raise NotImplementedError(f"{func} for {args} is not implemented. ")

        def apply(r):
            if len(args) > 1:
                self = args[0]
                dims = mkwargs.get('dim', args[1:])
                if len(dims) == 1 and isinstance(dims[0], (list, tuple)): dims = dims[0]
                if len(dims) == 0: r.special_from_()
                else: r.set_special_(special=[x - len([d for d in dims if 0 <= d < x or d + self.ndim < x]) if x not in dims else r.ndim for x in self._special], bf=self._batch_first)

        return Tensor.__torch_function_convert_apply__(ret, apply, cls)

    @classmethod
    def __torch_function_expanding_func__(cls, func, types, args=(), kwargs=None):
        "FOR: unsqueeze unsqueeze_ squeeze squeeze_"
        # self = args[0]
        # dims = kwget(kwargs, 'dim', args[1:])
        # if len(dims) == 1 and isinstance(dims[0], tuple): dims = dims[0]

        with torch._C.DisableTorchFunction():
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, type(NotImplemented)):
                raise NotImplementedError(f"{func} for {args} is not implemented. ")

        def apply(r): ...
            # a, b = self._special
            # ndim = self.ndim
            # for d in dims:
            #     if 0 <= d <= a or d + ndim <= a: a += 1
            #     if 0 <= d <= b or d + ndim <= b: b += 1
            #     ndim += 1
            # r.set_special_(special=[a, b], bf=self._batch_first)

        return Tensor.__torch_function_convert_apply__(ret, apply, cls)

    @classmethod
    def __torch_function_flatten_func__(cls, func, types, args=(), kwargs=None):
        "FOR: flatten flatten_"
        dims = kwget(kwargs, 'dim', args[1:])
        if isinstance(dims, tuple) and len(dims) <= 0:
            dims = kwget(kwargs, 'start_dim', 0), kwget(kwargs, 'end_dim', -1)

        with torch._C.DisableTorchFunction():
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, type(NotImplemented)):
                raise NotImplementedError(f"{func} for {args} is not implemented. ")

        def apply(r):
            self = args[0]
            if len(dims) == 1:
                dim = dims[0]
                if dim < 0: dim += self.ndim
                r.set_special_(special=[r.ndim if x >= dim else x for x in self._special], bf=self._batch_first)
            else:
                dim1, dim2 = dims
                if dim1 < 0: dim1 += self.ndim
                if dim2 < 0: dim2 += self.ndim
                r.set_special_(special=[r.ndim if dim1 <= x <= dim2 or x > r.ndim else x for x in self._special], bf=self._batch_first)

        return Tensor.__torch_function_convert_apply__(ret, apply, cls)

    @classmethod
    def __torch_function_getitem_func__(cls, func, types, args=(), kwargs=None):
        "FOR: __getitem__ __iter__"
        if len(args) == 1: self, dims = args[0], 0
        else: self, dims = args

        with torch._C.DisableTorchFunction():
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, type(NotImplemented)):
                raise NotImplementedError(f"{func} for {args} is not implemented. ")

            ndim = self.ndim

            if not isinstance(dims, tuple): dims = (dims,)
            types = [type(x) for x in dims]
            if type(...) in types:
                if types.count(type(...)) > 1:
                    raise TypeError("")
                lp = dims[:types.index(type(...))]
                rp = dims[types.index(type(...))+1:]
            else:
                lp = dims
                rp = tuple()
            offset = ndim - len(rp)
            isnormal = ''.join('1' if issubclass(t, (INT, slice, type(...))) else '0' for t in types)
            if '1' in isnormal.strip('1') or '0' not in isnormal: index = -1
            else: index = isnormal.index('0')

        def apply(r):
            a, b = self._special
            rdim = r.ndim
            offset2 = rdim - ndim + len(lp) + len(rp) - len([d for d in lp + rp if isinstance(d, slice)])
            if a < len(lp) and not isinstance(lp[a], slice): a = None
            elif offset <= a < ndim and not isinstance(rp[a - offset], slice): a = None
            if b < len(lp) and not isinstance(lp[b], slice): b = None
            elif offset <= b < ndim and not isinstance(rp[b - offset], slice): b = None
            if a is not None and a < ndim:
                a += offset2 if a > index else 0
                a -= len([d for d in RANGE(len(lp)) if (0 <= d < a or d + ndim < a) and not isinstance(lp[d], slice)])
                a -= len([d for d in RANGE(offset, ndim) if (0 <= d < a or d + ndim < a) and not isinstance(rp[d-offset], slice)])
            elif a is not None: a = rdim
            if b is not None and b < ndim:
                b += offset2 if b > index else 0
                b -= len([d for d in RANGE(len(lp)) if (0 <= d < b or d + ndim < b) and not isinstance(lp[d], slice)])
                b -= len([d for d in RANGE(offset, ndim) if (0 <= d < b or d + ndim < b) and not isinstance(rp[d-offset], slice)])
            elif b is not None: b = rdim
            r.set_special_(special=[a, b], bf=self._batch_first)

        return Tensor.__torch_function_convert_apply__(ret, apply, cls)

    @classmethod
    def __torch_function_concatenate_func__(cls, func, types, args=(), kwargs=None):
        "FOR: cat concatenate"
        self = args[0]
        if isinstance(self, tuple): self = self[0]

        with torch._C.DisableTorchFunction():
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, type(NotImplemented)):
                raise NotImplementedError(f"{func} for {args} is not implemented. ")

        def apply(r):
            r.special_from_(self)

        return Tensor.__torch_function_convert_apply__(ret, apply, cls)

    @classmethod
    def __torch_function_stack_func__(cls, func, types, args=(), kwargs=None):
        "FOR: stack"
        self = args[0]
        if isinstance(self, tuple): self = self[0]
        dim = kwget(kwargs, 'dim', args[-1] if isinstance(args[-1], INT) else 0)

        with torch._C.DisableTorchFunction():
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, type(NotImplemented)):
                raise NotImplementedError(f"{func} for {args} is not implemented. ")

        def apply(r):
            r.special_from_(special=[x + 1 if x >= dim else x for x in self._special])

        return Tensor.__torch_function_convert_apply__(ret, apply, cls)

    @classmethod
    def __torch_function_default_func__(cls, func, types, args=(), kwargs=None):
        "FOR: all the remainings"
        self = args[0]

        with torch._C.DisableTorchFunction():
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, type(NotImplemented)):
                raise NotImplementedError(f"{func} for {args} is not implemented. ")

        def apply(r):
            r.special_from_(self)

        return Tensor.__torch_function_convert_apply__(ret, apply, cls)

    def __torch_function_keys__(func):
        return func.__func__.__doc__.split('\n')[0].split(':')[-1].strip().split()

    __torch_function_map__ = {k: '__torch_function_ele_wise_func__' for k in __torch_function_keys__(__torch_function_ele_wise_func__)}
    __torch_function_map__.update({k: '__torch_function_resizing_func__' for k in __torch_function_keys__(__torch_function_resizing_func__)})
    __torch_function_map__.update({k: '__torch_function_resizing_as_func__' for k in __torch_function_keys__(__torch_function_resizing_as_func__)})
    __torch_function_map__.update({k: '__torch_function_randint_func__' for k in __torch_function_keys__(__torch_function_randint_func__)})
    __torch_function_map__.update({k: '__torch_function_transpose_func__' for k in __torch_function_keys__(__torch_function_transpose_func__)})
    __torch_function_map__.update({k: '__torch_function_permute_func__' for k in __torch_function_keys__(__torch_function_permute_func__)})
    __torch_function_map__.update({k: '__torch_function_matmul_func__' for k in __torch_function_keys__(__torch_function_matmul_func__)})
    __torch_function_map__.update({k: '__torch_function_addmm_func__' for k in __torch_function_keys__(__torch_function_addmm_func__)})
    __torch_function_map__.update({k: '__torch_function_reducing_func__' for k in __torch_function_keys__(__torch_function_reducing_func__)})
    __torch_function_map__.update({k: '__torch_function_expanding_func__' for k in __torch_function_keys__(__torch_function_expanding_func__)})
    __torch_function_map__.update({k: '__torch_function_flatten_func__' for k in __torch_function_keys__(__torch_function_flatten_func__)})
    __torch_function_map__.update({k: '__torch_function_getitem_func__' for k in __torch_function_keys__(__torch_function_getitem_func__)})
    __torch_function_map__.update({k: '__torch_function_concatenate_func__' for k in __torch_function_keys__(__torch_function_concatenate_func__)})
    __torch_function_map__.update({k: '__torch_function_stack_func__' for k in __torch_function_keys__(__torch_function_stack_func__)})

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):

        if len(args) == 0: return super().__torch_function__(func, types, args, kwargs)

        sfunc = str(func)
        if sfunc.startswith('<attribute') or sfunc.startswith('<property'):
            return super().__torch_function__(func, types, args, kwargs)

        func_name = sfunc.split(' of ')[0].split(' at ')[0].split()[-1].strip("'").split('.')[-1]
        if func_name in ('__get__', '__set__', '__delete__'):
            return super().__torch_function__(func, types, args, kwargs)

        self = args[0]
        types = tuple(cls if t in [torch.nn.Parameter, tp.nn.Parameter] else t for t in types)
        torch_func_name = Tensor.__torch_function_map__.get(func_name, None)
        if isinstance(self, Tensor) and self.init and self.has_special: pass
        elif torch_func_name in ('__torch_function_resizing_func__', '__torch_function_resizing_as_func__', '__torch_function_randint_func__'): pass
        else:
            with torch._C.DisableTorchFunction():
                ret = super().__torch_function__(func, types, args, kwargs)
                collection = []
            def apply(r): r.special_from_()
            return Tensor.__torch_function_convert_apply__(ret, apply, cls)

        if torch_func_name is None: return Tensor.__torch_function_default_func__(func, types, args, kwargs)
        else: return eval(f"Tensor.{torch_func_name}")(func, types, args, kwargs)


for func in '''
zeros ones
rand randn
'''.split():
    __all__.extend([func, func+'_like'])
    exec(f"""

def {func}(*args, **kwargs):

    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, torch.Tensor):
            return torch.{func}_like(arg, **kwargs).as_subclass(Tensor).special_from_(arg).autodevice_()
        elif isinstance(arg, (tuple, list)):
            size = Size(*arg)
            return torch.{func}(size, **kwargs).as_subclass(Tensor).special_from_(size).autodevice_()
        else:
            return torch.{func}(*args, **kwargs).as_subclass(Tensor).special_from_().autodevice_()

    size = Size(*args)
    return torch.{func}(size, **kwargs).as_subclass(Tensor).special_from_(size).autodevice_()

def {func}_like(tensor, **kwargs):
    return {func}(tensor, **kwargs)
""")

for func in '''
range arange
'''.split():
    __all__.append(func)
    exec(f"""
def {func}(*args, **kwargs):
    return torch.{func}(*args, **kwargs).as_subclass(Tensor).special_from_().autodevice_()
""")

class _Randint:

    def __init__(self):
        self.range = (0, 2)

    def __getitem__(self, t):
        if len(t) == 0: t = (0, 2)
        elif len(t) == 1: t = (0, t[0])
        elif len(t) > 2: raise TypeError(f"Please use randint[lower, upper] to specify the range with upper end excluded. ")
        self.range = t
        return self

    def __call__(self, *size, **kwargs):
        return torch.randint(self.range[0], self.range[1], Size(size), **kwargs).as_subclass(Tensor).special_from_(size).autodevice_()

class _Randint_like:

    def __init__(self):
        self.range = (0, 2)

    def __getitem__(self, t):
        if len(t) == 0: t = (0, 2)
        elif len(t) == 1: t = (0, t[0])
        elif len(t) > 2: raise TypeError(f"Please use randint_like[lower, upper] to specify the range with upper end excluded. ")
        self.range = t
        return self

    def __call__(self, data, **kwargs):
        return torch.randint_like(data, self.range[0], self.range[1], **kwargs).as_subclass(Tensor).special_from_(data.shape).autodevice_()

randint = _Randint()
randint_like = _Randint_like()
__all__.extend(["randint", "randint_like"])

__all__.extend(["eye", "cat", "stack", "t", "unsqueeze", "tensor"])

def eye(*size: tuple, **kwargs):
    if len(size) == 1 and isinstance(size, (tuple, list)): size = size[0]

    size = Size(size)
    if size.nspace < 1: raise TypeError("Empty size not valid for 'eye'. ")
    if size.nspace == 1: size = size + (size.space[0],)
    if size.nspace > 2: raise TypeError("No more than 2-D is allowed for 'eye'. ")
    n = MIN(*size.space)
    s = [slice(None)] * size.ndim
    for i in RANGE(size.ndim):
        if i not in size.special:
            s[i] = torch.arange(n)
    out = zeros(size, **kwargs)
    out[tuple(s)] = 1
    return out

def cat(*list_of_tensors, dim=None, **kwargs):
    if dim is None:
        if len(list_of_tensors) > 1 and not isinstance(list_of_tensors[-1], Tensor):
            dim = list_of_tensors[-1]
            list_of_tensors =list_of_tensors[:-1]
        else: dim = 0
    if len(list_of_tensors) == 1 and isinstance(list_of_tensors[0], (tuple, list)):
        list_of_tensors = list_of_tensors[0]
    ibatch = ichannel = None
    if isinstance(dim, list): dim = dim[0]; ibatch = dim
    if isinstance(dim, set): dim = dim.pop(); ichannel = dim
    self = [t for t in list_of_tensors if isinstance(t, Tensor)]
    if len(self) == 0: return torch.cat(list_of_tensors, dim, **kwargs).as_subclass(Tensor).special_from_().set_special_(batch_dim=ibatch, channel_dim=ichannel)
    return torch.cat(list_of_tensors, dim, **kwargs).as_subclass(Tensor).special_from_(self[0]).set_special_(batch_dim=ibatch, channel_dim=ichannel)

def stack(*list_of_tensors, dim=None, **kwargs):
    if dim is None:
        if len(list_of_tensors) > 1 and not isinstance(list_of_tensors[-1], Tensor):
            dim = list_of_tensors[-1]
            list_of_tensors =list_of_tensors[:-1]
        else: dim = 0
    if len(list_of_tensors) == 1 and isinstance(list_of_tensors[0], (tuple, list)):
        list_of_tensors = list_of_tensors[0]
    ibatch = ichannel = None
    if isinstance(dim, list): dim = dim[0]; ibatch = dim
    if isinstance(dim, set): dim = dim.pop(); ichannel = dim
    self = [t for t in list_of_tensors if isinstance(t, Tensor)]
    if len(self) == 0: return torch.stack(list_of_tensors, dim, **kwargs).as_subclass(Tensor).special_from_().set_special_(batch_dim=ibatch, channel_dim=ichannel)
    return torch.stack(list_of_tensors, dim, **kwargs).as_subclass(Tensor).set_special_(special=[x + 1 if x >= dim else x for x in self[0]._special], bf=self[0]._batch_first).set_special_(batch_dim=ibatch, channel_dim=ichannel)

def t(tensor: Array.Torch):
    if isinstance(tensor, Tensor): return tensor.T
    else: return torch.t(tensor).as_subclass(Tensor).special_from_(tensor)

def squeeze(tensor, *args, **kwargs):
    if isinstance(tensor, Tensor): return tensor.squeeze(*args, **kwargs)
    else: return torch.squeeze(tensor, *args, **kwargs).as_subclass(Tensor).special_from_(tensor)

def unsqueeze(tensor, *args, **kwargs):
    if isinstance(tensor, Tensor): return tensor.unsqueeze(*args, **kwargs)
    else: return torch.unsqueeze(tensor, *args, **kwargs).as_subclass(Tensor).special_from_(tensor)

def tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False):
    if device is None and _auto_device is True:
        device = Device
    if isinstance(data, Tensor):
        return data.clone()
    if isinstance(data, torch.Tensor):
        return data.as_subclass(Tensor).special_from_()
    return torch.tensor(data, dtype=dtype, device=device, requires_grad=requires_grad, pin_memory=pin_memory).as_subclass(Tensor).special_from_()

# print(torch_type_list - (torch_type_list - globals()))
# print(torch_dtype_list - (torch_dtype_list - globals()))
# print(torch_func_list - (torch_func_list - globals()))

for key in torch_dtype_list:
    if not (key in __all__ or key in globals()):
        exec(f"{key} = torch.{key}")
        __all__.append(key)

for key in torch_type_list:
    if not (key in __all__ or key in globals()):
        exec(f"{key} = torch.{key}")
        __all__.append(key)

for key in dir(torch):
    if key.startswith("_"):
        continue
    if inspect.isclass(eval(f"torch.{key}")):
        continue
    if (key in __all__ or key in globals()) and key not in {"typename"}:
        continue
    if key in torch_func_list:
        # functions
        exec(f"{key} = torch.{key}")
        __all__.append(key)
