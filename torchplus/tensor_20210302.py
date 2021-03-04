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
    set_autodevice
    unset_autodevice
    is_autodevice
""".split()

try:
    import torch
    import numpy as np
except ImportError:
    raise ImportError("'pyctlib.torchplus' cannot be used without dependency 'torch' and 'numpy'.")
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
from .device import AutoDevice as Device

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

# def totensor(x) -> 'Tensor':
#     if isinstance(x, Tensor):
#         return x
#     elif isinstance(x, torch.Tensor):
#         return x
#     elif isinstance(x, np.ndarray):
#         return torch.tensor(x)
#     elif isinstance(x, GeneratorType):
#         return torch.tensor(list(x))
#     else:
#         return torch.tensor(x)

# def tofloat(x):
#     if isinstance(x, Tensor):
#         return x._float_torch()
#     elif isinstance(x, torch.Tensor):
#         return x.float()

# class GradWrapper:
#     def __init__(self, name, gf):
#         self.gf = gf
#         self.__class__.__name__ = name
#     def __str__(self): return "<{} object at {}>".format(self.__class__.__name__, '%x'%id(self.gf))
#     __repr__ = __str__
#     def __call__(self, *args, **kwargs): return self.gf(*args, **kwargs)

SizeRep = Tuple[int, List[int][1], Set[int][1]]

class Size(tuple):

    NegSizeError = TypeError("Size cannot have negative values except -1 indicating arbitrary number. ")

    def __new__(cls, *args, **kwargs):
        if kwargs.get('force', False):
            if len(args) == 1 and not isinstance(args[0], builtins.int): args = args[0]
            return super().__new__(cls, args)
        kb, kc = 'batch_dim', 'channel_dim'
        if len(args) == 0:
            if kwargs.get(kb, None) is not None or kwargs.get(kc, None) is not None:
                raise TypeError("Cannot assign special dimensions for empty size. ")
            self = super().__new__(cls, **kwargs)
            self._batch_dimension = self._channel_dimension = None
            return self
        if len(args) == 1 and not type(args[0]) in [builtins.int, builtins.list, builtins.set]: args = args[0]
        if isarray(args): args = tuple(args)
        if isinstance(args, Size):
            if kb in kwargs: args._batch_dimension = kwargs[kb]
            if kc in kwargs: args._channel_dimension = kwargs[kc]
            return args
        if isinstance(args, GeneratorType): args = tuple(args)
        ibatch, ichannel = None, None
        new_args = []
        for i, x in enumerate(args):
            if isinstance(x, builtins.list):
                if ibatch is not None: raise TypeError("Only one batch dimension is allowed.")
                ibatch = i
            elif isinstance(x, builtins.set):
                if ichannel is not None: raise TypeError("Only one channel dimension is allowed.")
                ichannel = i
            if not isinstance(x, builtins.int):
                if len(x) > 0: x = x.pop()
                else: x = -1
            new_args.append(x)
        self = super().__new__(cls, new_args, **kwargs)
        self._batch_dimension = kwargs.get(kb, ibatch)
        self._channel_dimension = kwargs.get(kc, ichannel)
        return self

    @property
    def batch_dimension(self): return getattr(self, '_batch_dimension', None)

    @batch_dimension.setter
    @params
    def batch_dimension(self, value: [builtins.int, null]):
        self._batch_dimension = None
        if value is not None:
            if 0 <= value < self.ndim: self._batch_dimension = value
            elif value == self._channel_dimension: raise ValueError(f"batch_dimension can not be the same as channel_dimension: {value}")
            else: raise TypeError(f"batch_dimension should be a dimension index which is smaller than {self.ndim}")

    @params
    def batch_dimension_(self, value: [builtins.int, null]):
        self.batch_dimension = value
        return self

    @property
    def batch_size(self):
        if self.batch_dimension is None:
            raise ValueError("There is no batch dimension provided. ")
        return self[self.batch_dimension]

    @property
    def channel_dimension(self): return getattr(self, '_channel_dimension', None)

    @channel_dimension.setter
    @params
    def channel_dimension(self, value: [builtins.int, null]):
        self._channel_dimension = None
        if value is not None:
            if 0 <= value < self.ndim: self._channel_dimension = value
            elif value == self._batch_dimension: raise ValueError(f"channel_dimension can not be the same as batch_dimension: {value}")
            else: raise TypeError(f"channel_dimension should be a dimension index which is smaller than {self.dim()}")

    @params
    def channel_dimension_(self, value: [builtins.int, null]):
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

    def special_from_(self, other):
        self.batch_dimension = getattr(other, '_batch_dimension', None)
        self.channel_dimension = getattr(other, '_channel_dimension', None)
        return self

    @property
    def space(self):
        s = self.special
        if len(s) == 0: return self
        elif len(s) == 1: return (self[:s[0]] + self[s[0]+1:])
        return (self[:s[0]] + self[s[0]+1:s[1]] + self[s[1]+1:])

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
    def has_batch(self): return getattr(self, '_batch_dimension', None) is not None

    @property
    def has_channel(self): return getattr(self, '_channel_dimension', None) is not None

    @property
    def has_special(self): return self.has_batch or self.has_channel

    def remove_special(self):
        self.batch_dimension = None
        self.channel_dimension = None
        return self

    def copy(self): return Size(self.python_repr)

    def __add__(self, other: [tuple, 'Size']):
        if not isinstance(other, Size):
            return Size(tuple(self) + other, force=True).special_from_(self)
        if self.has_batch and other.has_batch: raise TypeError("Batch dimension conflict in addition. ")
        if self.has_channel and other.has_channel: raise TypeError("Channel dimension conflict in addition. ")
        ibatch = ichannel = None
        if self.has_batch: ibatch = self.batch_dimension
        elif other.has_batch: ibatch = other.batch_dimension + self.ndim
        if self.has_channel: ichannel = self.channel_dimension
        elif other.has_channel: ichannel = other.channel_dimension + self.ndim
        res = Size(tuple(self) + tuple(other), force=True)
        res.batch_dimension = ibatch
        res.channel_dimension = ichannel
        return res

    @params
    def __radd__(self, other: [tuple, 'Size']):
        if not isinstance(other, Size):
            other = Size(other, force=True)
        return other + self
    __iadd__ = __add__

    @params
    def __mul__(self, value: builtins.int):
        return Size(tuple(self) * value, force=True).special_from_(self)
    __imul__ = __rmul__ = __mul__

    @params
    @staticmethod
    def __op__(a: [builtins.int, tuple, 'Size'], b: [builtins.int, tuple, 'Size'], *, op):
        if not isinstance(a, Size): a = Size(a, force=True)
        if not isinstance(b, Size): b = Size(b, force=True)
        def getvalue(x, y):
            if x == -2: return y
            if y == -2: return x
            if x == -1 or y == -1: return -1
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
            c = Size((getvalue(x, y) for x, y in zip(a, b)), force=True)
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
            return Size((getvalue(x, y) for x, y in zip(a, tuple(b) * k)), force=True)
        bb = getattr(b, '_batch_dimension', -1)
        bc = getattr(b, '_channel_dimension', -1)
        if bb is None: bb = -1
        if bc is None: bc = -1
        nbatch = b[bb] if bb >= 0 else -2
        nchannel = b[bc] if bc >= 0 else -2
        b = tuple(b.space)
        b = b * k
        s = a.special
        if len(s) == 0: pass
        elif len(s) == 1: b = b[:s[0]] + (nbatch if a.has_batch else nchannel,) + b[s[0]:]
        else:
            order = s == a.bc
            b = b[:s[0]] + (nbatch if order else nchannel,) + b[s[0]:s[1]-1] + (nchannel if order else nbatch,) + b[s[1]-1:]
        b = Size(b, force=True)
        return Size.__op__(a, b, op=op)

    @params
    def __lshift__(self, other: [builtins.int, tuple, 'Size']): return self.__op__(self, other, op=lambda x, y: x + y)
    __ilshift__ = __rlshift__ = __lshift__

    @params
    def __rshift__(self, other: [builtins.int, tuple, 'Size']): return Size.__op__(self, other, op=lambda x, y: x - y)

    @params
    def __rrshift__(self, other: [builtins.int, tuple, 'Size']): return Size.__op__(other, self, op=lambda x, y: x - y)
    __irshift__ = __rshift__

    @params
    def __pow__(self, other: [builtins.int, tuple, 'Size']): return Size.__op__(self, other, op=lambda x, y: x * y)
    __ipow__ = __rpow__ = __pow__

    @params
    def __floordiv__(self, other: [builtins.int, tuple, 'Size']): return Size.__op__(self, other, op=lambda x, y: x // y)

    @params
    def __rfloordiv__(self, other: [builtins.int, tuple, 'Size']): return Size.__op__(other, self, op=lambda x, y: x // y)
    __ifloordiv__ = __floordiv__

    @params
    def __xor__(self, other: [tuple, 'Size']):
        if not isinstance(other, Size): other = Size(other, force=True)
        a, b = self, other
        if a.special == a.bc and b.special != b.bc or a.special != a.bc and b.special == b.bc:
            if a.nspace == 0 and a.ndim == 2: a = a[::-1]
            elif b.nspace == 0 and b.ndim == 2: b = b[::-1]
            else: raise RuntimeError(f"Batch and channel order not matched for {a} and {b}")
        a_batch, b_batch = a.batch_dimension, b.batch_dimension
        a_channel, b_channel = a.channel_dimension, b.channel_dimension
        if a_batch == None: a_batch = -1
        if b_batch == None: b_batch = -1
        if a_channel == None: a_channel = -1
        if b_channel == None: b_channel = -1
        a = tuple(a)
        b = tuple(b)
        a_freg, b_freg = [a], [b]
        a_range, b_range = [(0, len(a))], [(0, len(b))]
        def crack(l, r, i, j):
            x = (l[i][j],)
            if x[0] == -1: x = (1,)
            y, z = l[i][:j], l[i][j+1:]
            ry = (r[i][0], r[i][0] + j)
            rx = (r[i][0] + j, r[i][0] + j + 1)
            rz = (r[i][0] + j + 1, r[i][1])
            l[i] = z
            l.insert(i, x)
            l.insert(i, y)
            r[i] = rz
            r.insert(i, rx)
            r.insert(i, ry)
        def insert(l, r, i, j):
            y, z = l[i][:j], l[i][j:]
            ry = (r[i][0], r[i][0] + j)
            rx = (r[i][0] + j, r[i][0] + j)
            rz = (r[i][0] + j, r[i][1])
            l[i] = z
            l.insert(i, (1,))
            l.insert(i, y)
            r[i] = rz
            r.insert(i, rx)
            r.insert(i, ry)
        for i, (s, t, (s1, s2), (t1, t2)) in enumerate(zip(a_freg, b_freg, a_range, b_range)):
            if s1 <= a_batch < s2 and t1 <= b_batch <t2:
                u, v = a[a_batch], b[b_batch]
                if u != v and u not in (1, -1) and v not in (1, -1):
                    raise RuntimeError("Batch size should be the same or ±1")
                crack(a_freg, a_range, i, a_batch - s1)
                crack(b_freg, b_range, i, b_batch - t1)
                break
        for i, (s, t, (s1, s2), (t1, t2)) in enumerate(zip(a_freg, b_freg, a_range, b_range)):
            if s1 <= a_channel < s2 and t1 <= b_channel <t2:
                u, v = a[a_channel], b[b_channel]
                if u != v and u not in (1, -1) and v not in (1, -1):
                    raise RuntimeError("Channel size should be the same or ±1")
                crack(a_freg, a_range, i, a_channel - s1)
                crack(b_freg, b_range, i, b_channel - t1)
                break
        for i, (s, t, (s1, s2), (t1, t2)) in enumerate(zip(a_freg, b_freg, a_range, b_range)):
            ls = s2 - s1
            lt = t2 - t1
            if s1 <= a_batch < s2: ls -= 1
            if s1 <= a_channel < s2: ls -= 1
            if t1 <= b_batch < t2: lt -= 1
            if t1 <= b_channel < t2: lt -= 1
            if ls == lt:
                if s1 <= a_batch < s2 and not t1 <= b_batch < t2:
                    crack(a_freg, a_range, i, a_batch - s1)
                    insert(b_freg, b_range, i, b_batch - t1)
                    break
                if t1 <= b_batch < t2 and not s1 <= a_batch < s2:
                    insert(a_freg, a_range, i, a_batch - s1)
                    crack(b_freg, b_range, i, b_batch - t1)
                    break
        for i, (s, t, (s1, s2), (t1, t2)) in enumerate(zip(a_freg, b_freg, a_range, b_range)):
            ls = s2 - s1
            lt = t2 - t1
            if s1 <= a_batch < s2: ls -= 1
            if s1 <= a_channel < s2: ls -= 1
            if t1 <= b_batch < t2: lt -= 1
            if t1 <= b_channel < t2: lt -= 1
            if ls == lt:
                if s1 <= a_channel < s2 and not t1 <= b_channel < t2:
                    crack(a_freg, a_range, i, a_channel - s1)
                    insert(b_freg, b_range, i, b_channel - t1)
                    break
                if t1 <= b_channel < t2 and not s1 <= a_channel < s2:
                    insert(a_freg, a_range, i, a_channel - s1)
                    crack(b_freg, b_range, i, b_channel - t1)
                    break
        prev_len = 0
        res_batch = res_channel = None
        for i, (s, t, (s1, s2), (t1, t2)) in enumerate(zip(a_freg, b_freg, a_range, b_range)):
            ls = s2 - s1
            lt = t2 - t1
            if ls < lt:
                u, u1, u2, lu = s, s1, s2, ls
                v, v1, v2, lv = t, t1, t2, lt
                rev = False
            elif lt < ls:
                u, u1, u2, lu = t, t1, t2, lt
                v, v1, v2, lv = s, s1, s2, ls
                rev = True
            else:
                if s1 <= a_batch < s2: res_batch = prev_len + a_batch - s1
                if t1 <= b_batch < t2: res_batch = prev_len + b_batch - t1
                if s1 <= a_channel < s2: res_channel = prev_len + a_channel - s1
                if t1 <= b_channel < t2: res_channel = prev_len + b_channel - t1
                prev_len += ls
                continue
            k = lv - lu
            for offset in builtins.range(k+1):
                for p, q in zip(u, v[k-offset:]):
                    if p!=q and p not in (1, -1) and q not in (1, -1):
                        break
                else: break
            else: raise RuntimeError(f"Can not expand sizes {self} and {other}: sequences {s} and {t} do not match. ")
            if s1 <= a_batch < s2: res_batch = prev_len + a_batch - s1 + (offset if not rev else 0)
            if t1 <= b_batch < t2: res_batch = prev_len + b_batch - t1 + (offset if rev else 0)
            if s1 <= a_channel < s2: res_channel = prev_len + a_channel - s1 + (offset if not rev else 0)
            if t1 <= b_channel < t2: res_channel = prev_len + b_channel - t1 + (offset if rev else 0)
            prev_len += lv
            if ls < lt: a_freg[i] = (1,) * (k - offset) + s + (1,) * offset
            else: b_freg[i] = (1,) * (k - offset) + t + (1,) * offset
        a_res, b_res = Size(builtins.sum(a_freg, tuple()), force=True), Size(builtins.sum(b_freg, tuple()), force=True)
        a_res.batch_dimension = res_batch
        a_res.channel_dimension = res_channel
        b_res.batch_dimension = res_batch
        b_res.channel_dimension = res_channel
        return a_res, b_res

    def __getitem__(self, k):
        if isinstance(k, builtins.int): return super().__getitem__(k)
        if isinstance(k, slice):
            s = Size(super().__getitem__(k))
            if k.start is None: ks = 0
            if k.start is not None and k.start < 0: ks = k.start + self.ndim
            try:
                s.batch_dimension = self.batch_dimension - ks
                s.channel_dimension = self.channel_dimension - ks
            except: pass
            return s
        return Size(self.python_repr[k])

    @property
    def python_repr(self):
        args = list(self)
        if self.batch_dimension is not None:
            if isinstance(args[self.batch_dimension], builtins.int):
                args[self.batch_dimension] = [args[self.batch_dimension]]
        if self.channel_dimension is not None:
            if isinstance(args[self.channel_dimension], builtins.int):
                args[self.channel_dimension] = {args[self.channel_dimension]}
        return tuple(args)

    def __str__(self):
        rep = tuple(self.python_repr)
        if len(rep) == 1: rep = str(rep).replace(',', '')
        return f"torchplus.Size{rep}"
    __repr__ = __str__

class Tensor(torch.Tensor):

    @staticmethod
    def _make_subclass(cls, data, auto_device=_auto_device, requires_grad=False, device=None):
        cpu = torch.device('cpu')
        to_device = cpu
        if device is None and auto_device: to_device = Device
        if device is not None: to_device = device
        if to_device != data.device: data = data.to(to_device)
        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        if isinstance(data, cls):
            self.special_from_(data)
        return self

    def __new__(cls, *args, **kwargs):
        kb, kc = 'batch_dim', 'channel_dim'
        if len(args) == 1 and not isinstance(args[0], builtins.int): args = args[0]
        if isinstance(args, tuple) or isinstance(args, Size):
            if kwargs.get('auto_device', _auto_device):
                self = super().__new__(cls, *args, device=Device)
            else: self = super().__new__(cls, *args)
            if 'requires_grad' in kwargs: self.requires_grad = kwargs['requires_grad']
            self._batch_dimension = getattr(args, '_batch_dimension', None)
            self._channel_dimension = getattr(args, '_channel_dimension', None)
            if kb in kwargs: self._batch_dimension = kwargs[kb]
            if kc in kwargs: self._channel_dimension = kwargs[kc]
            return self
        if kwargs.get('auto_device', _auto_device):
            if isinstance(args, torch.Tensor): data = args.to(Device)
            else: data = torch.as_tensor(args, device=Device)
        else: data = torch.as_tensor(args)
        requires_grad = kwargs.get('requires_grad', data.requires_grad)
        self = super()._make_subclass(cls, data, requires_grad)
        self._batch_dimension = getattr(data, 'batch_dimension', None)
        self._channel_dimension = getattr(data, 'channel_dimension', None)
        if kb in kwargs: self._batch_dimension = kwargs[kb]
        if kc in kwargs: self._channel_dimension = kwargs[kc]
        return self

    # @classmethod
    # def __torch_function__(cls, func, types, args=(), kwargs=None):
    #     self = args[0]
    #     if type(self) != Tensor:
    #         isTensor = [type(x) == Tensor for x in args]
    #         isofTensor = [isinstance(x, Tensor) for x in args]
    #         if builtins.any(isTensor): self = args[isTensor.index(True)]
    #         elif builtins.any(isofTensor): self = args[isofTensor.index(True)]
    #         elif isinstance(args[0], tuple):
    #             isTensor = [type(x) == Tensor for x in args[0]]
    #             isofTensor = [isinstance(x, Tensor) for x in args[0]]
    #             if builtins.any(isTensor): self = args[0][isTensor.index(True)]
    #             elif builtins.any(isofTensor): self = args[0][isofTensor.index(True)]
    #             else: return super().__torch_function__(func, types, args, kwargs)
    #         else: return super().__torch_function__(func, types, args, kwargs)
    #     if func == torch.Tensor.dim:
    #         return super().__torch_function__(func, types, args, kwargs)
    #     ibatch = self.batch_dimension
    #     ichannel = self.channel_dimension
    #     ndim = self.dim()
    #     if func in [
    #         torch.reshape, torch.Tensor.reshape, torch.Tensor.view,
    #         torch.zeros, torch.ones, torch.rand, torch.randn
    #     ]:
    #         dims = args[1:]
    #         if len(dims) == 1: dims = dims[0]
    #         dims = Size(dims)
    #         args = args[:1] + (dims,)
    #     elif func in [torch.Tensor.view_as, torch.Tensor.reshape_as]:
    #         dims = args[1].shape
    #     elif func == torch.randint:
    #         dims = Size(args[3])
    #         args = args[:3] + (dims,)
    #     types = tuple(cls if t in [torch.nn.Parameter, tp.nn.Parameter] else t for t in types)
    #     ret = super().__torch_function__(func, types, args, kwargs)
    #     if isinstance(ret, type(NotImplemented)):
    #         raise NotImplementedError(f"{func} for {args} is not implemented. ")
    #     if not isinstance(self, Tensor): self = Tensor(self)
    #     def _convert(ret, cls):
    #         if isinstance(ret, torch.Tensor):
    #             ret = ret.as_subclass(cls)
    #         if isinstance(ret, (tuple, list)):
    #             # Also handles things like namedtuples
    #             ret = type(ret)(_convert(r, cls) for r in ret)
    #         return ret
    #     def _collect(r, c):
    #         if isinstance(r, (tuple, list)):
    #             for x in r: _collect(x, c)
    #         if isinstance(r, torch.Tensor): c.append(r)
    #     collection = []
    #     ret = _convert(ret, Tensor)
    #     _collect(ret, collection)
    #     for r in collection:
    #         if func in [
    #             torch.reshape, torch.Tensor.reshape, torch.Tensor.reshape_as,
    #             torch.Tensor.view, torch.Tensor.view_as,
    #             torch.zeros, torch.ones, torch.rand, torch.randn, torch.randint
    #         ]:
    #             r.special_from_(dims)
    #         if (ibatch is not None or ichannel is not None) and not r.has_special:
    #             if func in [torch.transpose, torch.Tensor.transpose, torch.Tensor.transpose_]:
    #                 dim1, dim2 = args[1:]
    #                 if dim1 < 0: dim1 += ndim
    #                 if dim2 < 0: dim2 += ndim
    #                 r._batch_dimension = ibatch
    #                 r._channel_dimension = ichannel
    #                 if ibatch == dim1: r._batch_dimension = dim2
    #                 elif ibatch == dim2: r._batch_dimension = dim1
    #                 if ichannel == dim1: r._channel_dimension = dim2
    #                 elif ichannel == dim2: r._channel_dimension = dim1
    #                 continue
    #             if func == torch.Tensor.permute:
    #                 dims = args[1:]
    #                 if ibatch is not None: r._batch_dimension = dims.index(ibatch)
    #                 if ichannel is not None: r._channel_dimension = dims.index(ichannel)
    #                 continue
    #             if func in [torch.mm, torch.bmm, torch.smm]:
    #                 self, other = args[:2]
    #                 r.special_from_(self.shape[:-1] + other.shape[-1:])
    #                 continue
    #             if func in [torch.addmm, torch.addbmm, torch.saddmm]:
    #                 self, other = args[1:]
    #                 r.special_from_(self.shape[:-1] + other.shape[-1:])
    #                 continue
    #             if ndim == len(r.ishape):
    #                 r._batch_dimension = ibatch
    #                 r._channel_dimension = ichannel
    #                 continue
    #             if func in [
    #                 torch.cummin, torch.Tensor.cummin, torch.cummax, torch.Tensor.cummax,
    #                 torch.cumsum, torch.Tensor.cumsum, torch.cumprod, torch.Tensor.cumprod,
    #                 torch.sum, torch.Tensor.sum, torch.prod, torch.Tensor.prod,
    #                 torch.min, torch.Tensor.min, torch.max, torch.Tensor.max,
    #                 torch.mean, torch.Tensor.mean,
    #                 torch.argmin, torch.Tensor.argmin, torch.argmax, torch.Tensor.argmax
    #             ]:
    #                 if len(args) > 1:
    #                     dims = args[1:]
    #                     if len(dims) == 1 and iterable(dims[0]): dims = dims[0]
    #                     if ibatch in dims: ibatch = None
    #                     if ichannel in dims: ichannel = None
    #                     if ibatch is not None: r._batch_dimension = ibatch - len([d for d in dims if 0 <= d < ibatch or d + ndim < ibatch])
    #                     if ichannel is not None: r._channel_dimension = ichannel - len([d for d in dims if 0 <= d < ichannel or d + ndim < ichannel])
    #             elif func in [torch.squeeze, torch.Tensor.squeeze, torch.Tensor.squeeze_]:
    #                 dims = args[1:]
    #                 if len(dims) == 1 and iterable(dims[0]): dims = dims[0]
    #                 if len(dims) == 0: dims = tuple(i for i, x in enumerate(self.ishape) if x == 1)
    #                 if ibatch in dims: ibatch = None
    #                 if ichannel in dims: ichannel = None
    #                 if ibatch is not None: r._batch_dimension = ibatch - len([d for d in dims if 0 <= d < ibatch or d + ndim < ibatch])
    #                 if ichannel is not None: r._channel_dimension = ichannel - len([d for d in dims if 0 <= d < ichannel or d + ndim < ichannel])
    #             elif func in [torch.unsqueeze, torch.Tensor.unsqueeze, torch.Tensor.unsqueeze_]:
    #                 dims = args[1:]
    #                 if len(dims) == 1 and iterable(dims[0]): dims = dims[0]
    #                 for d in dims:
    #                     if ibatch is not None and (0 <= d <= ibatch or d + ndim <= ibatch): ibatch += 1
    #                     if ichannel is not None and (0 <= d <= ichannel or d + ndim <= ichannel): ichannel += 1
    #                 if ibatch is not None: r._batch_dimension = ibatch
    #                 if ichannel is not None: r._channel_dimension = ichannel
    #             elif func in [torch.flatten, torch.Tensor.flatten]:
    #                 dims = args[1:]
    #                 if len(dims) == 1:
    #                     dim = dims[0]
    #                     if dim < 0: dim += ndim
    #                     if ibatch is not None and ibatch >= dim: r._batch_dimension = None
    #                     if ichannel is not None and ichannel >= dim: r._channel_dimension = None
    #                 else:
    #                     dim1, dim2 = dims
    #                     if dim1 < 0: dim1 += ndim
    #                     if dim2 < 0: dim2 += ndim
    #                     if ibatch is not None and dim1 <= ibatch <= dim2: r._batch_dimension = None
    #                     if ichannel is not None and dim1 <= ichannel <= dim2: r._channel_dimension = None
    #             elif func == torch.Tensor.__getitem__:
    #                 dims = args[1]
    #                 if not isinstance(dims, tuple): dims = (dims,)
    #                 types = [type(x) for x in dims]
    #                 if type(...) in types:
    #                     if types.count(type(...)) > 1:
    #                         raise TypeError("")
    #                     lp = dims[:types.index(type(...))]
    #                     rp = dims[types.index(type(...))+1:]
    #                 else:
    #                     lp = dims
    #                     rp = tuple()
    #                 offset = ndim - len(rp)
    #                 offset2 = r.ndim - ndim + len(lp) + len(rp) - len([d for d in lp + rp if isinstance(d, builtins.slice)])
    #                 isnormal = ''.join('1' if issubclass(t, (builtins.int, builtins.slice, type(...))) else '0' for t in types)
    #                 if '1' in isnormal.strip('1'):
    #                     if ibatch is not None:
    #                         if ibatch < len(lp) and not isinstance(lp[ibatch], builtins.slice): ibatch = None
    #                         elif ibatch >= offset and not isinstance(rp[ibatch - offset], builtins.slice): ibatch = None
    #                     if ichannel is not None:
    #                         if ichannel < len(lp) and not isinstance(lp[ichannel], builtins.slice): ichannel = None
    #                         elif ichannel >= offset and not isinstance(rp[ichannel - offset], builtins.slice): ichannel = None
    #                     if ibatch is not None:
    #                         r._batch_dimension = ibatch + offset2 - \
    #                             len([d for d in builtins.range(len(lp)) if (0 <= d < ibatch or d + ndim < ibatch) and not isinstance(lp[d], builtins.slice)]) - \
    #                             len([d for d in builtins.range(offset, ndim) if (0 <= d < ibatch or d + ndim < ibatch) and not isinstance(rp[d-offset], builtins.slice)])
    #                     if ichannel is not None:
    #                         r._channel_dimension = ichannel + offset2 - \
    #                             len([d for d in builtins.range(len(lp)) if (0 <= d < ichannel or d + ndim < ichannel) and not isinstance(lp[d], builtins.slice)]) - \
    #                             len([d for d in builtins.range(offset, ndim) if (0 <= d < ichannel or d + ndim < ichannel) and not isinstance(rp[d-offset], builtins.slice)])
    #                 else:
    #                     index = isnormal.index('0')
    #                     if ibatch is not None:
    #                         if ibatch < len(lp) and not isinstance(lp[ibatch], builtins.slice): ibatch = None
    #                         elif ibatch >= offset and not isinstance(rp[ibatch - offset], builtins.slice): ibatch = None
    #                     if ichannel is not None:
    #                         if ichannel < len(lp) and not isinstance(lp[ichannel], builtins.slice): ichannel = None
    #                         elif ichannel >= offset and not isinstance(rp[ichannel - offset], builtins.slice): ichannel = None
    #                     if ibatch is not None:
    #                         r._batch_dimension = ibatch + (offset2 if ibatch > index else 0) - \
    #                             len([d for d in builtins.range(len(lp)) if (0 <= d < ibatch or d + ndim < ibatch) and not isinstance(lp[d], builtins.slice)]) - \
    #                             len([d for d in builtins.range(offset, ndim) if (0 <= d < ibatch or d + ndim < ibatch) and not isinstance(rp[d-offset], builtins.slice)])
    #                     if ichannel is not None:
    #                         r._channel_dimension = ichannel + (offset2 if ichannel > index else 0) - \
    #                             len([d for d in builtins.range(len(lp)) if (0 <= d < ichannel or d + ndim < ichannel) and not isinstance(lp[d], builtins.slice)]) - \
    #                             len([d for d in builtins.range(offset, ndim) if (0 <= d < ichannel or d + ndim < ichannel) and not isinstance(rp[d-offset], builtins.slice)])
    #             else: raise RuntimeError(f"{func} needs to be override. ")
    #     return ret

    # __torch_function__ = torch._C._disabled_torch_function_impl

    def requires_grad_(self, mode=True):
        self.requires_grad = mode
        return self

    def refine_names(self, *args, kwargs):
        self.has_names = True
        return super().refine_names(*args, **kwargs)

    def special_from_(self, other):
        self.batch_dimension = getattr(other, '_batch_dimension', None)
        self.channel_dimension = getattr(other, '_channel_dimension', None)
        return self

    @property
    def ishape(self): return super().shape

    @property
    def shape(self):
        shape = Size(*super().shape, force=True)
        if hasattr(self, '_batch_dimension'): shape._batch_dimension = self._batch_dimension
        if hasattr(self, '_channel_dimension'): shape._channel_dimension = self._channel_dimension
        if not hasattr(self, 'has_names'):
            self.has_names = not all(x is None for x in self.names)
        if self.has_names:
            if not shape.has_batch:
                isbatch = [('batch' in x) if x else x for x in self.names]
                if builtins.any(isbatch):
                    ibatch = isbatch.index(True)
                    self.batch_dim = ibatch
                    shape.batch_dimension = ibatch
            if not shape.has_channel:
                ischannel = [('channel' in x) if x else x for x in self.names]
                if builtins.any(ischannel):
                    ichannel = ischannel.index(True)
                    self.channel_dim = ichannel
                    shape.channel_dimension = ichannel
        return shape

    @property
    def batch_dimension(self): return getattr(self, '_batch_dimension', None)

    @batch_dimension.setter
    def batch_dimension(self, value: [builtins.int, null]):
        self._batch_dimension = None
        if value is not None:
            if 0 <= value < self.ndim: self._batch_dimension = value
            elif value == self._batch_dimension: raise ValueError(f"batch_dimension can not be the same as batch_dimension: {value}")
            else: raise TypeError(f"batch_dimension should be a dimension index which is smaller than {self.dim()}")

    @params
    def batch_dimension_(self, value: [builtins.int, null]):
        self.batch_dimension = value
        return self

    @property
    def batch_size(self): return self.shape.batch_size

    @property
    def channel_dimension(self): return getattr(self, '_channel_dimension', None)

    @channel_dimension.setter
    @params
    def channel_dimension(self, value: [builtins.int, null]):
        self._channel_dimension = None
        if value is not None:
            if 0 <= value < self.ndim: self._channel_dimension = value
            elif value == self._batch_dimension: raise ValueError(f"channel_dimension can not be the same as batch_dimension: {value}")
            else: raise TypeError(f"channel_dimension should be a dimension index which is smaller than {self.dim()}")

    @params
    def channel_dimension_(self, value: [builtins.int, null]):
        self.channel_dimension = value
        return self

    @property
    def channel_size(self): return self.shape.channel_size

    @property
    def space(self): return self.shape.space
    def dim(self): return super().dim()
    @property
    def ndim(self): return super().dim()
    @property
    def nele(self): return super().numel()
    def numel(self): return super().numel()
    @property
    def nbatch(self): return self.shape.batch_size
    @property
    def nchannel(self): return self.shape.channel_size
    @property
    def nspace(self): return self.ndim - self.has_batch - self.has_channel
    @property
    def has_batch(self): return getattr(self, '_batch_dimension', None) is not None
    @property
    def has_channel(self): return getattr(self, '_channel_dimension', None) is not None
    @property
    def has_special(self): return self.has_batch or self.has_channel
    @property
    def special(self): return sorted([x for x in [self.batch_dimension, self.channel_dimension] if x is not None])

    def remove_special_(self):
        self._batch_dimension = None
        self._channel_dimension = None
        return self

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

    @params
    def size(self, *k: [builtins.int, str]):
        if len(k) == 0:
            return self.shape
        i = [(self.names.index(x) if x in self.names else None) if isoftype(x, str) else x for x in k]
        if None in i:
            return super().size(k[i.index(None)])
        if len(i) == 1:
            return self.ishape[i[0]]
        return tuple(self.ishape[x] for x in i)

    @params
    def unsqueeze(self, *dims: int, dim=None):
        if len(dims) > 0 and dim is not None:
            raise TypeError("unsqueeze function only accept either argument or positional argument. But both are given")
        if dim is None:
            dim = dims
        if isinstance(dim, builtins.int):
            dim = (dim,)
        if len(dim) == 0: dim = (0,)
        for d in dim:
            self = super(torch.Tensor, self).unsqueeze(d)
        return self

    @params
    def unsqueeze_(self, *dims:int, dim=None):
        if len(dims) > 0 and dim is not None:
            raise TypeError("unsqueeze function only accept either argument or positional argument. But both are given")
        if dim is None:
            dim = dims
        if isinstance(dim, builtins.int):
            dim = (dim,)
        if len(dim) == 0: dim = (0,)
        for d in dim: super(torch.Tensor, self).unsqueeze_(d)
        return self

    def expand_to(self, *target):
        if len(target) == 1 and not isinstance(target[0], builtins.int): target = target[0]
        if isinstance(target, torch.Tensor): target = target.shape
        if not isinstance(target, Size): target = Size(target)
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
                while p < target.ndim and target[p] not in (1, -1) and target[p] != s: p += 1
                axis_map[i] = p
                p += 1
            if p >= target.ndim  + 1: raise TypeError(f"Unable to expand sizes {self.shape} to {target}. ")
        return self.unsqueeze_to(target, axis_map).repeat(tuple(1 if i in axis_map else (x if x >= 0 else 1) for i, x in enumerate(target)))

    @overload
    def unsqueeze_to(self, target: Array | 'Tensor', axis_place: List):
        return self.expand_to(target.shape, axis_place)

    @overload
    def unsqueeze_to(self, target: Tuple[IntScalar] | 'Size', axis_place: List):
        target = Size(target)
        if target.has_batch and self.has_batch and axis_place[self.batch_dimension] != target.batch_dimension:
            raise TypeError("Conflict of batch dimension in 'unsqueeze_to'. ")
        if target.has_channel and self.has_channel and axis_place[self.channel_dimension] != target.channel_dimension:
            raise TypeError("Conflict of channel dimension in 'unsqueeze_to'. ")
        new_size = list(target)
        for i in builtins.range(len(new_size)):
            if i not in axis_place or self.shape[axis_place.index(i)] in (1, -1):
                new_size[i] = 1
        return self.view(*new_size)

    def sample(self, dim: builtins.int = None, number: builtins.int = 1, random: bool = True) -> 'Tensor':
        """
        sample(self, dim: int = self.batch_dimension, numbder: int = 1, random: bool = True) -> Tensor

        Sample a few subspaces from a given dimension.
        data.sample(2, 1, random=False) is equivalant to data[:, :, 0, ...].
        """
        if dim is None or isinstance(dim, list) and dim == []: dim = self.batch_dimension
        if dim is None or isinstance(dim, set) and dim == {}: dim = self.channel_dimension
        if dim is None: raise TypeError("Argument 'dim' needed for sampling Tensors with no special dimensions. ")
        if number < 1: raise TypeError("Argument 'number' for sampling Tensors can not be smaller than 1. ")
        sample_indices = [slice(None)] * self.dim()
        if self.shape[dim] < number: raise TypeError(f"Too many elements needed to be sampled from dimension {dim}")
        if random:
            import random
            samples = random.sample(builtins.range(self.shape[dim]), k = number)
        else: samples = list(builtins.range(number))
        sample_indices[dim] = samples if len(samples) > 1 else samples[0]
        output_tensor = Tensor(self[tuple(sample_indices)])
        output_tensor.indices = samples
        return output_tensor

    @params
    def pick(self, dim: builtins.int, index: builtins.int):
        """
        pick(self, dim, index) -> Tensor

        pick one of the item on dimension `dim` for big tensors. 
        data.pick(2, 4) is equivalent to data[:, :, 4]
        """
        if dim < 0: dim += self.ndim
        return self[(slice(None),) * dim + (index,)]

    @params
    def mvdim(self, dim1: builtins.int, dim2: builtins.int):
        """
        mvdim(self, dim1, dim2) -> Tensor

        move dim1 to dim2(specified in the targeting size)
        data of size (2, 3, 4, 5) can be transform to (2, 4, 5, 3) by data.mvdim(1, -1) or data.mvdim(1, 3)
        """
        if dim1 < 0: dim1 += self.ndim
        if dim2 < 0: dim2 += self.ndim
        if dim1 == dim2: return self
        elif dim1 < dim2: return self.unsqueeze(dim2+1).transpose(dim1, dim2+1).squeeze(dim1)
        else: return self.unsqueeze(dim2).transpose(dim1+1, dim2).squeeze(dim1+1)

    @params
    def mvdim_(self, dim1: builtins.int, dim2: builtins.int):
        """
        In-place operation for mvdim
        """
        if dim1 < 0: dim1 += self.ndim
        if dim2 < 0: dim2 += self.ndim
        if dim1 == dim2: return self
        elif dim1 < dim2: return self.unsqueeze_(dim2+1).transpose_(dim1, dim2+1).squeeze_(dim1)
        else: return self.unsqueeze_(dim2).transpose_(dim1+1, dim2).squeeze_(dim1+1)

    def cat(self, other, dim=0):
        return torch.cat((self, other), dim=dim)

    def stack(self, other, dim=0):
        return torch.cat((self, other), dim=dim)

    @property
    def T(self: 'Tensor') -> 'Tensor':
        if not self.has_special: return Tensor(super().T)
        s = self.special
        if len(s) == 1: permute_dim = tuple(builtins.range(s[0]))[::-1] + (s[0],) + tuple(builtins.range(s[0]+1, self.ndim))[::-1]
        elif len(s) == 2: permute_dim = tuple(builtins.range(s[0]))[::-1] + (s[0],) + tuple(builtins.range(s[0]+1, s[1]))[::-1] + (s[1],) + tuple(builtins.range(s[1]+1, self.ndim))[::-1]
        return self.permute(permute_dim)

    def t(self) -> 'Tensor':
        return self.T

    def t_(self) -> 'Tensor':
        """
        t_() -> Tensor

        In-place version of :meth:`~Tensor.t`
        """
        if not self.has_special: return super().t_()
        s = self.shape.special
        if len(s) == 1:
            for i in builtins.range(s[0] // 2):
                self.transpose_(i, s[0] - i - 1)
            for i in builtins.range((self.ndim - s[0] - 1) // 2):
                self.transpose_(s[0] + i + 1, self.ndim - i - 1)
        elif len(s) == 2:
            for i in builtins.range(s[0] // 2):
                self.transpose_(i, s[0] - i - 1)
            for i in builtins.range((s[1] - s[0] - 1) // 2):
                self.transpose_(s[0] + i + 1, s[1] - i - 1)
            for i in builtins.range((self.ndim - s[1] - 1) // 2):
                self.transpose_(s[1] + i + 1, self.ndim - i - 1)
        return self

    def __matmul__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], torch.Tensor):
            if not isinstance(args[0], Tensor): other = Tensor(args[0])
            else: other = args[0]
            if self.shape[-2:].has_special or other.shape[-2:].has_special:
                raise RuntimeError("'matmul' cannot operate for special dimensions. Please make sure that the last two dimension of both tensors are not batch/channel dimensions. ")
            a, b = self.shape[:-2] ^ other.shape[:-2]
            return super(Tensor, self.view(a + tuple(self.shape[-2:]))).__matmul__(other.view(b + tuple(other.shape[-2:])))
        return super().__matmul__(*args, **kwargs)

    def __op__(self, opname, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], torch.Tensor):
            if not isinstance(args[0], Tensor): other = Tensor(args[0])
            else: other = args[0]
            a, b = self.shape ^ other.shape
            return getattr(super(Tensor, self.view(a)), opname)(other.view(b)).special_from_(a)
        return getattr(super(), opname)(*args, **kwargs)

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

for func in '''
zeros ones
rand randn
'''.split():
    __all__.extend([func, func+'_like'])
    exec(f"""
@overload
def {func}(*size: SizeRep.itemtypes, **kwargs):
    return {func}(size, **kwargs)

@overload
def {func}(tensor: Array.Torch | Tensor, **kwargs):
    out = Tensor(torch.{func}_like(tensor, **kwargs), **kwargs)
    out.batch_dimension = tensor.batch_dimension
    out.channel_dimension = tensor.channel_dimension
    return out

@overload
def {func}(size: SizeRep | Size, **kwargs):
    size = Size(size)
    out = Tensor(torch.{func}(size, **kwargs), **kwargs)
    out.batch_dimension = size.batch_dimension
    out.channel_dimension = size.channel_dimension
    return out

@overload
def {func}__default__(*args, **kwargs):
    return Tensor(torch.{func}(*args, **kwargs), **kwargs)

@params
def {func}_like(tensor: Array.Torch, **kwargs):
    return {func}(tensor, **kwargs)
""")

for func in '''
range arange
'''.split():
    __all__.append(func)
    exec(f"""
def {func}(*args, **kwargs):
    return Tensor(torch.{func}(*args, **kwargs))
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
        return Tensor(torch.randint(self.range[0], self.range[1], Size(size), **kwargs)).special_from_(size)

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
        return Tensor(torch.randint_like(data, self.range[0], self.range[1], **kwargs)).special_from_(data.shape)

randint = _Randint()
randint_like = _Randint_like()
__all__.extend(["randint", "randint_like"])

__all__.extend(["eye", "cat", "stack", "t", "unsqueeze", "tensor"])

@overload
def eye(*size: SizeRep.itemtypes, **kwargs):
    return eye(size, **kwargs)

@overload
def eye(size: SizeRep | Size, **kwargs):
    size = Size(size)
    if size.nspace < 1: raise TypeError("Empty size not valid for 'eye'. ")
    if size.nspace == 1: size = size + (size.space[0],)
    if size.nspace > 2: raise TypeError("No more than 2-D is allowed for 'eye'. ")
    n = builtins.min(*size.space)
    s = [slice(None)] * size.ndim
    for i in builtins.range(size.ndim):
        if i not in size.special:
            s[i] = torch.arange(n)
    out = zeros(size, **kwargs)
    out[tuple(s)] = 1
    return out

def cat(*list_of_tensors, dim=None, **kwargs):
    if dim is None:
        dims = [isinstance(t, builtins.int) for t in list_of_tensors]
        if builtins.any(dims):
            idim = dims.index(True)
            dim = dims[idim]
            list_of_tensors = list_of_tensors[:idim]
        else: dim = 0
    if len(list_of_tensors) == 1 and isinstance(list_of_tensors[0], (tuple, list)):
        list_of_tensors = list_of_tensors[0]
    return torch.cat(list_of_tensors, dim, **kwargs)

def stack(*list_of_tensors, dim=None, **kwargs):
    if dim is None:
        dims = [isinstance(t, builtins.int) for t in list_of_tensors]
        if builtins.any(dims):
            idim = dims.index(True)
            dim = dims[idim]
            list_of_tensors = list_of_tensors[:idim]
        else: dim = 0
    if len(list_of_tensors) == 1 and isinstance(list_of_tensors[0], (tuple, list)):
        list_of_tensors = list_of_tensors[0]
    return torch.stack(list_of_tensors, dim, **kwargs)

@params
def t(tensor: Array.Torch):
    return Tensor(tensor).T

def unsqueeze(tensor, *args, **kwargs):
    return Tensor(tensor).unsqueeze(*args, **kwargs)

def tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False):
    if device is None and _auto_device is True:
        device = Device
    if isinstance(data, torch.Tensor):
        return data.as_subclass(Tensor)
    return torch.tensor(data, dtype=dtype, device=device, requires_grad=requires_grad, pin_memory=pin_memory).as_subclass(Tensor)

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
