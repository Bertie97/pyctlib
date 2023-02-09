#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package <main>
##############################
__all__ = """
    recursive_apply
    vector
    generator_wrapper
    ctgenerator
    IndexMapping
    NoDefault
    UnDefined
    OutBoundary
    chain_function
    EmptyClass
    vhelp
    fuzzy_obj
""".split()

from types import GeneratorType
from collections import Counter
from functools import wraps, reduce, partial
from zytlib.utils.touch import touch, crash, once
import copy
import numpy as np
from pyoverload import iterable
from tqdm import tqdm, trange
from rapidfuzz import fuzz
import curses
import re
import sys
import math
from typing import overload, Callable, Iterable, Union, Dict, Any, List, Tuple, Optional
import types
import traceback
import inspect
import os
from zytlib.utils.strtools import delete_surround
from zytlib.utils.wrapper import empty_wrapper, registered_method, registered_property, destory_registered_property
import os.path
import time
import pydoc
from collections.abc import Hashable
from matplotlib.axes._subplots import Axes
from zytlib.utils.utils import constant, str_type, totuple
import functools
import multiprocessing
from subprocess import Popen, PIPE
try:
    import numba as nb
    jit = nb.jit
except:
    jit = empty_wrapper

"""
Usage:
from zytlib.vector import *
from zytlib import touch
"""

def list_like(obj):
    return "__getitem__" in dir(obj) and "__len__" in dir(obj) and "__iter__" in dir(obj)

def unfold_tuple(*args, depth=0):
    if len(args) == 0:
        return tuple()
    elif len(args) == 1:
        x = args[0]
        if isinstance(x, types.GeneratorType):
            x = tuple(x)
        if not iterable(x):
            return tuple([x])
        if depth == 0:
            return tuple(x)
        else:
            return functools.reduce(lambda x, y: x + y, tuple(unfold_tuple(t, depth=depth - 1) for t in x), tuple())
    else:
        return functools.reduce(lambda x, y: x+y, tuple(unfold_tuple(t, depth=depth) for t in args), tuple())

def max_fuzz_score(x, y):
    def make_letter(index):
        return "a"
    qx = "".join(make_letter(index) for index in range(x))
    qy = "".join(make_letter(index) for index in range(y))
    return fuzz.ratio(qx, qy)

def class_name(x):
    ret = delete_surround(str(type(x)), "<class '", "'>").rpartition(".")[-1]
    if isinstance(x, (vector, set, list, tuple)):
        ret = ret + "<{}>".format(len(x))
    return ret

def raw_function(func):
    """
    if "__func__" in dir(func):
        return func.__func__
    return func
    """
    if "__func__" in dir(func):
        return func.__func__
    return func

def get_args_str(func, func_name):
    try:
        raw_code = inspect.getsource(func)
        raw_code = "\n".join(vector(raw_code.split("\n")).map(lambda x: x.strip()))
        def_index = raw_code.index("def ")
        depth = 0
        j = def_index
        start_index = def_index + 4
        double_string = False
        single_string = False
        while j < len(raw_code):
            if double_string:
                if raw_code[j] == "\\":
                    j += 2
                    continue
                if raw_code[j] == "\"":
                    double_string = False
            elif single_string:
                if raw_code[j] == "\\":
                    j += 2
                    continue
                if raw_code[j] == "'":
                    single_string = False
            elif raw_code[j] == "(":
                if depth == 0:
                    start_index = j + 1
                depth += 1
            elif raw_code[j] == ")":
                depth -= 1
                if depth == 0:
                    return "{}({})".format(func_name, raw_code[start_index: j].replace("\n", " "))
            elif raw_code[j] == "'":
                single_string = True
            elif raw_code[j] == "\"":
                double_string = True
            j += 1
        return ""
    except:
        pass
    try:
        doc = inspect.getdoc(func)
        assert doc != ""
        return doc.split("\n")[0]
    except:
        pass
    return ""

def hashable(x):
    if isinstance(x, vector):
        return x.ishashable()
    return isinstance(x, Hashable)

def _need_split_tuple(func):
    try:
        params = inspect.signature(func).parameters
        if len(params) == 1:
            if str(list(params.values())[0])[0] == "*":
                return True
            return False
        else:
            return True
    except:
        return False

class _Vector_Dict(dict):

    def values(self):
        return vector(super().values())

    def keys(self):
        return vector(super().keys())

def recursive_apply(container, func):
    if isinstance(container, vector):
        return container.map(lambda x: recursive_apply(x, func))
    if isinstance(container, list):
        return [recursive_apply(x, func) for x in container]
    if isinstance(container, tuple):
        return tuple([recursive_apply(x, func) for x in container])
    if isinstance(container, set):
        return set([recursive_apply(x, func) for x in container])
    if isinstance(container, dict):
        return {key: recursive_apply(value, func) for key, value in container.items()}
    try:
        return func(container)
    except:
        return container

class EmptyClass:

    def __init__(self, name="EmptyClass"):
        self.name = name
        pass

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

NoDefault = EmptyClass("No Default Value")
OutBoundary = EmptyClass("Out of Boundary")
UnDefined = EmptyClass("Not Defined")

def chain_function(*funcs):
    """chain_function.

    Parameters
    ----------
    funcs :
        tuple or list of function
    return :
        the composition function of funcs

    Examples:
    ----------
    f1 = lambda x: x+1
    f2 = lambda x: x**2
    g = chain_function((f1, f2))
    then g = f2(f1)
    """
    def ret(funcs, *args):
        for index, func in enumerate(funcs):
            if index == 0:
                x = func(*args)
            else:
                x = func(x)
        return x
    funcs = totuple(funcs)
    if len(funcs) == 0:
        return funcs[0]
    return partial(ret, funcs)

def slice_length(index):
    if index is None:
        return 0
    if index.step > 0:
        return (index.stop - index.start - 1) // index.step + 1
    else:
        return (index.stop - index.start + 1) // index.step + 1

def slice_complete(index, length):
    if index is None:
        return None
    if index.step is None:
        step = 1
    else:
        step = index.step
    if index.start is None:
        if step > 0:
            start = 0
        else:
            start = length - 1
    else:
        start = index.start
        if start < 0:
            start = length + start
    if index.stop is None:
        if step > 0:
            stop = length
        else:
            stop = -1
    else:
        stop = index.stop
        if stop < 0:
            stop = length + stop
        stop = min(max(stop, -1), length)
    if step == 0:
        return None
    if start < 0 and stop <= 0:
        return None
    if start >= length and stop >= length - 1:
        return None
    if (stop - start) * step <= 0:
        return None
    return slice(start, stop, step)

def slice_to_list(index: Union[slice, None], length, forward=False):
    if index is None:
        return [-1] * length
    start = index.start
    stop = index.stop
    step = index.step
    if start is None or stop is None or step is None:
        return slice_to_list(slice_complete(index, length), length, forward=forward)
    if not forward:
        temp = list()
        current_index = start
        while (stop - current_index) * step > 0:
            if 0 <= current_index < length:
                temp.append(current_index)
            current_index += step
        return temp
    else:
        ret = [-1] * length
        current_index = start
        current_nu = 0
        while (stop - current_index) * step > 0:
            if 0 <= current_index < length:
                ret[current_index] = current_nu
                current_nu += 1
            current_index += step
        return ret

@jit(nopython=True, cache=True)
def numba_cumsum(x):
    return np.cumsum(x)

@jit(nopython=True, cache=True)
def numba_exp(x):
    return np.exp(x)

@jit(nopython=True, cache=True)
def numba_cos(x):
    return np.cos(x)

@jit(nopython=True, cache=True)
def numba_sin(x):
    return np.sin(x)

@jit(nopython=True, cache=True)
def numba_log(x):
    return np.log(x)

@jit(nopython=True, cache=True)
def numba_sum(x):
    return np.sum(x)

@jit(nopython=True, cache=True)
def numba_max(x):
    return np.max(x)

@jit(nopython=True, cache=True)
def numba_min(x):
    return np.min(x)

@jit(nopython=True, cache=True)
def numba_variance(x):
    return np.var(x)

@jit(nopython=True, cache=True)
def numba_plus(x, element):
    return x + element

@jit(nopython=True, cache=True)
def numba_abs(x):
    return np.abs(x)

@jit(nopython=True, cache=True)
def numba_relu(x):
    return x * (x > 0)

@jit(nopython=True, cache=True)
def numba_clip(x, x_low, x_upper):
    shape = x.shape
    x_flatten = x.flatten()
    for i in range(len(x_flatten)):
        if x_flatten[i] < x_low:
            x_flatten[i] = x_low
        elif x_flatten[i] > x_upper:
            x_flatten[i] = x_upper
    return x_flatten.reshape(shape)

@jit(nopython=True, cache=True)
def numba_maximum(x, a):
    return np.maximum(x, a)

@jit(nopython=True, cache=True)
def numba_minimum(x, a):
    return np.minimum(x, a)

class IndexMapping:

    def __init__(self, index_map: Union[list, slice]=None, range_size: int=-1, reverse: bool=False):
        """
        Paramters:
        -----------
        index_map: list
        range_size: int
            if reverse is True, range_size means domain size
                # domain: range_size
                # range:  len(index_map)
            else, range_size means range size
                # domain: len(index_map)
                # range:  range_size
        reverse: bool
        """
        self.__isslice = False
        if index_map is None:
            assert range_size == -1
            self.__index_map = None
            self.__index_map_reverse = None
            self.__range_size = 0
            self.__domain_size = 0
            return
        if isinstance(index_map, slice):
            assert reverse is True
            assert range_size > 0
            self.slice = slice_complete(index_map, length=range_size)
            self.__index_map = None
            self.__index_map_reverse = None
            self.__domain_size = range_size
            self.__range_size = slice_length(self.slice)
            self.__isslice = True
            return
        if isinstance(index_map, vector):
            index_map = list(index_map)
        if range_size == -1:
            if len(index_map) == 0:
                range_size = 0
            else:
                range_size = max(index_map) + 1
        if not reverse:
            self.__range_size = range_size
            self.__domain_size = len(index_map)
        else:
            self.__range_size = len(index_map)
            self.__domain_size = range_size
        if not reverse:
            self.__index_map = index_map
            self.__index_map_reverse = None
        else:
            self.__index_map_reverse = index_map
            self.__index_map = None

    @property
    def isslice(self):
        return self.__isslice

    def reverse(self):
        if self.isidentity:
            return self
        elif not self.isslice:
            ret = IndexMapping()
            ret.__index_map = self._index_map_reverse
            ret.__index_map_reverse = self._index_map
            ret.__range_size = self.__domain_size
            ret.__domain_size = self.__range_size
            return ret
        else:
            ret = IndexMapping(slice_to_list(self.slice, self.domain_size), range_size=self.domain_size, reverse=True, isslice=False)
            return ret.reverse()

    @staticmethod
    def from_slice(index: slice, length, tolist=False):
        if tolist:
            return IndexMapping(slice_to_list(index, length), range_size=length, reverse=True)
        else:
            return IndexMapping(index, range_size=length, reverse=True)

    @property
    def domain_size(self) -> int:
        return self.__domain_size

    @property
    def range_size(self) -> int:
        return self.__range_size

    @property
    def _index_map(self):
        return self.__index_map

    @property
    def _index_map_reverse(self):
        if self.isslice:
            return self.slice
        return self.__index_map_reverse

    @property
    def index_map(self) -> Union[list, None]:
        if self.isslice:
            self.__index_map = slice_to_list(self.slice, length=self.domain_size, forward=True)
            return self._index_map
        if self._index_map is None and self._index_map_reverse is None:
            return None
        if self._index_map is not None:
            return self._index_map
        else:
            self.__index_map = self._reverse_mapping(self._index_map_reverse, range_size=self.domain_size)
            return self._index_map

    @property
    def index_map_reverse(self) -> Union[list, slice, None]:
        if self._index_map is None and self._index_map_reverse is None:
            return None
        if self._index_map_reverse is not None:
            return self._index_map_reverse
        else:
            self.__index_map_reverse = self._reverse_mapping(self._index_map, range_size=self.range_size)
            return self._index_map_reverse

    def map(self, other):
        assert isinstance(other, IndexMapping)
        if self.isidentity:
            return copy.deepcopy(other)
        if other.isidentity:
            return copy.deepcopy(self)
        assert self.range_size == other.domain_size
        if self.range_size == 0 or other.range_size == 0:
            return IndexMapping([], range_size=self.domain_size, reverse=True)
        if self._index_map is not None and other._index_map is not None:
            forward = True
        elif self._index_map_reverse is not None and other._index_map_reverse is not None:
            forward = False
        elif self._index_map is not None and other._index_map_reverse is not None:
            forward = True
        elif self._index_map_reverse is not None and other._index_map is not None:
            forward = False

        if forward:
            temp = [-1] * self.domain_size
            for index, to in enumerate(self.index_map):
                if 0 <= to < other.domain_size:
                    temp[index] = other.index_map[to]
            return IndexMapping(temp, range_size=other.range_size, reverse=False)
        elif self.isslice and other.isslice:
            start = self.slice.start + self.slice.step * other.slice.start
            stop = self.slice.start + self.slice.step * other.slice.stop
            step = self.slice.step * other.slice.step
            return IndexMapping(slice(start, stop, step), range_size=self.domain_size, reverse=True)
        elif self.isslice and not other.isslice:
            temp = [-1] * other.range_size
            for index, to in enumerate(other.index_map_reverse):
                if 0 <= to < self.range_size:
                    temp[index] = self.slice.start + self.slice.step * to
            return IndexMapping(temp, range_size=self.domain_size, reverse=True)
        elif not self.isslice and other.isslice:
            return IndexMapping(self.index_map_reverse[other.slice], range_size=self.domain_size, reverse=True)
        else:
            temp = [-1] * other.range_size
            for index, to in enumerate(other.index_map_reverse):
                if 0 <= to < self.range_size:
                    temp[index] = self.index_map_reverse[to]
            return IndexMapping(temp, range_size=self.domain_size, reverse=True)

    def copy(self):
        return copy.deepcopy(self)

    @staticmethod
    def _reverse_mapping(mapping, range_size=0):
        if isinstance(mapping, slice):
            return slice_to_list(mapping, range_size, forward=True)
        if len(mapping) == 0:
            return [-1] * range_size
        range_size = max(range_size, max(mapping) + 1)
        ret = [-1] * range_size
        for index, to in enumerate(mapping):
            if to == -1:
                continue
            assert ret[to] == -1
            ret[to] = index
        return ret

    @property
    def isidentity(self):
        if self.isslice:
            return self.domain_size == self.range_size and self.slice.step == 1
        return self._index_map is None and self._index_map_reverse is None

    def __str__(self):
        if self.isidentity:
            return "id()"
        if self.isslice:
            return "slice: {}".format(self.slice)
        return "->: {}\n<-: {}\n[{}] -> [{}]".format(str(self._index_map), str(self._index_map_reverse), self.domain_size, self.range_size)

    def __repr__(self):
        return self.__str__()

    def check_valid(self):
        if self._index_map is None:
            return True
        if self._index_map_reverse is None:
            return True
        for index, to in enumerate(self._index_map):
            if to != -1:
                if self._index_map_reverse[to] != index:
                    return False
        for index, to in enumerate(self._index_map):
            if to != -1:
                if self._index_map[to] != index:
                    return False
        return True

    def __getitem__(self, index):
        assert isinstance(index, int)
        if self.isidentity:
            return index
        if self.range_size == 0:
            return -1
        if self.isslice:
            return self.index_map[index]
        if self._index_map is None:
            if self.domain_size > self.range_size * 4:
                return self._index_map_reverse.index(index)
        return self.index_map[index]

    def reverse_getitem(self, index: int) -> int:
        assert isinstance(index, int)
        if index < 0 or index >= self.range_size:
            return -1
        if self.isidentity:
            return index
        if self.isslice:
            return self.slice.start + self.slice.step * index
        if self._index_map_reverse is None:
            if self.range_size > self.domain_size * 4:
                return self._index_map.index(index)
        return self.index_map_reverse[index]


class vector(list):
    """vector
    vector is actually list in python with advanced method like map, filter and reduce
    """

    @overload
    def __init__(self, list, *, recursive=False, index_mapping=IndexMapping(), allow_undefined_value=False, content_type=NoDefault, isleaf=None) -> "vector":
        ...

    @overload
    def __init__(self, tuple, *, recursive=False, index_mapping=IndexMapping(), allow_undefined_value=False, content_type=NoDefault, isleaf=None) -> "vector":
        ...

    @overload
    def __init__(self, *data, recursive=False, index_mapping=IndexMapping(), allow_undefined_value=False, content_type=NoDefault, isleaf=None) -> "vector":
        ...

    def __init__(self, *args, recursive=False, index_mapping=IndexMapping(), allow_undefined_value=False, content_type=NoDefault, str_function=None, isleaf=None):
        """__init__.

        Parameters
        ----------
        args :
            args
        recursive : bool
            recursive determine whether to convert element in vector whose type is list to vector.

        Examples
        ----------
        vec = vector(1,2,3)
        vec = vector([1,2,3])
        vec = vector((1,2,3))
        will all get [1,2,3]
        """
        self._recursive=recursive
        self.allow_undefined_value = allow_undefined_value
        self.content_type = content_type
        self.str_function = str_function
        if isinstance(isleaf, bool):
            self.isleaf = isleaf
        if isinstance(index_mapping, list):
            self._index_mapping = IndexMapping(index_mapping)
        elif isinstance(index_mapping, IndexMapping):
            self._index_mapping = index_mapping
        else:
            self._index_mapping = IndexMapping()
        if len(args) == 0:
            list.__init__(self)
        elif len(args) == 1:
            if args[0] is None:
                list.__init__(self)
            elif isinstance(args[0], np.ndarray):
                if args[0].ndim == 0:
                    list.__init__(self)
                elif args[0].ndim == 1:
                    list.__init__(self, args[0].tolist())
                else:
                    temp = vector.from_numpy(args[0])
                    list.__init__(self, temp)
            elif isinstance(args[0], vector):
                list.__init__(self, args[0])
                self._index_mapping = args[0]._index_mapping
            elif isinstance(args[0], list):
                if recursive:
                    def to_vector(array):
                        """to_vector.

                        Parameters
                        ----------
                        array :
                            array
                        """
                        if isinstance(array, list):
                            return [vector.from_list(x) for x in array]
                    temp = to_vector(args[0])
                else:
                    temp = args[0]
                list.__init__(self, temp)
            elif isinstance(args[0], ctgenerator):
                list.__init__(self, args[0])
            elif isinstance(args[0], str):
                list.__init__(self, [args[0]])
            else:
                try:
                    list.__init__(self, args[0])
                except:
                    list.__init__(self, args)
        else:
            list.__init__(self, args)

    @staticmethod
    def map_from(vectors, reduce_func) -> "vector":
        """
        map_from(vectors([[0, 1, 2, 3], [0, 1, 2, 3]]), sum) will get:
        [0, 2, 4, 6]
        """
        assert any(isinstance(x, list) for x in vectors)
        for vec in vectors:
            if isinstance(vec, list):
                length = len(vec)
                break
        vectors = vector(vectors).map(lambda x: x if isinstance(x, list) else [x] * length)
        return vector.zip(vectors).map(reduce_func, split_tuple=True)

    @property
    def index_mapping(self) -> "IndexMapping":
        """
        property of vector

        Returns
        ----------
        IndexMapping
            IndexMapping of the current vector
        """
        return touch(lambda: self._index_mapping, IndexMapping())

    def filter(self, func=None, ignore_error=True, func_self=None, register_result=None) -> "vector":
        """
        filter element in the vector with which func(x) is True

        Parameters
        ----------
        func : callable
            a function, filter condition. After filter, element with which func value is True will be left.
        ignore_error : bool
            whether to ignore Error in the filter process, default True

        Example:
        ----------
        vector([1,2,3,4,5,6]).filter(lambda x: x>3)
        will produce [4,5,6]
        """
        if register_result:
            if not hasattr(self, "_vector__filter_register"):
                self.__filter_register: Dict[Union[tuple, str], Any] = dict()
            elif register_result is True:
                if (func, func_self) in self.__filter_register:
                    return self.map_index(self.__filter_register[(func, func_self)])
            elif isinstance(register_result, str):
                if register_result in self.__filter_register:
                    return self.map_index(self.__filter_register[register_result])
        if func is None:
            return self
        if func_self is None:
            new_func = func
        else:
            input_from_self = func_self(self)
            def new_func(x):
                return func(x, input_from_self)
        try:
            if ignore_error:
                filtered_index = [index for index, a in enumerate(self) if touch(lambda: new_func(a), False)]
                index_mapping = IndexMapping(filtered_index, reverse=True, range_size=self.length)
                ret = self.map_index(index_mapping)
            else:
                filtered_index = [index for index, a in enumerate(self) if new_func(a)]
                index_mapping = IndexMapping(filtered_index, reverse=True, range_size=self.length)
                ret = self.map_index(index_mapping)
            if register_result is True:
                self.__filter_register[(func, func_self)] = index_mapping
            if isinstance(register_result, str) and register_result:
                self.__filter_register[register_result] = index_mapping
            return ret
        except Exception as e:
            error_info = str(e)
            error_trace = traceback.format_exc()
        for index, a in enumerate(self):
            if touch(lambda: new_func(a)) is None:
                try:
                    error_information = "Error info: {}. \nException raised in filter function at location {} for element {}".format(error_info, index, a)
                except:
                    error_information = "Error info: {}. \nException raised in filter function at location {} for element {}".format(error_info, index, "<unknown>")
                error_information += "\n" + "-" * 50 + "\n" + error_trace + "-" * 50
                raise RuntimeError(error_information)
        return vector()

    def filter_(self, func=None, func_self=None, ignore_error=True) -> None:
        """
        **Inplace** function: filter element in the vector with which func(x) is True

        Parameters
        ----------
        func : callable
            a function, filter condition. After filter, element with which func value is True will be left.
        ignore_error : bool
            whether to ignore Error in the filter process, default True

        Example:
        ----------
        vector([1,2,3,4,5,6]).filter(lambda x: x>3)
        will produce [4,5,6]
        """
        if func_self is None:
            new_func = func
        else:
            input_from_self = func_self(self)
            def new_func(x):
                return func(x, input_from_self)
        try:
            if ignore_error:
                filtered_index = [index for index, a in enumerate(self) if touch(lambda: new_func(a), False)]
                index_mapping = IndexMapping(filtered_index, reverse=True, range_size=self.length)
                self.map_index_(index_mapping)
                return
            filtered_index = [index for index, a in enumerate(self) if new_func(a)]
            index_mapping = IndexMapping(filtered_index, reverse=True, range_size=self.length)
            self.map_index_(index_mapping)
            return
        except Exception as e:
            error_info = str(e)
            error_trace = traceback.format_exc()
        for index, a in enumerate(self):
            if touch(lambda: new_func(a)) is None:
                try:
                    error_information = "Error info: {}. \nException raised in filter function at location {} for element {}".format(error_info, index, a)
                except:
                    error_information = "Error info: {}. \nException raised in filter function at location {} for element {}".format(error_info, index, "<unknown>")

                error_information += "\n" + "-" * 50 + "\n" + error_trace + "-" * 50
                raise RuntimeError(error_information)

    def test(self, func, *args) -> "vector":
        """
        filter element with which func will not produce Error

        Parameters
        ----------
        func : callable
        args :
            more functions

        Example:
        ----------
        vector(0,1,2,3).test(lambda x: 1/x)
        will produce [1,2,3]
        """
        if len(args) > 0:
            func = chain_function((func, *args))
        return self.filter(lambda x: touch(lambda: (func(x), True)[-1], False))

    def testnot(self, func, *args) -> "vector":
        """testnot
        filter element with which func will produce Error

        Parameters
        ----------
        func :
            func
        args :
            more function

        Example:
        ----------
        vector(0,1,2,3).testnot(lambda x: 1/x)
        will produce [0]
        """
        if len(args) > 0:
            func = chain_function((func, *args))
        return self.filter(lambda x: not touch(lambda: (func(x), True)[-1], False))

    def map_async(self, func, processes=None, split_tuple=None) -> "vector":
        if split_tuple is None:
            split_tuple = _need_split_tuple(func)
        if not split_tuple:
            with multiprocessing.Pool(processes) as pool:
                return vector(pool.map_async(func, self).get())
        else:
            with multiprocessing.Pool(processes) as pool:
                return vector(pool.starmap_async(func, self).get())

    def map(self, func: Callable, *args, func_self=None, default=NoDefault, processing_bar=False, register_result=False, split_tuple=None, filter_function=None) -> "vector":
        """
        generate a new vector with each element x are replaced with func(x)

        Parameters
        ----------
        func: callable
        args:
            more function
        func_self:
            if func_self is not None, then func_self(self) will be passed as another argument to func
            x -> func(x, func_self(self))
        default:
            default value used when func cause an error
        register_result:
            It can be True / False / <str >
            If it is set, second time you call the same map function, result will be retrieved from the buffer.
            warning:
            if register_result = True, plz not call map with lambda expression in the argument, for example:
                wrong: v.map(lambda x: x + 1, register_result=True)
                right: 1. f = lambda x: x + 1
                          v.map(f, register_result)
                       2. v.map(lambda x: x + 1, register_result="plus 1")
        filter_function:
            if filter_function(index, element) is False, map will not be executed.

        Example:
        ----------
        vector([0, 1, 2]).map(lambda x: x ** 2)
        will produce[0, 1, 4]
        """
        if func is None:
            return self
        func = vector.__hook_function(func)
        args = tuple(vector.__hook_function(x) for x in args)
        if split_tuple is None:
            split_tuple = _need_split_tuple(func)
        if register_result:
            if not hasattr(self, "_vector__map_register"):
                self.__map_register: Dict[Union[tuple, str], Any] = dict()
            elif register_result is True:
                if (func, *args, default) in self.__map_register:
                    return self.__map_register[(func, *args, default)]
            elif isinstance(register_result, str):
                if register_result in self.__map_register:
                    return self.__map_register[register_result]
        if func_self is None:
            if len(args) > 0:
                new_func = chain_function((func, *args))
            else:
                new_func = func
        else:
            input_from_self = func_self(self)
            func = chain_function((func, *args))
            def new_func(x):
                return func(x, input_from_self)
        if not isinstance(default, EmptyClass):
            if filter_function is None:
                if processing_bar:
                    if split_tuple and self.check_type(tuple):
                        ret = vector([touch(lambda: new_func(*a), default=default) for a in tqdm(super().__iter__(), total=self.length)], recursive=self._recursive, index_mapping=self.index_mapping, allow_undefined_value=self.allow_undefined_value)
                    else:
                        ret = vector([touch(lambda: new_func(a), default=default) for a in tqdm(super().__iter__(), total=self.length)], recursive=self._recursive, index_mapping=self.index_mapping, allow_undefined_value=self.allow_undefined_value)
                else:
                    if split_tuple and self.check_type(tuple):
                        ret = vector([touch(lambda: new_func(*a), default=default) for a in super().__iter__()], recursive=self._recursive, index_mapping=self.index_mapping, allow_undefined_value=self.allow_undefined_value)
                    else:
                        ret = vector([touch(lambda: new_func(a), default=default) for a in super().__iter__()], recursive=self._recursive, index_mapping=self.index_mapping, allow_undefined_value=self.allow_undefined_value)
            else:
                if processing_bar:
                    if split_tuple and self.check_type(tuple):
                        ret = vector([touch(lambda: new_func(*a), default=default) if filter_function(index, a) else a for index, a in tqdm(enumerate(super().__iter__()), total=self.length)], recursive=self._recursive, index_mapping=self.index_mapping, allow_undefined_value=self.allow_undefined_value)
                    else:
                        ret = vector([touch(lambda: new_func(a), default=default) if filter_function(index, a) else a for index, a in tqdm(enumerate(super().__iter__()), total=self.length)], recursive=self._recursive, index_mapping=self.index_mapping, allow_undefined_value=self.allow_undefined_value)
                else:
                    if split_tuple and self.check_type(tuple):
                        ret = vector([touch(lambda: new_func(*a), default=default) if filter_function(index, a) else a for index, a in enumerate(super().__iter__())], recursive=self._recursive, index_mapping=self.index_mapping, allow_undefined_value=self.allow_undefined_value)
                    else:
                        ret = vector([touch(lambda: new_func(a), default=default) if filter_function(index, a) else a for index, a in enumerate(super().__iter__())], recursive=self._recursive, index_mapping=self.index_mapping, allow_undefined_value=self.allow_undefined_value)
            if register_result is True:
                self.__map_register[(func, *args, default, filter_function)] = ret
            elif isinstance(register_result, str) and register_result:
                self.__map_register[register_result] = ret
            return ret
        try:
            if filter_function is None:
                if processing_bar:
                    if split_tuple and self.check_type(tuple):
                        ret = vector([new_func(*a) for a in tqdm(super().__iter__(), total=self.length)], recursive=self._recursive, index_mapping=self.index_mapping, allow_undefined_value=self.allow_undefined_value)
                    else:
                        ret = vector([new_func(a) for a in tqdm(super().__iter__(), total=self.length)], recursive=self._recursive, index_mapping=self.index_mapping, allow_undefined_value=self.allow_undefined_value)
                else:
                    if split_tuple and self.check_type(tuple):
                        ret = vector([new_func(*a) for a in super().__iter__()], recursive=self._recursive, index_mapping=self.index_mapping, allow_undefined_value=self.allow_undefined_value)
                    else:
                        ret = vector([new_func(a) for a in super().__iter__()], recursive=self._recursive, index_mapping=self.index_mapping, allow_undefined_value=self.allow_undefined_value)
            else:
                if processing_bar:
                    if split_tuple and self.check_type(tuple):
                        ret = vector([new_func(*a) if filter_function(index, a) else a for index, a in tqdm(enumerate(super().__iter__()), total=self.length)], recursive=self._recursive, index_mapping=self.index_mapping, allow_undefined_value=self.allow_undefined_value)
                    else:
                        ret = vector([new_func(a) if filter_function(index, a) else a for index, a in tqdm(enumerate(super().__iter__()), total=self.length)], recursive=self._recursive, index_mapping=self.index_mapping, allow_undefined_value=self.allow_undefined_value)
                else:
                    if split_tuple and self.check_type(tuple):
                        ret = vector([new_func(*a) if filter_function(index, a) else a for index, a in enumerate(super().__iter__())], recursive=self._recursive, index_mapping=self.index_mapping, allow_undefined_value=self.allow_undefined_value)
                    else:
                        ret = vector([new_func(a) if filter_function(index, a) else a for index, a in enumerate(super().__iter__())], recursive=self._recursive, index_mapping=self.index_mapping, allow_undefined_value=self.allow_undefined_value)
            if register_result is True:
                self.__map_register[(func, *args, default, filter_function)] = ret
            elif isinstance(register_result, str) and register_result:
                self.__map_register[register_result] = ret
            return ret
        except Exception as e:
            error_info = str(e)
            error_trace = traceback.format_exc()
        for index, content in self.enumerate():
            if filter_function is not None and not filter_function(index, content):
                continue
            if touch(lambda: new_func(content)) is None:
                if split_tuple and isinstance(content, tuple):
                    try:
                        new_func(*content)
                    except Exception as e:
                        print(f"map error! happend at index {index} for value {content}")
                        raise e
                else:
                    try:
                        new_func(content)
                    except Exception as e:
                        print(f"map error! happend at index {index} for value {content}")
                        raise e
        return vector()

    def map_add(self, value: Any)->"vector":
        return self.map(lambda x: x + value)

    def map_mul(self, value: Any)->"vector":
        return self.map(lambda x: x * value)

    def map_where(self, *args, default=NoDefault, processing_bar=False, register_result=False, split_tuple=None, filter_function=None) -> "vector":
        assert len(args) % 2 == 1
        args_init = vector(args).map_k(lambda x: ([vector.__hook_function(_) for _ in x]), k=2, overlap=False)
        args_last = vector.__hook_function(args[-1])
        if split_tuple is None:
            split_tuple = _need_split_tuple(args_last)
        ret = vector()
        def _f(x):
            for func, entry in args_init:
                if func(x):
                    return entry(x)
            return args_last(x)
        def _f_split(x):
            for func, entry in args_init:
                if func(x):
                    return entry(*x)
            return args_last(*x)
        if split_tuple:
            _f = _f_split
        return self.map(_f, default=default, processing_bar=processing_bar, register_result=register_result, split_tuple=split_tuple, filter_function=filter_function)

    def rmap_where(self, *args, default=NoDefault, processing_bar=False, register_result=False, split_tuple=None, filter_function=None) -> "vector":
        assert len(args) % 2 == 1
        args_init = vector(args).map_k(lambda x: ([vector.__hook_function(_) for _ in x]), k=2, overlap=False)
        args_last = vector.__hook_function(args[-1])
        if split_tuple is None:
            split_tuple = _need_split_tuple(args_last)
        ret = vector()
        def _f(x):
            for func, entry in args_init:
                if func(x):
                    return entry(x)
            return args_last(x)
        def _f_split(x):
            for func, entry in args_init:
                if func(x):
                    return entry(*x)
            return args_last(*x)
        if split_tuple:
            _f = _f_split
        return self.rmap(_f, default=default, processing_bar=processing_bar, register_result=register_result, split_tuple=split_tuple, filter_function=filter_function)

    def chunk(self, k, drop_last=True) -> "vector":
        func = int if drop_last else math.ceil
        return vector([self[i * k: (i + 1) * k] for i in range(func(self.length / k))])

    def map_k(self, func, k, overlap=True, split_tuple=None) -> "vector":
        if self.length < k:
            return vector()
        func = vector.__hook_function(func)
        if split_tuple is None:
            split_tuple = _need_split_tuple(func)
        assert k > 0
        t = vector()
        if overlap:
            for index in range(self.length - k + 1):
                t.append(super().__getitem__(slice(index, index+k)))
        else:
            index = 0
            while index <= self.length - k:
                t.append(super().__getitem__(slice(index, index+k)))
                index += k
        if split_tuple:
            return t.map(lambda x: func(*x))
        else:
            return t.map(lambda x: func(vector(x)))

    def map_(self, func: Callable, *args, func_self=None, default=NoDefault, processing_bar=False) -> None:
        """
        **Inplace function**: generate a new vector with each element x are replaced with func(x)

        Parameters
        ----------
        func : callable
        args :
            more function
        default :
            default value used when func cause an error

        Example:
        ----------
        vector([0,1,2]).map(lambda x: x ** 2)
        will produce [0,1,4]
        """
        if func is None:
            return self
        func = vector.__hook_function(func)
        args = tuple(vector.__hook_function(x) for x in args)
        if func_self is None:
            if len(args) > 0:
                new_func = chain_function((func, *args))
            else:
                new_func = func
        if func_self is not None:
            input_from_self = func_self(self)
            func = chain_function((func, *args))
            def new_func(x):
                return func(x, input_from_self)
        if not isinstance(default, EmptyClass):
            if processing_bar:
                for index in trange(self.length):
                    self[index] = touch(lambda: new_func(self[index]), default=default)
            else:
                for index in range(self.length):
                    self[index] = touch(lambda: new_func(self[index]), default=default)
            return
        try:
            if processing_bar:
                for index in trange(self.length):
                    self[index] = new_func(self[index])
            else:
                for index in range(self.length):
                    self[index] = new_func(self[index])
            return
        except Exception as e:
            error_info = str(e)
            error_trace = traceback.format_exc()
        for index, a in self.enumerate():
            if touch(lambda: new_func(a)) is None:
                try:
                    error_information = "Error info: {}. ".format(error_info) + "\nException raised in map function at location [{}] for element [{}] with function [{}] and default value [{}]".format(index, a, new_func, default)
                except:
                    error_information = "Error info: {}. ".format(error_info) + "\nException raised in map function at location [{}] for element [{}] with function [{}] and default value [{}]".format(index, "<unknown>", new_func, default)
                error_information += "\n" + "-" * 50 + "\n" + error_trace + "-" * 50
                raise RuntimeError(error_information)

    def insert_between(self, func_element=None, func_space=None) -> "vector":
        """
        x, y, z -> {1} x' {2} y' {3} z' {4}
        where
            {1} = func_space([], [x, y, z])
            {2} = func_space([x], [y, z])
            {3} = func_space([x, y], [z])
            {4} = func_space([x, y, z], [])
            x' = func_element(x, [], [y, z])
            y' = func_element(y, [x], [z])
            z' = func_element(z, [x, y], [])
        """
        ret = vector()
        if func_element is None:
            func_element = lambda x, left, right: x
        if func_space is None:
            func_space = lambda left, right: None
        if self.length == 0:
            return ret
        for index in range(self.length):
            ret.append(touch(lambda: func_space(super(vector, self).__getitem__(slice(index)), super(vector, self).__getitem__(slice(index, None)))), refuse_value=None)
            ret.append(func_element(super(vector, self).__getitem__(index), super(vector, self).__getitem__(slice(index)), super(vector, self).__getitem__(slice(index + 1, None))), refuse_value=None)
        ret.append(touch(lambda: func_space(self, vector())), refuse_value=None)
        return ret

    def rmap(self, func, *args, default=NoDefault, max_depth=-1, split_tuple=None) -> "vector":
        """rmap
        recursively map each element in vector

        Parameters
        ----------
        func :
            func
        args :
            args
        default :
            default value used when func cause an error

        Example:
        ----------
        vector([[0,1], [2,3]], recursive=True).rmap(lambda x: x+1)
        will produce [[1,2], [3,4]]
        """
        if func is None:
            return self
        func = vector.__hook_function(func)
        args = tuple(vector.__hook_function(x) for x in args)
        if split_tuple is None:
            split_tuple = _need_split_tuple(func)
        if max_depth == 0:
            return self.map(func, *args, default=default, split_tuple=split_tuple)
        if len(args) > 0:
            func = chain_function((func, *args))
        if split_tuple:
            return self.map(lambda x: x.rmap(func, default=default, max_depth=max_depth-1, split_tuple=True) if isinstance(x, vector) else func(*x), default=default)
        else:
            return self.map(lambda x: x.rmap(func, default=default, max_depth=max_depth-1, split_tuple=False) if isinstance(x, vector) else func(x), default=default)

    def replace(self, element, toelement=NoDefault) -> "vector":
        """
        replace element in vector with to element

        Parameters
        ----------
        element :
            element
        toelement :
            toelement

        Usages
        ---------
        There are three usages:
        1. replace(a, b)
            replace a with b
            vector(0,1,2,3,1).replace(1, -1)
            will produce [0,-1,2,3,-1]
        2. replace(func, b):
            replace element with which func is True with b
            vector(0,1,2,3,4).replace(lambda x: x>2, 2)
            will produce [0,1,2,2,2]
        3. replace(func, another_func):
            replace element x with which func is True with another_func(x)
            vector(0,1,2,3,4).replace(lambda x: x>2, lambda x: x+2)
            will produce [0,1,2,5,6]
        """
        ret = self.copy()
        if toelement is NoDefault:
            if callable(element):
                for index in range(self.length):
                    ret[index] = element(self[index])
            else:
                for index in range(self.length):
                    ret[index] = element
        else:
            replace_indexs = self.findall(element)
            for index in replace_indexs:
                if callable(toelement):
                    ret[index] = toelement(self[index])
                else:
                    ret[index] = toelement
        return ret

    def replace_(self, element, toelement=NoDefault) -> None:
        """
        **Inplace function**: inplace replace element in vector with to element

        Parameters
        ----------
        element :
            element
        toelement :
            toelement

        Usages
        ---------
        There are three usages:
        1. replace(a, b)
            replace a with b
            vector(0,1,2,3,1).replace(1, -1)
            will produce [0,-1,2,3,-1]
        2. replace(func, b):
            replace element with which func is True with b
            vector(0,1,2,3,4).replace(lambda x: x>2, 2)
            will produce [0,1,2,2,2]
        3. replace(func, another_func):
            replace element x with which func is True with another_func(x)
            vector(0,1,2,3,4).replace(lambda x: x>2, lambda x: x+2)
            will produce [0,1,2,5,6]
        """
        if toelement is NoDefault:
            if callable(element):
                for index in range(self.length):
                    self[index] = element(self[index])
            else:
                for index in range(self.length):
                    self[index] = element
        else:
            replace_indexs = self.findall(element)
            for index in replace_indexs:
                if callable(toelement):
                    self[index] = toelement(self[index])
                else:
                    self[index] = toelement
        return

    @staticmethod
    def stack(*args, dim=0) -> "vector":
        args = totuple(args)
        args = vector(vector(x) for x in args)
        assert args.all_equal(lambda x: x.shape)
        shape = args[0].shape
        assert dim <= len(shape)
        if dim == -1:
            dim = len(shape)
        if len(shape) == 1 and shape[0] == 1 and dim == 0:
            return args.map(lambda x: x[0])
        if dim > 0:
            ret = vector.meshrange(shape[:dim]).rmap(lambda index: vector.stack(*args.map(lambda x: x[index], split_tuple=False)), split_tuple=False)
            return ret
        return args

    def apply(self, command) -> None:
        """apply
        apply command to each element

        Parameters
        ----------
        command : Tuple[str, callable]
            command

        Returns
        -------
        None

        """
        if isinstance(command, str):
            for x in self:
                exec(command.format(x))
        else:
            for x in self:
                command(x)

    def int(self) -> "vector":
        def _int(x):
            if isinstance(x, str):
                return ord(x)
            else:
                return int(x)
        return self.rmap(_int, split_tuple=False)

    def float(self) -> "vector":
        return self.rmap(float, split_tuple=False)

    @property
    def element_type(self):
        if self.length == 0:
            return None
        if hasattr(self, "_vector__type"):
            return self.__type
        def add_element(x, y):
            x.add(type(y))
            return x
        ret = self.reduce(add_element, first=set())
        if len(ret) == 0:
            self.__type == None
        elif len(ret) == 1:
            self.__type = ret.pop()
        else:
            self.__type = ret
        return self.__type

    @property
    def element_type_recursive(self):
        if self.length == 0:
            return None
        if hasattr(self, "_vector__type_recursive"):
            return self.__type_recursive
        def add_element(x, y):
            x.add(type(y))
            return x
        ret = self.reduce(add_element, first=set(), recursive=True)
        if len(ret) == 0:
            self.__type_recursive == None
        elif len(ret) == 1:
            self.__type_recursive = ret.pop()
        else:
            self.__type_recursive = ret
        return self.__type_recursive

    def check_type(self, instance, recursive=False) -> bool:
        """check_type
        check if all the element in the vector is of type instance

        Parameters
        ----------
        instance : Type
            instance
        """
        if self.length == 0:
            return False
        if not recursive:
            if not hasattr(self, "_vector__type"):
                if not isinstance(super(vector, self).__getitem__(0), instance):
                    return False
        if not recursive:
            element_type = self.element_type
        else:
            element_type = self.element_type_recursive
        if element_type is None:
            return False
        if isinstance(element_type, set):
            for t in element_type:
                if instance not in t.__mro__:
                    return False
            return True
        return instance in element_type.__mro__

    @property
    def enumerate(self) -> "vector":
        return vector(enumerate(self))

    @staticmethod
    def __hook_function(func):
        if callable(func):
            return func
        if func is None:
            return lambda x: x
        else:
            return lambda x: func

    def __and__(self, other) -> "vector":
        if isinstance(other, vector):
            if self.length == other.length:
                return vector(zip(self, other)).map(lambda x: x[0] and x[1])
            raise RuntimeError("length of vector A [{}] isnot compatible with length of vector B [{}]".format(self.length, other.length))
        raise RuntimeError("can only support vector and vector")

    def __or__(self, other) -> "vector":
        if isinstance(other, vector):
            if self.length == other.length:
                return vector(zip(self, other)),map(lambda x: x[0] or x[1])
            raise RuntimeError("length of vector A [{}] isnot compatible with length of vector B [{}]".format(self.length, other.length))
        raise RuntimeError("can only support vector or vector")

    def __mul__(self, other) -> "vector":
        """__mul__.

        Usages
        -----------
        1. vector * n will repeat the vector n times
        2. vector * vector will zip the two vector
            vector([1,2,3]) * vector([4,5,6])
            will produce [(1,4),(2,5),(3,6)]
        """
        if isinstance(other, int):
            return vector(super().__mul__(other))
        if touch(lambda: self.check_type(tuple) and other.check_type(tuple)):
            return vector(zip(self, other)).map(lambda x: (*x[0], *x[1]))
        elif touch(lambda: self.check_type(tuple)):
            return vector(zip(self, other)).map(lambda x: (*x[0], x[1]))
        elif touch(lambda: other.check_type(tuple)):
            return vector(zip(self, other)).map(lambda x: (x[0], *x[1]))
        else:
            return vector(zip(self, other))

    @staticmethod
    def product(*args) -> "vector":
        def _product(*args, pre=None):
            if len(args) == 1:
                return vector(args[0]).map(lambda x: tuple([*pre, x]))
            if pre is None:
                pre = tuple()
            return vector(args[0]).map(lambda x: _product(*args[1:], pre=tuple([*pre, x])))
        if len(args) == 0:
            return vector()
        if len(args) == 1:
            assert isinstance(args[0], list)
            if len(args[0]) == 0:
                return vector()
            if isinstance(args[0], list):
                return vector.product(*args[0])
            return vector(args[0]).map(lambda x: (x, ))
        else:
            return _product(*args, pre=None)

    @staticmethod
    def zip(*args, index_mapping=NoDefault) -> "vector":
        args = totuple(args)
        ret = vector(zip(*args)).map(lambda x: totuple(x), split_tuple=False)
        if isinstance(index_mapping, EmptyClass):
            ret._index_mapping = args[0].index_mapping
        else:
            ret._index_mapping = index_mapping
        return ret

    @staticmethod
    def rzip(*args) -> "vector":
        """
        Usage:
        x = vector.zeros(2, 3)
        y = vector.ones(2, 3)
        vector.rzip(x, y) will get
            [[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
             [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]]
        """
        args = totuple(args)
        assert len(args) >= 1
        if not isinstance(args[0], list):
            for x in args:
                assert not isinstance(x, list)
            return args
        args = tuple(vector(x) for x in args)
        length = len(args[0])
        for x in args:
            assert len(x) == length
        ret = vector()
        for index in range(length):
            element = vector.rzip(x[index] for x in args)
            ret.append(element)
        return ret

    def zip_split(self) -> Tuple["vector"]:
        """
        Usage:
        x, y, z = vector([(1,2,3), (4,5,6)]).zip_split()
        then:
        x = [1,4]
        y = [2,5]
        z = [3,6]
        """
        return vector(zip(*self)).map(lambda x: vector(x))

    def rzip_split(self) -> Tuple["vector"]:
        """
        Usage:
        t = vector([[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
         [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]])
        x, y = vector.rzip_split(t)
        """
        mem = constant()
        ret = vector()
        for item, x in enumerate(self):
            if isinstance(x, tuple):
                mem.length = len(x)
                if len(ret) == 0:
                    ret = vector(x).map(lambda e: vector([e]))
                else:
                    ret = vector([ret[index].append(x[index]) for index in range(mem.length)])
            elif isinstance(x, vector):
                temp = x.rzip_split()
                mem.length = len(temp)
                if len(ret) == 0:
                    ret = temp.map(lambda r: vector([r]))
                else:
                    ret = vector([ret[index].append(temp[index]) for index in range(mem.length)])
            else:
                raise RuntimeError()
        return ret

    def __pow__(self, other) -> "vector":
        """__pow__.
        Cartesian Product of two vector

        Parameters
        ----------
        other :
            other

        Example
        ----------
        vector([1,2,3]) ** vector([2,3,4])
        will produce [(1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4)]

        vector([1,2,3]) ** 3
        """
        if isinstance(other, int):
            if other == 1:
                return self.map(lambda x: (x, ))
            elif other == 2:
                return vector([(i, j) for i in self for j in self])
            elif other > 2:
                return vector([(*i, j) for i in self ** (other - 1) for j in self])
            else:
                raise ValueError()

        return vector([(i, j) for i in self for j in other])

    def __add__(self, other: list) -> "vector":
        """__add__.

        Parameters
        ----------
        other : list
            other
        """
        return vector(super().__add__(other))

    def __radd__(self, left) -> "vector":
        return vector(left).__add__(self)

    def matrix_operation(self, other, op) -> "vector":
        assert self.shape == other.shape
        if self.dim == 1:
            return vector.map_from([self, other], op)
        else:
            return vector.zip(self, other).map(lambda x, y: x.matrix_operation(y, op), split_tuple=True)

    def _transform(self, element, func=None) -> "vector":
        """_transform.

        Parameters
        ----------
        element :
            element
        func :
            func
        """
        if not func:
            return element
        return func(element)

    def __eq__(self, other) -> "vector":
        """__eq__.

        Parameters
        ----------
        other :
            other
        """
        if isinstance(other, list):
            return vector(zip(self, other)).map(lambda x: x[0] == x[1])
        else:
            return self.map(lambda x: x == other)

    def __neq__(self, other) -> "vector":
        """__neq__.

        Parameters
        ----------
        other :
            other
        """
        if isinstance(self, list):
            return vector(zip(self, other)).map(lambda x: x[0] != x[1])
        else:
            return self.map(lambda x: x != other)

    def __lt__(self, element) -> "vector":
        """__lt__.

        Parameters
        ----------
        element :
            element
        """
        if isinstance(element, list):
            return vector(zip(self, element)).map(lambda x: x[0] < x[1])
        else:
            return self.map(lambda x: x < element)

    def __gt__(self, element) -> "vector":
        """__gt__.

        Parameters
        ----------
        element :
            element
        """
        if isinstance(element, list):
            return vector(zip(self, element)).map(lambda x: x[0] > x[1])
        else:
            return self.map(lambda x: x > element)

    def __le__(self, element) -> "vector":
        """__le__.

        Parameters
        ----------
        element :
            element
        """
        if isinstance(element, list):
            return vector(zip(self, element)).map(lambda x: x[0] <= x[1])
        else:
            return self.map(lambda x: x < element)

    def __ge__(self, element) -> "vector":
        """__ge__.

        Parameters
        ----------
        element :
            element
        """
        if isinstance(element, list):
            return vector(zip(self, element)).map(lambda x: x[0] >= x[1])
        else:
            return self.map(lambda x: x >= element)

    def get(self, index, default=None):
        try:
            return self[index]
        except:
            return default

    def __getitem__(self, index):
        """__getitem__.

        Parameters
        ----------
        index :
            index

        """
        if isinstance(index, int):
            return super().__getitem__(index)
        if isinstance(index, slice):
            if self.length == 0:
                return vector()
            return self.map_index(IndexMapping(index, self.length, True))
        if isinstance(index, list):
            if self.length == 0:
                return vector()
            if len(self) != len(index) and vector(index).check_type(int) and vector(index).all(lambda x: -self.length <= x < self.length):
                index = vector(index).map(lambda x: x if x >= 0 else x + self.length)
                return self.map_index(IndexMapping(index, self.length, True))
            elif len(self) == len(index):
                if vector(index).all(lambda x: 0 <= int(x) <= 1):
                    return vector(zip(self, index), recursive=self._recursive, allow_undefined_value=self.allow_undefined_value).filter(lambda x: x[1]).map(lambda x: x[0])
                elif vector(index).check_type(int) and vector(index).all(lambda x: -self.length <= x < self.length):
                    index = vector(index).map(lambda x: x if x >= 0 else x + self.length)
                    return self.map_index(IndexMapping(index, self.length, True))
            raise RuntimeError()
        if isinstance(index, np.ndarray) and len(index.shape) == 1:
            assert len(self) == len(index)
            return vector(zip(self, index), recursive=self._recursive, allow_undefined_value=self.allow_undefined_value).filter(lambda x: x[1]).map(lambda x: x[0])
        if str_type(index) == "torch.Tensor":
            import torch
            if isinstance(index, torch.Tensor) and len(index.shape) == 1:
                assert len(self) == len(index)
                return vector(zip(self, index), recursive=self._recursive, allow_undefined_value=self.allow_undefined_value).filter(lambda x: x[1]).map(lambda x: x[0])
        if isinstance(index, tuple):
            if len(index) == 0:
                return vector()
            elif len(index) == 1:
                if index[0] is None:
                    return vector([self])
                else:
                    return self[index[0]]
            else:
                if index[0] is None:
                    return vector([self[index[1:]]])
                elif index[1] is None:
                    temp = self[unfold_tuple(index[0], index[2:])]
                    return temp.map(lambda x: vector([x]))
                else:
                    if isinstance(index[0], int):
                        return self[index[0]][index[1:]]
                    else:
                        return self[index[0]].map(lambda content: content[index[1:]])
        if isinstance(index, IndexMapping):
            return self.map_index(index)
        if index is None:
            return vector([self])
        return super().__getitem__(index)

    def select_index(self, index_list) -> "vector":
        return self.map_index(IndexMapping(index_list, self.length, True))

    def getitem(self, index: int, index_mapping: IndexMapping=None, outboundary_value=OutBoundary):
        if index < 0 or index >= self.length:
            return outboundary_value
        if not index_mapping:
            return super().__getitem__(index)
        elif index_mapping.isidentity():
            return super().__getitem__(index)
        else:
            return self.getitem(index_mapping.reverse_getitem(index), outboundary_value=outboundary_value)

    def __sub__(self, other) -> "vector":
        """__sub__.

        Parameters
        ----------
        other :
            other
        """
        if isinstance(other, (list, set, vector, tuple)):
            try:
                other = set(other)
            except:
                other = list(other)
            finally:
                return self.filter(lambda x: x not in other)
        else:
            return self.filter(lambda x: x != other)

    def __setitem__(self, i, t):
        """__setitem__.

        Parameters
        ----------
        i :
            i
        t :
            t
        """
        self.clear_appendix()
        if isinstance(i, int):
            super().__setitem__(i, t)
        elif isinstance(i, slice):
            super().__setitem__(i, t)
        elif isinstance(i, tuple):
            if len(i) == 1:
                super().__setitem__(i[0], t)
            else:
                self[i[0]][i[1:]] = t
        elif isinstance(i, list):
            if all([isinstance(index, bool) for index in i]):
                if iterable(t):
                    p_index = 0
                    for value in t:
                        while i[p_index] == False:
                            p_index += 1
                        super().__setitem__(p_index, value)
                else:
                    for index in range(len(self)):
                        if i[index] == True:
                            super().__setitem__(index, t)
            elif all([isinstance(index, int) for index in i]):
                if iterable(t):
                    p_index = 0
                    for p_index, value in enumerate(t):
                        super().__setitem__(i[p_index], value)
                else:
                    for index in i:
                        super().__setitem__(index, t)
            else:
                raise TypeError("only support the following usages: \n [int] = \n [slice] = \n [list] = ")
        else:
            raise TypeError("only support the following usages: \n [int] = \n [slice] = \n [list] = ")


    def ishashable(self) -> bool:
        """ishashable.
        chech whether every element in the vector is hashable
        """
        if hasattr(self, "_vector__hashable"):
            return self.__hashable
        self.__hashable = self.all(hashable)
        return self.__hashable

    def __hash__(self):
        """__hash__.
        get the hash value of the vector if every element in the vector is hashable
        """
        if not self.ishashable():
            raise Exception("not all elements in the vector is hashable, the index of first unhashable element is %d" % self.index(lambda x: not hashable(x)))
        else:
            return hash(tuple(self))

    def combinations(self, L):
        import itertools
        return vector(itertools.combinations(self, L))

    def unique(self, key=None) -> "vector":
        """unique.
        get unique values in the vector

        Example
        ----------
        vector([1,2,3,2,3,1]).unique()
        will produce [1,2,3]
        """
        if len(self) == 0:
            return vector([], recursive=False)
        hashable = self.ishashable()
        if key is not None:
            hashable = isinstance(key(self[0]), Hashable)
        else:
            key = lambda x: x
        explored = set() if hashable else list()
        pushfunc = explored.add if hashable else explored.append
        unique_elements = list()
        for x in self:
            if key(x) not in explored:
                unique_elements.append(x)
                pushfunc(key(x))
        return vector(unique_elements, recursive=False)

    def count_all(self) -> Counter:
        """count_all.
        count all the elements in the vector, sorted by occurance time

        Example
        -----------
        vector([1,2,3,2,3,1]).count_all()
        will produce Counter({1: 2, 2: 2, 3: 2})
        """
        if len(self) == 0:
            return Counter()
        hashable = self.ishashable()
        if hashable:
            return Counter(self)
        else:
            return Counter(dict(self.unique().map(lambda x: (x, self.count(x)))))

    def count(self, *args) -> int:
        """count.

        Parameters
        ----------
        args :
            args

        Usages
        ----------
        1. count(element)
            count the occurance time of element
            vector([1,2,3,1]).count(1)
            will produce 2
        2. count(func)
            count the number of elements with will func is True
            vector([1,2,3,4,5]).count(lambda x: x%2 == 1)
            will produce 3
        """
        if len(args) == 0:
            return len(self)
        if callable(args[0]):
            return len(self.filter(args[0]))
        return super().count(args[0])

    def index(self, element) -> int:
        """index.

        Parameters
        ----------
        element :
            element

        Usages
        ----------
        1. index(element)
            get the first index of element
            vector([1,2,3,4,2,3]).index(3)
            will produce 2
        2. index(func)
            get the first index of element with which func is True
            vector([1,2,3,4,2,3]).index(lambda x: x>2)
            will produce 2
        """
        if callable(element):
            for index in range(len(self)):
                if touch(lambda: element(self[index])):
                    return index
            return -1
        else:
            return super().index(element)

    def findall(self, element) -> "vector":
        """findall.

        Parameters
        ----------
        element :
            element

        Usages:
        ---------
        1. findall(element)
            get all indexs of element
            vector([1,2,3,4,2,3]).findall(3)
            will produce [2,5]
        2. findall(func)
            get all indexs of elements with which func is True
            vector([1,2,3,4,2,3]).findall(lambda x: x>2)
            will produce [2,3,5]
        """
        if callable(element):
            return vector([index for index in range(len(self)) if touch(lambda: element(self[index]))], recursive=False)
        else:
            return vector([index for index in range(len(self)) if self[index] == element], recursive=False)

    def findall_crash(self, func) -> "vector":
        """findall_crash.
        get all indexs of elements with which func will cause an error

        Parameters
        ----------
        func :
            func

        Examples
        ----------
        vector(0,1,2,3,4,0,1).findall_crash(lambda x: 1/x)
        will produce [0,5]
        """
        assert callable(func)
        return vector([index for index in range(len(self)) if crash(lambda: func(self[index]))], recursive=False)

    def all(self, func=lambda x: x) -> bool:
        """all.
        check if all element in vector are True or all func(element) for element in vector are True

        Parameters
        ----------
        func :
            func
        """
        for t in self:
            if not touch(lambda: func(t)):
                return False
        return True

    def any(self, func=lambda x: x) -> bool:
        """any.
        check if any element in vector are True or any func(element) for element in vector are True

        Parameters
        ----------
        func :
            func
        """
        for t in self:
            if touch(lambda: func(t)):
                return True
        return False

    def max(self, key=None, with_index=False, recursive=False):
        """max.

        Parameters
        ----------
        key :
            key
        with_index :
            with_index
        """
        if len(self) == 0:
            return None
        if recursive:
            return self.flatten().max(key=key, with_index=False)
        if key is None and not with_index:
            if hasattr(self, "_vector__max"):
                return self.__max
            if self.check_type(int) or self.check_type(float):
                self.__max = numba_max(np.array(self))
                return self.__max
        m_index = 0
        m_key = self._transform(self[0], key)
        for index in range(1, len(self)):
            i_key = self._transform(self[index], key)
            if i_key > m_key:
                m_key = i_key
                m_index = index
        if key is None:
            self.__max = self[m_index]
        if with_index:
            return self[m_index], m_index
        return self[m_index]


    def min(self, key=None, with_index=False, recursive=False):
        """min.

        Parameters
        ----------
        key :
            key
        with_index :
            with_index
        """
        if len(self) == 0:
            return None
        if recursive:
            return self.flatten().min(key=key, with_index=False)
        if key is None and not with_index:
            if hasattr(self, "_vector__min"):
                return self.__min
            if self.check_type(int) or self.check_type(float):
                self.__min = numba_min(np.array(self))
                return self.__min
        m_index = 0
        m_key = self._transform(self[0], key)
        for index in range(1, len(self)):
            i_key = self._transform(self[index], key)
            if i_key < m_key:
                m_key = i_key
                m_index = index
        if with_index:
            return self[m_index], m_index
        return self[m_index]

    def map_numba_function(self, numba_function, *args) -> "vector":
        assert self.check_type(int, recursive=True) or self.check_type(float, recursive=True)
        ret = numba_function(self.to_numpy(), *args)
        if isinstance(ret, np.ndarray):
            return vector(ret)
        else:
            return ret

    def exp(self) -> "vector":
        return self.map_numba_function(numba_exp)

    def sin(self) -> "vector":
        return self.map_numba_function(numba_sin)

    def cos(self) -> "vector":
        return self.map_numba_function(numba_cos)

    def log(self) -> "vector":
        return self.map_numba_function(numba_log)

    def plus(self, element) -> "vector":
        return self.map_numba_function(numba_plus, element)

    @overload
    def clip(self, min_value) -> "vector": ...

    @overload
    def clip(self, min_value, max_value) -> "vector": ...

    @overload
    def clip(self, min_value: float=None, max_value: float=None) -> "vector": ...

    def clip(self, *args, **kwargs) -> "vector":
        min_value = max_value = None
        if len(args) >= 1:
            min_value = args[0]
        if len(args) >= 2:
            max_value = args[1]
        if len(kwargs) == 0:
            if "min_value" in kwargs:
                min_value = kwargs["min_value"]
            if "max_value" in kwargs:
                max_value = kwargs["max_value"]
        if min_value is None and max_value is None:
            return self
        if min_value is not None and max_value is not None:
            return self.map_numba_function(numba_clip, min_value, max_value)
        if min_value is not None:
            return self.map_numba_function(numba_maximum, min_value)
        if max_value is not None:
            return self.map_numba_function(numba_minimum, max_value)

    def relu(self) -> "vector":
        return self.map_numba_function(numba_relu)

    def sum(self, dim=None, *, default=None):
        """sum.

        Parameters
        ----------
        default:
            default
        """
        if dim is None or self.shape is None or self.ndim <= 1:
            if self.length == 0:
                return default
            if hasattr(self, "_vector__sum"):
                return self.__sum
            if self.check_type(int) or self.check_type(float):
                self.__sum = numba_sum(self.to_numpy())
            else:
                self.__sum = self.reduce(lambda x, y: x + y, default)
            return self.__sum
        else:
            if dim < 0:
                dim = self.ndim + dim
            assert 0 <= dim < self.ndim
            new_shape = tuple([*self.shape[:dim], *self.shape[dim + 1:]])
            ret = vector.constant_vector(None, new_shape)
            for index in self.meshgrid(new_shape):
                ret[index] = self[tuple([*index[:dim], slice(None, None, None), *index[dim:]])].sum()
            return ret

    def mean(self, dim=None, default=NoDefault):
        if self.length == 0:
            if isinstance(default, EmptyClass):
                raise TypeError("vector is empty, plz set default to prevent error")
            return default
        if dim is None:
            return self.sum() / self.length
        else:
            ret = self.sum(dim=dim)
            N = self.shape[dim]
            return ret.rmap(lambda x: x / N)

    def variance(self, default=NoDefault) -> float:
        if self.length == 0:
            if isinstance(default, EmptyClass):
                raise TypeError("vector is empty, plz set default to prevent error")
            return default
        if hasattr(self, "_vector__variance"):
            return self.__variance
        if self.check_type(int) or self.check_type(float):
            self.__variance = numba_variance(np.array(self))
        else:
            self.__variance = self.map(lambda x: x ** 2).mean() - (self.mean()) ** 2
        return self.__variance

    def std(self, default=NoDefault) -> float:
        if self.length == 0:
            if isinstance(default, EmptyClass):
                raise TypeError("vector is empty, plz set default to prevent error")
            return default
        return self.variance() ** 0.5

    @overload
    def corr(self): ...

    @overload
    def corr(self, other): ...

    def corr(self, *args) -> "vector":
        if len(args) == 1:
            x = self
            y = args[0]
            assert isinstance(y, list)
            y = vector(y)
            assert len(x) == len(y)
            xy = vector.zip(x, y).map(lambda t: t[0] * t[1])
            n = len(x)
            if x.std() == 0 or y.std() == 0:
                ret = 0
            else:
                ret = (n * xy.sum() - x.sum() * y.sum()) / math.sqrt((n * x.map(lambda x: x ** 2).sum() - x.sum() ** 2) * (n * y.map(lambda x: x**2).sum() - y.sum() ** 2))
            return ret
        elif len(args) == 0:
            if self.check_type(tuple):
                n = self.length
                if n == 0:
                    return vector()
                m = len(self[0])
                for t in self:
                    assert len(t) == m
                ret = vector.zeros(m, m)
                sx = self.zip_split()
                sy = self.zip_split()
                for ix in range(m):
                    for iy in range(m):
                        ret[ix, iy] = sx[ix].corr(sy[iy])
                return ret
            elif self.check_type(vector) and len(self.shape) == 2:
                return self.map(lambda x: tuple(x)).corr()
        else:
            return vector()

    def cumsum(self):
        """
        cumulation summation of vector
        [a_1, a_2, \ldots, a_n]
        ->
        [a_1, a_1+a_2, \ldots, a_1+a_2+\ldots+a_n]
        """
        if self.length == 0:
            return vector()
        if self.check_type(int) or self.check_type(float):
            return vector(numba_cumsum(self.to_numpy()))
        return self.cumulative_reduce(lambda x, y: x + y)

    def smooth(self, window_len=11, window='hanning') -> "vector":
        """smooth the data using a window with requested size.

        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.

        input:
            x: the input signal
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.

        output:
            the smoothed signal

        example:

        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)

        see also:

        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve

        TODO: the window parameter could be the window itself if an array instead of a string
        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
        """


        if self.length < window_len // 2 + 1:
            raise ValueError("Input vector needs to be bigger than window size.")

        if window_len<3:
            return self

        self = self.map(lambda x: float(x))

        assert window_len % 2 == 1

        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

        x = self.to_numpy()
        s = np.r_[x[window_len // 2:0:-1], x, x[-2:-(window_len // 2) - 2:-1]]
        if window == 'flat': #moving average
            w = np.ones(window_len, 'd')
        else:
            w = np.__getattribute__(window)(window_len)

        y = np.convolve(w / w.sum(), s, mode='valid')
        return vector(y)

    def norm(self, p=2):
        """
        norm of vector
        is equivalent to
        self.map(lambda x: abs(x) ** p).sum() ** (1/p)

        Parameters:
        ------------
        p: float
            p can be a positive number or 0 or "inf"
        """
        if touch(lambda: self.__norm[p], None):
            return self.__norm[p]
        if not hasattr(self, "_vector__norm"):
            self.__norm = dict()
        if p == "inf":
            self.__norm[p] = self.map(abs).max()
        elif p > 0:
            self.__norm[p] = math.pow(self.map(lambda x: math.pow(abs(x), p)).sum(), 1/p)
        elif p == 0:
            self.__norm[p] = self.count(lambda x: x != 0)
        else:
            raise TypeError("p can be a positive number or 0 or 'inf'")
        return self.__norm[p]

    def normalization(self, p=1) -> "vector":
        """
        normaize the vector using p-norm
        is equivalent to self.map(lambda x: x / self.norm(p))

        result is $\frac{x}{\|x\|_p}$

        if all element are zeros, self will be returned
        """
        norm_p = self.norm(p)
        if norm_p == 0:
            return self
        return self.map(lambda x: x / norm_p)

    def normalization_(self, p=1) -> "vector":
        """
        **inplace function:** normaize the vector using p-norm
        is equivalent to self.map(lambda x: x / self.norm(p))

        result is $\frac{x}{\|x\|_p}$
        """
        norm_p = self.norm(p)
        return self.map_(lambda x: x / self.norm(p))

    def softmax(self, beta=1) -> "vector":
        """
        softmax function

        the i-th element is $\frac{\exp(\beta a_i )}{\sum_{j} \exp(\beta a_j)}$
        """
        return self.map(lambda x, y: x - y, func_self=lambda x: x.max()).map(lambda x: math.exp(x * beta)).map(lambda x, y: x / y, func_self = lambda x: x.sum())

    def entropy(self):
        assert self.all(lambda x: 0 <= x <= 1)
        def negative_xlogx(x):
            if x == 0:
                return 0
            return - x * math.log(x)
        return self.map(negative_xlogx).sum()

    def prod(self, default=None):
        """prod.

        Parameters
        ----------
        default :
            default
        """
        return self.reduce(lambda x, y: x * y, default)

    def group_by(self, key=lambda x: x[0]) -> dict:
        """group_by.

        Parameters
        ----------
        key :
            key

        Example
        ----------
        vector([1,2], [1,3], [2,3], [2,2], [2,1], [3,1]).group_by()
        will produce {1: [[1, 2], [1, 3]], 2: [[2, 3], [2, 2], [2, 1]], 3: [[3, 1]]}
        """
        result = _Vector_Dict()
        for x in self:
            k_x = key(x)
            if k_x not in result:
                result[k_x] = vector([x], recursive=False)
            else:
                result[k_x].append(x)
        return result

    def reduce(self, func, default=NoDefault, first=NoDefault, recursive=False):
        """reduce.
        reduce the vector with func (refer to map-reduce)
        for vector(a_1, a_2, \ldots, a_n)
        define:
            $$b_1 = a_1$$

            $$b_i = func(b_{i-1}, a_i)$$
        then the result is $b_n$

        Parameters
        ----------
        func :
            func
        default :
            default value will be returned if the vector is empty

        Example
        ----------
        vector(1,2,3,4).reduce(lambda x, y: x+y)
        will produce 10
        """
        if len(self) == 0:
            if not isinstance(default, EmptyClass):
                return default
            if not isinstance(first, EmptyClass):
                return first
            return None
        if not recursive:
            if not isinstance(first, EmptyClass):
                temp = first
                for x in self:
                    temp = func(temp, x)
            else:
                temp = self[0]
                for x in self[1:]:
                    temp = func(temp, x)
            return temp
        else:
            if isinstance(first, EmptyClass):
                if isinstance(self[0], vector):
                    temp = self[0].reduce(func, first=first, recursive=True)
                else:
                    temp = self[0]
                for x in self[1:]:
                    if isinstance(x, vector):
                        temp = x.reduce(func, first=temp, recursive=True)
                    else:
                        temp = func(temp, x)
            else:
                temp = first
                for x in self:
                    if isinstance(x, vector):
                        temp = x.reduce(func, first=temp, recursive=True)
                    else:
                        temp = func(temp, x)
            return temp

    def cumulative_reduce(self, func):
        if self.length == 0:
            return vector()
        ret = vector([self[0]] * self.length)
        for index in range(1, self.length):
            ret[index] = func(ret[index-1], self[index])
        return ret

    def enumerate(self):
        """enumerate.
        equivalent to enumerate(vector)
        """
        return enumerate(self)


    def join(self, sep: str) -> str:
        return sep.join(self.map(str))

    def flatten(self, depth=-1) -> "vector":
        """flatten.
        flatten the vector

        Parameters
        ----------
        depth :
            depth

        Example
        ----------
        vector([[1,2], [3,4,5], 6]).flatten()
        will produce [1,2,3,4,5,6]
        """
        def temp_flatten(array, depth=-1):
            """temp_flatten.

            Parameters
            ----------
            array :
                array
            depth :
                depth
            """
            if depth == 0:
                return array
            if not isinstance(array, list):
                return array
            if not isinstance(array, vector):
                array = vector(array)
            if array.isleaf:
                return array
            return array.map(lambda x: temp_flatten(x, depth - 1)).reduce(lambda x, y: x + y)
        return temp_flatten(self, depth)

    def permute(self, *args) -> "vector":
        args = totuple(args)
        if len(args) <= 1:
            return self
        assert len(args) == len(self.shape)
        assert vector(args).sort() == vector.range(len(args))
        def permute_tuple(t, order):
            return tuple(t[order[index]] for index in range(len(t)))
        new_shape = permute_tuple(self.shape, args)
        ret = vector.constant_vector(None, new_shape)
        for index in vector.meshgrid(self.shape):
            ret[permute_tuple(index, args)] = self[index]
        return ret

    @property
    def T(self) -> "vector":
        return self.permute(vector.range(self.ndim)[::-1])

    def transpose(self, dim1: int, dim2: int) -> "vector":
        if dim1 == dim2:
            return self
        if dim1 > dim2:
            return self.transpose(dim2, dim1)
        if dim1 > 0:
            return self.rmap(lambda x: x.transpose(0, dim2-dim1), max_depth=dim1-1, split_tuple=False)
        shape = self.shape[:(dim2 + 1)]
        ret = vector.constant_vector(None, tuple([shape[-1], *shape[1:-1], shape[0]]))
        for index in vector.meshgrid(shape):
            ret[tuple([index[-1], *index[1:-1], index[0]])] = self[index]
        return ret

    def reshape(self, *args) -> "vector":
        """reshape.

        Parameters
        ----------
        args :
            args

        Example
        ----------
        vector(1,2,3,4,5,6).reshape(2,3)
        will produce
        [[1, 2, 3], [4, 5, 6]]
        """
        size = reduce(lambda x, y: x*y, self.shape)
        args = totuple(args)
        assert args.count(-1) <= 1
        if args.count(-1) == 1:
            if len(args) == 1:
                return self.flatten()
            args = vector(args)
            assert size % abs(reduce(lambda x, y: x*y, args)) == 0
            args.replace_(-1, size // abs(reduce(lambda x, y: x*y, args)))
            args = tuple(args)
        assert reduce(lambda x, y: x * y, args) == reduce(lambda x, y: x * y, self.shape)
        if args == self.shape:
            return self
        def _reshape(value, target_shape):
            """_reshape.

            Parameters
            ----------
            value :
                value
            target_shape :
                target_shape
            """
            if len(target_shape) == 1:
                if not isinstance(value, vector):
                    return vector(value)
                else:
                    return value
            piece_length = len(value) // target_shape[0]
            next_target_shape = target_shape[1:]
            # ret = vector(_reshape(super(vector, value).__getitem__(slice(piece_length * index, piece_length * (index + 1))), next_target_shape) for index in range(target_shape[0]))
            ret = vector(_reshape(value[piece_length * index: piece_length * (index + 1)], next_target_shape) for index in range(target_shape[0]))
            return ret
        return _reshape(self.flatten(), args)

    def generator(self):
        """generator.
        change vector to ctgenerator
        """
        return ctgenerator(self)

    @property
    def head(self):
        return self.get(0, None)

    @property
    def tail(self) -> "vector":
        return self[1:]

    @property
    def numel(self):
        """
        get the total number of no-vector element in the vector
        """
        return sum([1 if not isinstance(x, vector) else x.numel for x in self])

    @property
    def length(self) -> int:
        """length.
        length of the vector
        """
        return len(self)

    def onehot(self, max_length: int=-1, default_dict: Dict[Any, int]={}) -> "vector":
        """onehot.
        get onehot representation of the vector

        Parameters
        ----------
        max_length : int
            max dimension of onehot vector
        default_dict :
            default_dict

        Example
        ----------
        vector(["apple", "banana", "peach", "apple"]).onehot()
        will produce
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        """
        assert isinstance(default_dict, dict)
        assert isinstance(max_length, int)
        assert len(default_dict) <= max_length or max_length == -1
        value = self.count_all().keys()
        index_dict = copy.copy(default_dict)
        if max_length == -1:
            max_length = len(set(value).union(set(default_dict.keys())))
        index_table = [EmptyClass() for _ in range(max_length)]
        for key, v in enumerate(default_dict):
            index_table[v] = key
        current_index = 0
        for v in value:
            if v in default_dict:
                continue
            while current_index < max_length and not isinstance(index_table[current_index], EmptyClass):
                current_index += 1
            if current_index == max_length:
                index_dict[v] = max_length - 1
            else:
                index_table[current_index] = v
                index_dict[v] = current_index
                current_index += 1
        temp_list = self.map(lambda x: index_dict[x])
        def create_onehot_vector(index, length):
            ret = vector.zeros(length)
            ret[index] = 1.
            return ret
        return temp_list.map(lambda x: create_onehot_vector(x, max_length))

    def sort(self, key: Callable=lambda x: x, reverse: bool=False) -> "vector":
        if key == None:
            return self
        if self.length == 0:
            return self
        temp = sorted(vector.zip(self, vector.range(self.length)), key=lambda x: key(x[0]), reverse=reverse)
        index_mapping_reverse = [x[1] for x in temp]
        index_mapping = IndexMapping(index_mapping_reverse, reverse=True)
        return self.map_index(index_mapping)

    def sort_(self, key=lambda x: x, reverse: bool=False) -> None:
        if key == None:
            return
        if self.length == 0:
            return
        temp = sorted(vector.zip(self, vector.range(self.length)), key=lambda x: key(x[0]), reverse=reverse)
        index_mapping_reverse = [x[1] for x in temp]
        index_mapping = IndexMapping(index_mapping_reverse, reverse=True)
        self.map_index_(index_mapping)

    def sort_by_index(self, key=lambda index: index) -> "vector":
        """sort_by_index.
        sort vector by function of index

        Parameters
        ----------
        key :
            key

        Example
        ----------
        vector([1,2,3,4,1]).sort_by_index(key=lambda x: -x)
        will produce
        [1, 4, 3, 2, 1]
        """
        afflicated_vector = vector(key(index) for index in range(self.length)).sort()
        return self.map_index(afflicated_vector.index_mapping)

    def sort_by_index_(self, key=lambda index: index) -> "vector":
        """sort_by_index.
        **Inplace function**: sort vector by function of index

        Parameters
        ----------
        key :
            key

        Example
        ----------
        vector([1,2,3,4,1]).sort_by_index(key=lambda x: -x)
        will produce
        [1, 4, 3, 2, 1]
        """
        afflicated_vector = vector(key(index) for index in range(self.length)).sort()
        self.map_index_(afflicated_vector.index_mapping)

    def sort_by_vector(self, other, func=lambda x: x) -> "vector":
        """sort_by_vector.
        sort vector A by vector B or func(B)

        Parameters
        ----------
        other :
            other
        func :
            func

        Example
        ---------
        vector(['apple', 'banana', 'peach']).sort_by_vector([2,3,1])
        will produce
        ['peach', 'apple', 'banana']
        """
        assert isinstance(other, list)
        assert self.length == len(other)
        return self.sort_by_index(lambda index: func(other[index]))

    def sort_by_vector_(self, other, func=lambda x: x) -> "vector":
        """sort_by_vector.
        **Inplace function**: sort vector A by vector B or func(B)

        Parameters
        ----------
        other :
            other
        func :
            func

        Example
        ---------
        vector(['apple', 'banana', 'peach']).sort_by_vector([2,3,1])
        will produce
        ['peach', 'apple', 'banana']
        """
        assert isinstance(other, list)
        assert self.length == len(other)
        self.sort_by_index_(lambda index: func(other[index]))

    @staticmethod
    def from_range(shape, func) -> "vector":
        if isinstance(shape, int):
            return vector.range(shape).map(func, split_tuple=False)
        elif isinstance(shape, tuple):
            return vector.meshrange(shape).rmap(func, split_tuple=True)
        else:
            raise RuntimeError()

    @staticmethod
    def from_tensor(tensor) -> "vector":
        import torch
        if isinstance(tensor, np.ndarray):
            return tensor
        elif isinstance(tensor, torch.Tensor):
            return vector.from_numpy(tensor.detach().cpu().numpy())
        return tensor

    @staticmethod
    def from_numpy(array) -> "vector":
        """from_numpy.

        Parameters
        ----------
        array :
            array
        """
        try:
            assert isinstance(array, np.ndarray)
            if array.ndim == 1:
                return vector(array.tolist())
            else:
                return vector(list(array)).map(lambda x: vector.from_numpy(x))
        except Exception as e:
            print("warning: input isn't pure np.ndarray")
            return vector(array.tolist())

    def to_numpy(self) -> np.ndarray:
        if hasattr(self, "_vector__numpy"):
            return self.__numpy
        if self.length == 0:
            return np.array([])
        elif self.check_type(int):
            ret = np.fromiter(self, dtype=np.int64)
        elif self.check_type(float):
            ret = np.fromiter(self, dtype=np.float64)
        else:
            ret = np.array(self)
        self.__numpy = ret
        return ret

    def to_dict(self, key_func, value_func) -> Dict:
        return {key_func(x): value_func(x) for x in super().__iter__()}

    def cpu(self) -> "vector":
        import torch
        def func(x):
            if isinstance(x, torch.Tensor):
                return x.cpu()
            else:
                return x
        return self.rmap(func, split_tuple=False)

    def plot_hist(self, bins=None, range=None, density=False, color=None, edgecolor=None, alpha=None, with_pdf=False, ax: Optional[Axes]=None, title: Optional[str]=None, saved_path: Optional[str]=None):
        from matplotlib import pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        _has_ax = ax is not None
        if ax is None:
            ax = plt.gca()
            ax.clear()
        else:
            assert saved_path is None
        if not with_pdf:
            ax.hist(self, bins=bins, range=range, density=density, color=None, edgecolor=edgecolor, alpha=alpha)
        else:
            ax.hist(self, bins=bins, range=range, density=True, color=None, edgecolor=edgecolor, alpha=alpha)
            import scipy.stats as st
            xmin, xmax = ax.get_xlim()
            kde_xs = np.linspace(xmin, xmax, 300)
            kde = st.gaussian_kde(self)
            ax.plot(kde_xs, kde.pdf(kde_xs), color="k")

        if title:
            ax.set_title(title)
        if not _has_ax:
            if saved_path is not None:
                if saved_path.endswith("pdf"):
                    with PdfPages(saved_path, "w") as f:
                        plt.savefig(f, format="pdf")
                else:
                    plt.savefig(saved_path, dpi=300)
            else:
                plt.show()
        return ax

    def plot(self, x=None, ax: Optional[Axes]=None, title: Optional[str]=None, smooth: int=-1, saved_path: Optional[str]=None, legend: Optional[List[str]]=None, hline: Optional[List[str]]=None, xticks=None, xticklabels=None, xlabel=None, yticks=None, yticklabels=None, ylabel=None, xlim=None, ylim=None, marker=None, color=None, linestyle=None, **kwargs):
        """
        plot line graph for vector
        title: title of the graph
        smooth: windows size of smoothing
        saved_path: path to save the graph
        legend: list of legend string
        hline: list, can be None or contains "top" or/and "bottom", to plot a horizontal line corresponding to the biggest or smallest value
        x/yticks: list
        x/yticklabels: list[string] or (list[string], fontsize)
        x/ylabel: str or (str, fontsize)
        marker: str
        color: str
        linestyle: str
        """
        from matplotlib import pyplot as plt
        _has_ax = ax is not None
        if ax is None:
            ax = plt.gca()
            ax.clear()
        else:
            assert saved_path is None
        if self.check_type(float) or self.check_type(int):
            if x is not None:
                x = list(x)
            else:
                x = list(range(self.length))
            ax.plot(x, self.smooth(smooth), marker=marker, color=color, linestyle=linestyle, **kwargs)
        elif (self.check_type(list) or self.check_type(tuple)) and self.map(len).all_equal():
            splited_vector = self.zip_split()
            if x is not None:
                x = list(x)
            else:
                x = list(range(len(splited_vector[0])))
            if color is None:
                for sv in splited_vector:
                    ax.plot(x, sv.smooth(smooth), marker=marker, linestyle=linestyle, **kwargs)
            else:
                for index, sv in enumerate(splited_vector):
                    ax.plot(x, sv.smooth(smooth), marker=marker, color=color[index], linestyle=linestyle, **kwargs)
            if not legend:
                legend = vector.range(len(splited_vector)).map(str)
        else:
            raise ValueError
        if xlim is not None:
            ax.set_xlim(*xlim)
            xmin, xmax = xlim
        else:
            xmin, xmax = min(x), max(x)
            boundary_margin = 1 / 30 * (xmax - xmin)
            plt.xlim(xmin - boundary_margin, xmax + boundary_margin)
        if ylim is not None:
            ax.set_ylim(*ylim)
            ymin, ymax = ylim
        else:
            ymin, ymax = ax.get_ylim()

        if title:
            ax.set_title(title)
        if legend:
            ax.legend(legend)
        if xlabel:
            if isinstance(xlabel, str):
                ax.set_xlabel(xlabel)
            else:
                ax.set_xlabel(xlabel[0], fontsize=xlabel[1])
        if ylabel:
            if isinstance(ylabel, str):
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel(ylabel[0], fontsize=ylabel[1])
        if xticks:
            ax.set_xticks(xticks)
        if yticks:
            ax.set_yticks(yticks)
        if xticklabels:
            if isinstance(xticklabels, list) and isinstance(xticklabels[0], str):
                ax.set_xticklabels(xticklabels)
            else:
                ax.set_xticklabels(xticklabels[0], fontsize=xticklabels[1])
        if yticklabels:
            if isinstance(yticklabels, list) and isinstance(yticklabels[0], str):
                ax.set_yticklabels(yticklabels)
            else:
                ay.set_yticklabels(yticklabels[0], fontsize=yticklabels[1])
        for index in range(2):
            if index == 0 and touch(lambda: "top" in hline):
                h_line = self.max(recursive=True)
                text = "max: {:.4g}".format(h_line)
            elif index == 1 and touch(lambda: "bottom" in hline):
                h_line = self.min(recursive=True)
                text = "min: {:.4g}".format(h_line)
            else:
                continue
            ax.plot([-boundary_margin, self.length - 1 + boundary_margin], [h_line, h_line], "-.", linewidth=0.5, color="gray")
            ax.text(-boundary_margin / 2, h_line + (ymax - ymin) / 100, text, color="gray", fontsize=10)
        if not _has_ax:
            if saved_path is not None:
                if saved_path.endswith("pdf"):
                    with PdfPages(saved_path, "w") as f:
                        plt.savefig(f, format="pdf")
                else:
                    plt.savefig(saved_path, dpi=300)
            else:
                plt.show()
        return ax

    @staticmethod
    def from_list(array) -> "vector":
        """from_list.

        Parameters
        ----------
        array :
            array
        """
        if not isinstance(array, list):
            return array
        return vector(vector.from_list(x) for x in array)

    def tolist(self) -> list:
        ret = list()
        for item in self:
            if isinstance(item, vector):
                ret.append(item.tolist())
            else:
                ret.append(item)
        return ret

    def numpy(self) -> np.ndarray:
        return np.array(self)

    @overload
    @staticmethod
    def zeros(size: Iterable) -> "vector": ...

    @overload
    @staticmethod
    def zeros(*size) -> "vector": ...

    @staticmethod
    def zeros(*args):
        """zeros.

        Parameters
        ----------
        args :
            args
        """
        args = totuple(args)
        return vector.from_numpy(np.zeros(args))

    @staticmethod
    def constant_vector(value, *args) -> "vector":
        args = totuple(args)
        if len(args) == 0:
            return vector()
        elif len(args) == 1:
            return vector(value for _ in range(args[0]))
        else:
            return vector(vector.constant_vector(value, args[1:]) for _ in range(args[0]))

    @overload
    @staticmethod
    def ones(size: Iterable) -> "vector": ...

    @overload
    @staticmethod
    def ones(*size) -> "vector": ...

    @staticmethod
    def ones(*args):
        """ones.

        Parameters
        ----------
        args :
            args
        """
        args = totuple(args)
        ret = vector.from_numpy(np.ones(args))
        ret._shape = args
        return ret

    @staticmethod
    def linspace(low, high, nbins, endpoint=True) -> "vector":
        step = (high - low) / max(nbins - bool(endpoint), 1)
        return vector.range(nbins).map(lambda x: x * step + low)

    @staticmethod
    def logspace(low, high, nbins, endpoint=True) -> "vector":
        """
        type: endpoint[True / False]
        """
        assert low > 0 and high > 0
        low_log = math.log(low)
        high_log = math.log(high)
        ret = vector.linspace(low_log, high_log, nbins, endpoint=endpoint).map(math.exp)
        ret[0] = low
        if endpoint:
            ret[-1] = high
        return ret

    @staticmethod
    def meshgrid(*args) -> "vector":
        args = totuple(args)
        if len(args) == 0:
            return vector()
        if isinstance(args[0], int):
            if len(args) == 1:
                return vector.range(args[0]).map(lambda x: (x, ))
            return vector.meshgrid(*[vector.range(d) for d in args])
        if len(args) == 1:
            return vector(args[0]).map(lambda x: (x, ))
        import itertools
        return vector(itertools.product(*args)).map(lambda x: x)

    @staticmethod
    def rand(*args, low=0, high=1) -> "vector":
        """rand.

        Parameters
        ----------
        args :
            args
        """
        args = totuple(args)
        ret = vector.from_numpy(np.random.rand(*args) * (high - low) + low)
        ret._shape = args
        return ret

    @staticmethod
    def randint(low, high=None, size=(1, )) -> "vector":
        ret = vector.from_numpy(np.random.randint(low, high=high, size=size))
        ret._shape = totuple(size)
        return ret

    @staticmethod
    def randn(*args, mean=0, std=1, truncate_std=None) -> "vector":
        """randn.

        Parameters
        ----------
        args :
            args
        """
        args = totuple(args)
        ret = vector.from_numpy(np.random.randn(*args))
        ret._shape = args
        if truncate_std is not None:
            ret = ret.clip(- truncate_std / std, truncate_std / std)
        if mean != 0 or std != 1:
            ret = ret.rmap(lambda x: std * x + mean)
        return ret

    @staticmethod
    def multinomial(n, pvals, size=None):
        ret = np.random.multinomial(n, pvals, size)
        return vector.from_numpy(ret)

    @overload
    @staticmethod
    def range(stop) -> "vector": ...

    @overload
    def range(start, stop) -> "vector": ...

    @overload
    def range(start, stop, step) -> "vector": ...

    @staticmethod
    def range(*args):
        """range.

        Parameters
        ----------
        args :
            args
        """
        return vector(range(*args))

    @staticmethod
    def meshrange(*args) -> "vector":
        """
        vector.meshrange(3, 4) will get:
        [[(0, 0), (0, 1), (0, 2), (0, 3)],
         [(1, 0), (1, 1), (1, 2), (1, 3)],
         [(2, 0), (2, 1), (2, 2), (2, 3)]]
        """
        args = totuple(args)
        if len(args) == 0:
            return vector()
        elif len(args) == 1:
            return vector.range(args[0])
        elif len(args) == 2:
            return vector.range(args[0]).map(lambda index_1: vector.range(args[1]).map(lambda index_2: (index_1, index_2), split_tuple=False))
        else:
            return vector.range(args[0]).map(lambda index: vector.meshrange(args[1:]).rmap(lambda other: tuple([index]) + other, split_tuple=False))

    @staticmethod
    def from_randomwalk(start, transition_function, length) -> "vector":
        ret = vector([start])
        temp = start
        for index in range(length-1):
            temp = transition_function(start)
            ret.append(temp)
        return ret

    def iid(self, sample_func, length, args=()) -> "vector":
        return vector([sample_func(*args) for _ in range(length)])

    @registered_property
    def isleaf(self) -> bool:
        return all(not isinstance(_, vector) for _ in self)

    @isleaf.setter
    def isleaf(self, p: bool) -> bool:
        if not hasattr(self, "__registered_property"):
            self.__registered_property = dict()
        self.__registered_property["isleaf"] = p

    @registered_property
    def ndim(self) -> int:
        return len(self.shape)

    @registered_property
    def shape(self) -> tuple:
        """shape.
        """
        if self.isleaf:
            return (self.length, )
        if any(not isinstance(x, vector) for x in self):
            return None
        if not self.map(lambda x: x.shape).all_equal():
            return None
        if self[0].shape is None:
            return None
        return (self.length, *(self[0].shape))

    @property
    def dim(self) -> int:
        return len(self)

    def append(self, element, refuse_value=NoDefault) -> "vector":
        """append.

        Parameters
        ----------
        args :
            args
        """
        if not isinstance(refuse_value, EmptyClass):
            if element == refuse_value:
                return self
        self.update_appendix(element)
        self._index_mapping = IndexMapping()
        if not isinstance(self.content_type, EmptyClass):
            assert isinstance(element, self.content_type)
        super().append(element)
        return self

    def extend(self, other) -> "vector":
        """extend.

        Parameters
        ----------
        args :
            args
        """
        self.clear_appendix()
        self._index_mapping = IndexMapping()
        if hasattr(self, "content_type") and not isinstance(self.content_type, EmptyClass):
            assert vector(other).check_type(self.content_type)
        super().extend(other)
        return self

    def pop(self, *args):
        """pop.

        Parameters
        ----------
        args :
            args
        """
        self.clear_appendix()
        self._index_mapping = IndexMapping()
        if len(args) == 0:
            number = 1
        elif len(args) == 1:
            number = args[0]
        else:
            raise TypeError("pop expected at most 1 argument, got {}".format(len(args)))

        return super().pop(*args)

    def insert(self, location, element) -> "vector":
        """insert.

        Parameters
        ----------
        args :
            args
        """
        self.clear_appendix()
        if not isinstance(self.content_type, EmptyClass):
            assert isinstance(element, self.content_type)
        self._index_mapping = IndexMapping()

        super().insert(*args)
        return self

    @destory_registered_property
    def clear_appendix(self):
        # touch(lambda: delattr(self, "_vector__shape"))
        touch(lambda: delattr(self, "_vector__hashable"))
        touch(lambda: delattr(self, "_vector__set"))
        touch(lambda: delattr(self, "_vector__sum"))
        touch(lambda: delattr(self, "_vector__max"))
        touch(lambda: delattr(self, "_vector__min"))
        touch(lambda: delattr(self, "_vector__variance"))
        touch(lambda: delattr(self, "_vector__norm"))
        touch(lambda: delattr(self, "_vector__type"))
        touch(lambda: delattr(self, "_vector__type_recursive"))
        touch(lambda: delattr(self, "_vector__numpy"))

    @destory_registered_property
    def update_appendix(self, element):
        if self.length == 0:
            self.__hashable = hashable(element)
            if self.__hashable:
                self.__set = set([element])
            if isinstance(element, int) or isinstance(element, float):
                self.__sum = element
                self.__max = element
                self.__min = element
            self.__type = type(element)
            return
        # touch(lambda: delattr(self, "_vector__shape"))
        if hasattr(self, "_vector__hashable"):
            self.__hashable = self.__hashable and hashable(element)
        if hasattr(self, "_vector__set"):
            if hashable(element):
                self.__set.add(element)
            else:
                self.__set = None
        if hasattr(self, "_vector__sum"):
            self.__sum = touch(lambda: self.__sum + element)
        if hasattr(self, "_vector__max"):
            self.__max = touch(lambda: max(self.__max, element))
        if hasattr(self, "_vector__min"):
            self.__min = touch(lambda: min(self.__min, element))
        touch(lambda: delattr(self, "_vector__variance"))
        touch(lambda: delattr(self, "_vector__norm"))
        if hasattr(self, "_vector__type"):
            if isinstance(self.__type, set):
                self.__type.add(type(element))
            elif self.__type is not type(element):
                self.__type = set([self.__type, type(element)])
        if hasattr(self, "_vector__type") and not isinstance(self.__type, set) and self.__type is not vector:
            self.__type_recursive = self.__type
        else:
            touch(lambda: delattr(self, "_vector__type_recursive"))
        touch(lambda: delattr(self, "_vector__norm"))
        touch(lambda: delattr(self, "_vector__numpy"))

    def clear(self) -> "vector":
        """clear
        """
        self.clear_appendix()
        self._index_mapping = IndexMapping()
        super().clear()
        return self

    def remove(self, *args) -> "vector":
        """remove.

        Parameters
        ----------
        args :
            args
        """
        self.clear_appendix()
        super().remove(*args)
        return self

    def all_equal(self, func=None):
        """
            test if all element in a vector are equal or not
        """
        if self.length <= 1:
            return True
        if func is None:
            func = lambda x: x
        return self.all(lambda x: func(x) == func(self[0]))

    @overload
    def sample(self, size: Iterable, replace=True, batch_size=1, p=None): ...

    @overload
    def sample(self, *size, replace=True, batch_size=1, p=None): ...

    @overload
    def sample(self, n=None, replace=True, batch_size=1, p=None): ...

    def sample(self, *args, n=None, replace=True, batch_size=1, p=None):
        """sample.

        Parameters
        ----------
        args :
            args
        replace :
            replace
        p :
            p

        sample from vector for given probability p
        example:
            vector.range(10).sample()
            vector.range(10).sample(10)
            vector.range(10).sample(0.5)
            vector.range(10).sample(10, replace=False)
        """
        args = totuple(args)
        if len(args) == 0 and n is not None:
            args = totuple(n)
        if len(args) == 0:
            return self.sample(1, replace=replace, batch_size=batch_size, p=p)[0]
        if len(args) == 1 and isinstance(args[0], float) and 0 < args[0] < 1:
            return self.sample(math.ceil(self.length * args[0]), replace=replace, batch_size=batch_size, p=p)
        if isinstance(args[-1], bool):
            replace = args[-1]
            args = args[:-1]
        if len(args) >= 2 and isinstance(args[-2], bool) and isinstance(args[-1], (list, np.ndarray)):
            replace = args[-2]
            p = args[-1]
            args = args[:-2]
        if batch_size > 1:
            args = (*args, batch_size)
        if len(args) == 1 and replace == False:
            index_mapping = IndexMapping(np.random.choice(vector.range(self.length), size = args, replace=False, p=p), range_size=self.length, reverse=True)
            return self.map_index(index_mapping)
        if len(args) == 1 and replace == True:
            index_mapping = IndexMapping(vector.randint(0, high=self.length, size=args[0]), range_size=self.length, reverse=True)
            return self.map_index(index_mapping)
        return vector(np.random.choice(self, size=args, replace=replace, p=p), recursive=False)

    def batch(self, batch_size=1, random=True, drop=True) -> "vector":
        if random:
            if self.length % batch_size == 0:
                return self.sample(self.length // batch_size, batch_size, replace=False)
            if drop:
                return self.sample(self.length // batch_size, batch_size, replace=False)
            else:
                return (self + self.sample(batch_size - self.length % batch_size)).batch(batch_size=batch_size, drop=True)
        else:
            return self[:(self.length - self.length % batch_size)].reshape(-1, batch_size)

    def shuffle(self) -> "vector":
        """
        shuffle the vector
        """
        index_mapping = IndexMapping(vector.range(self.length).sample(self.length, replace=False))
        return self.map_index(index_mapping)

    def reverse(self) -> "vector":
        """
        reverse the vector
        vector(0,1,2).reverse()
        will get
        [2,1,0]
        """
        index_mapping= IndexMapping(vector.range(self.length-1, -1, -1))
        return self.map_index(index_mapping)

    def reverse_(self) -> "vector":
        """
        **inplace function:** reverse the vector
        vector(0,1,2).reverse()
        will get
        [2,1,0]
        """
        index_mapping= IndexMapping(vector.range(self.length-1, -1, -1))
        return self.map_index_(index_mapping)

    def shuffle_(self) -> "vector":
        """
        **Inplace function:** shuffle the vector
        """
        index_mapping = IndexMapping(vector.range(self.length).sample(self.length, replace=False))
        self.map_index_(index_mapping)

    def split(self, *args) -> "vector":
        """
        split vector in given position
        """
        if len(args) == 0:
            if self.length == 0 and self.check_type(str):
                return vector(super(vector, self).__getitem__(0).split())
            else:
                return self
        if len(args) == 1 and isinstance(args[0], str):
            if self.length == 1 and self.check_type(str):
                return vector(super(vector, self).__getitem__(0).split(args[0]))

        if len(args) == 1:
            pivot = args[0]
            ret = list()
            temp = vector()
            for item in self:
                if item != pivot:
                    temp.append(item)
                else:
                    ret.append(temp)
                    temp = vector()
            ret.append(temp)
            return vector(ret)

    def split_index(self, *args) -> "vector":
        args = totuple(args)
        args = vector(args).sort()
        args.all(lambda x: 0 <= x <= self.length)
        if args[0] != 0:
            args = [0] + args
        if args[-1] != self.length:
            args.append(self.length)
        ret_split = vector()
        for index in range(len(args)-1):
            ret_split.append(IndexMapping(vector.range(args[index], args[index+1]), range_size=self.length, reverse=True))
        ret = ret_split.map(lambda x: self.map_index(x))
        return ret

    def split_random(self, *args) -> Tuple:
        args = totuple(args)
        args = vector(args).sort(lambda x: -x).normalization(p=1)
        sorted_index_mapping = args.index_mapping
        split_num = vector()
        remain_len = self.length
        while len(args) > 0:
            args = args.normalization(p=1)
            split_len = round(remain_len * args[0])
            remain_len -= split_len
            split_num.append(split_len)
            args = args[1:]
        assert split_num.sum() == self.length
        cumsum = split_num.cumsum()
        return tuple(self.shuffle().split_index(cumsum).map_index(sorted_index_mapping.reverse()))

    def copy(self, deep_copy=False) -> "vector":
        if not deep_copy:
            ret = vector(self)
        else:
            ret = vector(copy.deep_copy(self))
        if self.index_mapping is not None:
            ret._index_mapping = self.index_mapping.copy()
        else:
            ret._index_mapping = IndexMapping()
        ret.allow_undefined_value = self.allow_undefined_value
        ret._recursive = self._recursive
        ret.content_type = self.content_type
        return ret

    def map_index(self, index_mapping: "IndexMapping") -> "vector":
        """
        change the index_mapping of the current vector.
        for example:
        t is a vector and t[1] = "apple". And im is an index_mapping which map index 1 to 3. Then for new vector t_new = t.map_index(im), t_new[3] = "apple"

        more example:
        for a vector t = vector(3,2,1). t.sort() will have index_mapping which map:
        index   ->    to
        0       ->    2
        1       ->    1
        2       ->    0

        for another vector p = vector("apple", "banana", "peach")
        p.map_index(t.index_mapping)
        will generate:
        vector("peach", "banana", "apple")
        """
        assert isinstance(index_mapping, IndexMapping)
        if index_mapping.isidentity:
            return self
        assert self.length == index_mapping.domain_size
        if index_mapping.range_size == 0:
            return vector([], index_mapping=self.index_mapping.map(index_mapping), allow_undefined_value=self.allow_undefined_value)
        if index_mapping.isslice:
            slice_index = index_mapping.slice
            if slice_index.step < 0 and slice_index.stop == -1:
                slice_index = slice(slice_index.start, None, slice_index.step)
            ret = vector(super(vector, self).__getitem__(slice_index), recursive=self._recursive, index_mapping=self.index_mapping.map(index_mapping), allow_undefined_value=False)
            return ret
        if not self.allow_undefined_value:
            # assert all(0 <= index < self.length for index in index_mapping.index_map_reverse)
            ret = vector([super(vector, self).__getitem__(index) for index in index_mapping.index_map_reverse], recursive=self._recursive, index_mapping=self.index_mapping.map(index_mapping), allow_undefined_value=False)
            return ret
        else:
            ret = vector([super(vector, self).__getitem__(index) if index >= 0 else UnDefined for index in index_mapping.index_map_reverse], recursive=self._recursive, index_mapping=self.index_mapping.map(index_mapping), allow_undefined_value=True)
            return ret

    def map_index_(self, index_mapping: "IndexMapping") -> None:
        """
        inplacement implementation of map_index
        """
        assert isinstance(index_mapping, IndexMapping)
        if index_mapping.isidentity:
            return self
        assert self.length == index_mapping.domain_size
        ret = self.map_index(index_mapping)
        self.clear_appendix()
        super().clear()
        super().extend(ret)
        self._index_mapping = ret.index_mapping

    def original_index(self, index):
        if self.index_mapping.isidentity:
            return index
        return self.index_mapping.reverse_getitem(index)

    def register_index_mapping(self, index_mapping=IndexMapping()):
        ret = self.copy()
        ret._index_mapping = index_mapping
        return ret

    def register_index_mapping_(self, index_mapping=IndexMapping()):
        self._index_mapping = index_mapping

    def map_index_from(self, x) -> "vector":
        assert isinstance(x, vector)
        return self.map_index(x.index_mapping)

    def map_index_from_(self, x) -> None:
        assert isinstance(x, vector)
        self.map_index_(x.index_mapping)

    def map_reverse_index(self, reverse_index_mapping: "IndexMapping") -> "vector":
        assert isinstance(reverse_index_mapping, IndexMapping)
        return self.map_index(reverse_index_mapping.reverse())

    def map_reverse_index_(self, reverse_index_mapping: "IndexMapping") -> None:
        assert isinstance(reverse_index_mapping, IndexMapping)
        self.map_index(reverse_index_mapping.reverse())

    def clear_index_mapping(self):
        return self.register_index_mapping(index_mapping=IndexMapping())

    def clear_index_mapping_(self):
        self.register_index_mapping_(index_mapping=IndexMapping())

    def unmap_index(self):
        if self.index_mapping.isidentity:
            return self
        temp_flag = self.allow_undefined_value
        if self.index_mapping.domain_size > self.index_mapping.range_size:
            self.allow_undefined_value = True
        ret = self.map_index(self.index_mapping.reverse())
        ret.clear_index_mapping_()
        self.allow_undefined_value = temp_flag
        return ret

    def unmap_index_(self):
        if self.index_mapping.isidentity:
            return
        if self.index_mapping.domain_size > self.index_mapping.range_size:
            self.allow_undefined_value = True
        self.map_index_(self.index_mapping.reverse())
        self.clear_index_mapping_()

    def roll(self, shift=1) -> "vector":
        index_mapping = IndexMapping([(index - shift) % self.length for index in range(self.length)], range_size=self.length, reverse=True)
        return self.map_index(index_mapping)

    def roll_(self, shift=1):
        index_mapping = IndexMapping([(index - shift) % self.length for index in range(self.length)], range_size=self.length, reverse=True)
        return self.map_index_(index_mapping)

    def __str__(self):
        if self.str_function is not None:
            return self.str_function(self)
        if self.shape is not None and len(self.shape) > 1:
            ret: List[str] = vector()
            for index, child in self.enumerate():
                if isinstance(child, str):
                    contents = ['\'{}\''.format(child)]
                else:
                    contents = str(child).split("\n")
                for j, content in enumerate(contents):
                    temp = ""
                    if index == 0 and j == 0:
                        temp = "["
                    else:
                        temp = " "
                    temp += content.rstrip()
                    if j == len(contents) - 1:
                        if index < self.length - 1:
                            temp += ","
                        else:
                            temp += "]"
                    ret.append(temp)
            return "\n".join(ret)
        else:
            ret = "["
            for index, child in self.enumerate():
                if isinstance(child, str):
                    ret += "'{}'".format(child)
                else:
                    ret += str(child)
                if index < self.length - 1:
                    ret += ", "
            ret += "]"
            return ret

    def __repr__(self):
        ret = self.__str__()
        if not self.index_mapping.isidentity:
            ret += ", with index mapping"
        return ret

    def set(self):
        if hasattr(self, "_vector__set"):
            if self.__set is None:
                raise RuntimeError("this vector is not hashable")
            return self.__set
        if not self.ishashable():
            raise RuntimeError("this vector is not hashable")
        self.__set = set(self)
        return self.__set

    def __contains__(self, item):
        if self.ishashable():
            return item in self.set()
        return super().__contains__(item)

    def __bool__(self):
        return self.length > 0

    def popen(self, *, max_process=-1, delay_time=0.1, interval=0.1, popen_kwargs={}, info=False):
        if not self:
            return
        assert all([isinstance(x, list) for x in self])
        assert all([all([isinstance(y, str) for y in x]) for x in self])

        proc_vector = list(self.copy())
        running_procs = list()

        while proc_vector and (max_process <= 0 or len(running_procs) < max_process):
            cmd = proc_vector[0]
            proc_vector = proc_vector[1:]
            running_procs.append(Popen(cmd, **popen_kwargs))
            time.sleep(delay_time)

        while proc_vector or running_procs:
            while running_procs:
                for proc in running_procs:
                    retcode = proc.poll()
                    if info:
                        print(retcode)
                    if retcode is not None: # process finished.
                        running_procs.remove(proc)
                        if proc_vector:
                            cmd = proc_vector[0]
                            proc_vector = proc_vector[1:]
                            running_procs.append(Popen(cmd, **popen_kwargs))
                            time.sleep(delay_time)
                        break
                    else: # no process is done, wait a bit and check again.
                        time.sleep(interval)
                        continue

    def function_search(self, search_func, query="", max_k=NoDefault, str_func=str, str_display=None, display_info=None, sorted_function=None, pre_sorted_function=None, history=None, show_line_number=False, return_tuple=False, stdscr=None):
        """
        Provide interactive search function for item in vector

        Parameters
        ---------------------
        search_func: Callable
            search_func(candidate: vector[str], query: str) -> vector[str]
            candidate is a vector of all string representation (1) of items, query is the current query string
            output the query result
        max_k: int
            maximum number of queried results
        str_func: Callable
            function to get the string representation (1) which is used for query
        display_info: Callable
            function to get the string which determined how each item is displayed
        sorted_function: Callable
            key function determining how result of search_func is sorted
        pre_sorted_function: Callable
            key function determining how self is sorted before query

            process is:
            pre_sorted_function -> search_function -> sorted_function
        history: dict
            dict to restore preview query scession
            with key:
            display_bias
            select_number
            query
            x_bias
        show_line_number: bool
            whether to show line_number e.g. [11] before each item
        return_tuple: bool
            if return_tuple, when you press enter at some item, (item_index, item) tuple will be returned
            otherwise, only item will be returned
        """
        if str_display is None:
            str_display = str_func
        if len(query) > 0:
            self = self.sort(key=pre_sorted_function)
            candidate = self.clear_index_mapping().map(str_func)
            selected = search_func(candidate, query)
            if not selected:
                return None
            if isinstance(max_k, EmptyClass):
                return self.map_index_from(selected).sort(sorted_function)[0]
            else:
                return self.map_index_from(selected).sort(sorted_function)[:max_k]
        elif stdscr is None:
            def c_main(stdscr: "curses._CursesWindow"):
                return self.function_search(search_func, query="", max_k=max_k, str_func=str_func, str_display=str_display, display_info=display_info, sorted_function=sorted_function, pre_sorted_function=pre_sorted_function, history=history, show_line_number=show_line_number, return_tuple=return_tuple, stdscr=stdscr)
            return curses.wrapper(c_main)
        else:
            def write_line(row, col=0, content=""):
                content = " ".join(vector(content.split("\n")).map(lambda x: x.strip()))
                if len(content) >= cols:
                    stdscr.addstr(row, col, content[:cols])
                else:
                    stdscr.addstr(row, col, content)
                    stdscr.clrtoeol()
            stdscr.clear()
            new_self = self.sort(key=pre_sorted_function).clear_index_mapping()
            candidate = new_self.map(str_func)
            query_done = False
            select_number = 0
            rows, cols = stdscr.getmaxyx()
            for index in range(rows):
                write_line(index, 0, "")
            x_init = len("token to search: ")

            stdscr.addstr(0, 0, "token to search: ")
            search_k = max_k
            if search_k is NoDefault:
                search_k = int(max(min(rows - 8, rows * 0.85), rows * 0.5))

            display_bias = 0
            select_number = 0
            query = ""
            char = ""
            x_bias = 0
            error_info = ""
            if isinstance(history, dict):
                display_bias = history.get("display_bias", 0)
                select_number = history.get("select_number", 0)
                query = history.get("query", "")
                x_bias = history.get("x_bias", 0)

            selected = search_func(candidate, query)
            result = new_self.map_index_from(selected).sort(key=sorted_function)[display_bias:display_bias + search_k]
            result_str = result.map(str_display).map(lambda x: x.replace("\n", " ").replace("\t", " "* 4)[:cols])

            write_line(search_k + 1, 0, "-" * int(0.8 * cols))

            while True:
                write_line(0, 0, "token to search: ")
                stdscr.addstr(0, len("token to search: "), query)

                write_line(search_k + 2, 0, "# match: " + str(selected.length))
                write_line(search_k + 3, 0, "# dispaly: " + str(result.length))
                error_nu = search_k + 4
                if display_info is not None:
                    info = display_info(self, query, selected)
                    if isinstance(info, str):
                        info = vector([info])
                    for index in range(len(info)):
                        if search_k + 4 + index < rows:
                            write_line(search_k + 4 + index, 0, info[index][:cols])
                            error_nu += 1
                        else:
                            break
                if error_info:
                    if error_nu < rows:
                        for line in error_info.split("\n"):
                            write_line(error_nu, 0, line[:cols])
                            error_nu += 1
                for index in range(error_nu, rows):
                    write_line(index, 0, "")

                for index in range(len(result)):
                    if show_line_number:
                        display_str = "[{}] {}".format(result.original_index(index), result_str[index])
                    else:
                        display_str = result_str[index]
                    if index == select_number:
                        write_line(1 + index, 0, "* " + display_str)
                    else:
                        write_line(1 + index, 0, display_str)
                    assert index < search_k
                for index in range(len(result), search_k):
                    write_line(1 + index, 0, "")
                write_line(rows-1, cols-5, content=str(char))

                def new_len(x): return 1 + int(u'\u4e00' <= x <= u'\u9fff')
                new_x_bias = sum([new_len(t) for t in query[:x_bias]])
                stdscr.addstr(0, x_init + new_x_bias, "")
                search_flag = False
                char = stdscr.get_wch()
                # logger.info(str(char))
                if char == "\x1b" or char == curses.KEY_EXIT or char == "`":
                    return None
                elif isinstance(char, str) and char == "π":
                    if history is not None:
                        assert isinstance(history, dict)
                        history["select_number"] = select_number
                        history["display_bias"] = display_bias
                        history["query"] = query
                        history["x_bias"] = x_bias
                    if len(result) > 0:
                        return ("p", result.original_index(select_number), result[select_number])
                    return None
                elif isinstance(char, str) and char.isprintable():
                    query = query[:x_bias] + char + query[x_bias:]
                    select_number = 0
                    display_bias = 0
                    x_bias += 1
                    search_flag = True
                elif char == curses.KEY_BACKSPACE or char == "\x7f" or touch(lambda: ord(char), 0) == 8:
                    query = query[:max(x_bias - 1, 0)] + query[x_bias:]
                    select_number = 0
                    display_bias = 0
                    x_bias = max(x_bias - 1, 0)
                    search_flag = True
                elif char == "\n":
                    if history is not None:
                        assert isinstance(history, dict)
                        history["select_number"] = select_number
                        history["display_bias"] = display_bias
                        history["query"] = query
                        history["x_bias"] = x_bias
                    if len(result) > 0:
                        if return_tuple:
                            return (result.original_index(select_number), result[select_number])
                        else:
                            return result[select_number]
                    return None
                elif char == curses.KEY_UP or char == 259:
                    if select_number == 2 and display_bias > 0:
                        display_bias -= 1
                        result = new_self.map_index_from(selected).sort(key=sorted_function)[display_bias:display_bias + search_k]
                        result_str = result.map(str_display).map(lambda x: x.replace("\n", " ").replace("\t", " " * 4)[:cols])
                    else:
                        select_number = max(select_number - 1, 0)
                elif char == curses.KEY_DOWN or char == 258:
                    if select_number == search_k - 3 and display_bias + search_k < selected.length:
                        display_bias += 1
                        result = new_self.map_index_from(selected).sort(key=sorted_function)[display_bias:display_bias + search_k]
                        result_str = result.map(str_display).map(lambda x: x.replace("\n", " ").replace("\t", " " * 4)[:cols])
                    else:
                        select_number = max(min(select_number + 1, len(result) - 1), 0)
                elif char == curses.KEY_LEFT:
                    x_bias = max(x_bias - 1, 0)
                    continue
                elif char == curses.KEY_RIGHT:
                    x_bias = min(x_bias + 1, len(query))
                elif char == '\x01':
                    x_bias = 0
                elif char == '\x05':
                    x_bias = len(query)
                elif char == 338:
                    # page down
                    increase_amount = max(min(search_k, selected.length - search_k - display_bias), 0)
                    display_bias += increase_amount
                    if increase_amount != 0:
                        result = new_self.map_index_from(selected).sort(key=sorted_function)[display_bias:display_bias + search_k]
                        result_str = result.map(str_display).map(lambda x: x.replace("\n", " ").replace("\t", " " * 4)[:cols])
                elif char == 339:
                    # page up
                    decrease_amount = max(min(search_k, display_bias), 0)
                    display_bias -= decrease_amount
                    if increase_amount != 0:
                        result = new_self.map_index_from(selected).sort(key=sorted_function)[display_bias:display_bias + search_k]
                        result_str = result.map(str_display).map(lambda x: x.replace("\n", " ").replace("\t", " " * 4)[:cols])
                elif char == 262:
                    # home
                    display_bias = 0
                    select_number = 0
                    result = new_self.map_index_from(selected).sort(key=sorted_function)[display_bias:display_bias + search_k]
                    result_str = result.map(str_display).map(lambda x: x.replace("\n", " ").replace("\t", " " * 4)[:cols])
                elif char == 360:
                    # end
                    display_bias = max(selected.length - search_k, 0)
                    result = new_self.map_index_from(selected).sort(key=sorted_function)[display_bias:display_bias + search_k]
                    result_str = result.map(str_display).map(lambda x: x.replace("\n", " ").replace("\t", " " * 4)[:cols])
                else:
                    # raise RuntimeError()
                    pass

                try:
                    if search_flag:
                        selected = search_func(candidate, query)
                        result = new_self.map_index_from(selected).sort(key=sorted_function)[display_bias:display_bias + search_k]
                        result_str = result.map(str_display).map(lambda x: x.replace("\n", " ").replace("\t", " " * 4)[:cols])
                except Exception as e:
                    error_info = str(e)
                else:
                    error_info = ""

    def regex_search(self, query="", max_k=NoDefault, str_func=str, str_display=None, display_info=None, sorted_function=None, pre_sorted_function=None, history=None, show_line_number=False, return_tuple=False, stdscr=None):

        def regex_function(candidate, query):
            if len(query) == 0:
                return candidate
            regex = re.compile(query)
            selected = candidate.filter(lambda x: regex.search(x), ignore_error=False).sort(len)
            return selected

        return self.function_search(regex_function, query=query, max_k=max_k, str_func=str_func, str_display=str_display, display_info=display_info, sorted_function=sorted_function, pre_sorted_function=pre_sorted_function, history=history, show_line_number=show_line_number, return_tuple=return_tuple, stdscr=stdscr)

    def fuzzy_search(self, query="", max_k=NoDefault, str_func=str, str_display=None, display_info=None, sorted_function=None, pre_sorted_function=None, history=None, show_line_number=False, return_tuple=False, stdscr=None):

        def fuzzy_function(candidate, query):
            upper = any(x.isupper() for x in query)
            if len(query) == 0:
                return candidate
            else:
                candidate = candidate.filter(lambda x: query[0] in x)
                if not upper:
                    candidate = candidate.map(lambda x: x.lower())
            def eta(x):
                if len(x) > len(query):
                    return 1 - (len(x) - len(query)) / (len(x) + len(query))
                else:
                    return 1 - (len(query) - len(x)) / (len(x) + len(query)) / 2
            if len(candidate) < 1000:
                partial_ratio = candidate.map(lambda x: (fuzz.ratio(x, query) / eta(x) ** 0.8, x))
                selected = partial_ratio.filter(lambda x: x[0] > 49)
            else:
                if len(query) == 1:
                    return candidate.map(lambda x: 100)
                if len(candidate) > 5000:
                    candidate = candidate.filter(lambda x: query[:2] in x)
                else:
                    candidate = candidate.filter(lambda x: query[0] in x and query[1] in x)
                partial_ratio = candidate.map(lambda x: (fuzz.ratio(x, query) / eta(x) ** 0.8, x))
                selected = partial_ratio.filter(lambda x: x[0] > 49)
            # score = selected.map(lambda x: 100 * (x[0] == 100) + x[0] * min(1, len(x[1]) / len(query)) * min(1, len(query) / len(x[1])) ** 0.3, lambda x: round(x * 10) / 10).sort(lambda x: -x)
            score = selected.map(lambda x: x[0]).sort(lambda x: -x)
            return score

        return self.function_search(fuzzy_function, query=query, max_k=max_k, str_func=str_func, str_display=str_display, display_info=display_info, sorted_function=sorted_function, pre_sorted_function=pre_sorted_function, history=history, show_line_number=show_line_number, return_tuple=return_tuple, stdscr=stdscr)

    def get_size(self):
        return self.rmap(sys.getsizeof).reduce(lambda x, y: x + y, first=0)

    def __dir__(self):
        return vector(super().__dir__())

    # def __getattr__(self, name):
    #     try:
    #         return object.__getattr__(self, name)
    #     except:
    #         raise RuntimeError("{} is not a method/attribute of vector, the most similar name is {}".format(name, vector(dir(self)).fuzzy_search(name, 3)))

    @staticmethod
    def search_content(obj, history=None, prefix="", stdscr=None):
        assert isinstance(obj, (list, vector, set, dict, tuple)) or list_like(obj)
        if isinstance(obj, (list, vector, set, tuple)):
            vector_obj = vector(obj)
            selected = vector_obj.fuzzy_search(history=history, show_line_number=True, return_tuple=True, display_info=lambda x, y, z: vector(["prefix: " + prefix]), stdscr=stdscr)
            return selected
        elif isinstance(obj, dict):
            vector_obj = vector(obj.keys())
            selected = vector_obj.fuzzy_search(str_display=lambda x: "[{}]: {}".format(x, obj[x]), history=history, return_tuple=True, display_info=lambda x, y, z: vector(["prefix: " + prefix]), stdscr=stdscr)
            return selected
        elif list_like(obj):
            vector_obj = vector(obj)
            selected = vector_obj.fuzzy_search(history=history, show_line_number=True, return_tuple=True, display_info=lambda x, y, z: vector(["prefix: " + prefix]), stdscr=stdscr)
            return selected

    def help(self, only_content=False, prefix="", stdscr=None):
        return vhelp(self, only_content=only_content, prefix=prefix, stdscr=stdscr)

    def save(self, filepath):
        try:
            import pickle
        except:
            print("Please install pickle package")
            return
        with open(filepath, "wb") as output:
            pickle.dump(self.tolist(), output)
            pickle.dump(self.index_mapping, output)

    @staticmethod
    def load(filepath) -> "vector":
        try:
            import pickle
        except:
            print("Please install pickle package")
            return
        with open(filepath, "rb") as input:
            content = pickle.load(input)
            index_mapping = pickle.load(input)
            ret = vector.from_list(content)
            ret._index_mapping = index_mapping
        return ret

    def detect_peaks(self, mph=None, mpd=1, threshold=0, edge='rising',
                     kpsh=False, valley=False, show=False, ax=None):
        """Detect peaks in data based on their amplitude and other features.
        Parameters
        ----------
        x : 1D array_like
            data.
        mph : {None, number}, optional (default = None)
            detect peaks that are greater than minimum peak height.
        mpd : positive integer, optional (default = 1)
            detect peaks that are at least separated by minimum peak distance (in
            number of data).
        threshold : positive number, optional (default = 0)
            detect peaks (valleys) that are greater (smaller) than `threshold`
            in relation to their immediate neighbors.
        edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
            for a flat peak, keep only the rising edge ('rising'), only the
            falling edge ('falling'), both edges ('both'), or don't detect a
            flat peak (None).
        kpsh : bool, optional (default = False)
            keep peaks with same height even if they are closer than `mpd`.
        valley : bool, optional (default = False)
            if True (1), detect valleys (local minima) instead of peaks.
        show : bool, optional (default = False)
            if True (1), plot data in matplotlib figure.
        ax : a matplotlib.axes.Axes instance, optional (default = None).
        Returns
        -------
        ind : 1D array_like
            indeces of the peaks in `x`.
        Notes
        -----
        The detection of valleys instead of peaks is performed internally by simply
        negating the data: `ind_valleys = detect_peaks(-x)`

        The function can handle NaN's
        See this IPython Notebook [1]_.
        References
        ----------
        "Marcos Duarte, https://github.com/demotu/BMC"
        [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
        """

        x = np.atleast_1d(self).astype('float64')
        if x.size < 3:
            return np.array([], dtype=int)
        if valley:
            x = -x
        # find indexes of all peaks
        dx = x[1:] - x[:-1]
        # handle NaN's
        indnan = np.where(np.isnan(x))[0]
        if indnan.size:
            x[indnan] = np.inf
            dx[np.where(np.isnan(dx))[0]] = np.inf
        ine, ire, ife = np.array([[], [], []], dtype=int)
        if not edge:
            ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
        else:
            if edge.lower() in ['rising', 'both']:
                ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
            if edge.lower() in ['falling', 'both']:
                ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
        ind = np.unique(np.hstack((ine, ire, ife)))
        # handle NaN's
        if ind.size and indnan.size:
            # NaN's and values close to NaN's cannot be peaks
            ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
        # first and last values of x cannot be peaks
        if ind.size and ind[0] == 0:
            ind = ind[1:]
        if ind.size and ind[-1] == x.size - 1:
            ind = ind[:-1]
        # remove peaks < minimum peak height
        if ind.size and mph is not None:
            ind = ind[x[ind] >= mph]
        # remove peaks - neighbors < threshold
        if ind.size and threshold > 0:
            dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
            ind = np.delete(ind, np.where(dx < threshold)[0])
        # detect small peaks closer than minimum peak distance
        if ind.size and mpd > 1:
            ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
            idel = np.zeros(ind.size, dtype=bool)
            for i in range(ind.size):
                if not idel[i]:
                    # keep peaks with the same height if kpsh is True
                    idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                           & (x[ind[i]] > x[ind] if kpsh else True)
                    idel[i] = 0  # Keep current peak
            # remove the small peaks and sort back the indexes by their occurrence
            ind = np.sort(ind[~idel])

        return vector.from_numpy(ind)

def vhelp(obj=None, history=None, only_content=False, prefix="", stdscr=None, enhanced=False):
    if obj is None:
        return vhelp(vector, history=history, only_content=only_content, stdscr=stdscr, enhanced=enhanced)
    elif stdscr is None:
        def help_main(stdscr):
            ret = vhelp(obj, history=history, only_content=only_content, prefix=prefix, stdscr=stdscr, enhanced=enhanced)
            if ret and prefix:
                return prefix + ret
            else:
                return ret
        return curses.wrapper(help_main)
    else:
        content_type = (list, vector, set, tuple, dict)
        if only_content:
            if isinstance(obj, content_type) or (enhanced and list_like(obj)):
                if history is None:
                    history = {}
                selected = vector.search_content(obj, history=history, prefix=prefix, stdscr=stdscr)
                if selected:
                    if len(selected) == 2:
                        if isinstance(obj, (list, vector, set, tuple)):
                            ret = vhelp(selected[1], only_content=True, prefix=prefix + "[{}]".format(selected[0]), stdscr=stdscr, enhanced=enhanced)
                            if ret is not None:
                                if isinstance(obj, (list, vector, tuple)):
                                    return "[{}]".format(selected[0]) + ret
                                else:
                                    return ".set<{}>".format(selected[1]) + ret
                            ret = vhelp(obj, history=history, only_content=True, prefix=prefix, stdscr=stdscr, enhanced=enhanced)
                            if ret is not None:
                                return ret
                        elif isinstance(obj, dict):
                            value = obj.get(selected[1], None)
                            if value is None:
                                return
                            def get_dict_string_key(key):
                                if isinstance(key, str):
                                    return "[\"{}\"]".format(key)
                                return "[{}]".format(key)
                            ret = vhelp(value, only_content=True, prefix=prefix + get_dict_string_key(selected[1]), stdscr=stdscr, enhanced=enhanced)
                            if ret is not None:
                                return get_dict_string_key(selected[1])
                            ret = vhelp(obj, history=history, only_content=True, prefix=prefix, stdscr=stdscr, enhanced=enhanced)
                            if ret is not None:
                                return ret
                        elif list_like(obj):
                            ret = vhelp(selected[1], only_content=True, prefix=prefix + "[{}]".format(selected[0]), stdscr=stdscr, enhanced=enhanced)
                            if ret is not None:
                                return "[{}]".format(selected[0]) + ret
                            ret = vhelp(obj, history=history, only_content=True, prefix=prefix, stdscr=stdscr, enhanced=True)
                            if ret is not None:
                                return ret
                    elif len(selected) == 3 and selected[0] == "p":
                        if isinstance(obj, (list, vector, tuple)):
                            return "[{}]".format(selected[1])
                        elif isinstance(obj, dict):
                            if isinstance(selected[2], str):
                                return "[\"{}\"]".format(selected[2])
                            else:
                                return "[{}]".format(selected[2])
                        elif isinstance(obj, set):
                            return ".set<{}>".format(selected[1])
                        elif list_like(obj):
                            return "[{}]".format(selected[1])
                return
            if isinstance(obj, (int, str, float)):
                raw_display_str = str(obj).replace("\t", "    ")
                str_length = len(raw_display_str)
                line_number = raw_display_str.count("\n") + 1

                def write_line(row, col=0, content=""):
                    if col >= cols:
                        return
                    if row >= rows:
                        return
                    if len(content) + col >= cols:
                        stdscr.addstr(row, col, content[:cols - col])
                    else:
                        stdscr.addstr(row, col, content)
                        stdscr.clrtoeol()
                stdscr.clear()
                rows, cols = stdscr.getmaxyx()
                for index in range(rows):
                    stdscr.addstr(index, 0, "")
                    stdscr.clrtoeol()
                display_str = vector(raw_display_str.split("\n"))
                for index in range(len(display_str)):
                    display_str[index] = "[{}] ".format(index+1) + display_str[index]
                def split_len(s, l):
                    if len(s) <= l:
                        return vector([s])
                    ret = vector()
                    while s:
                        if len(s) > l:
                            ret.append(s[:l-1] + "\\")
                        else:
                            ret.append(s[:l])
                        s = s[l:]
                    return ret
                display_str = display_str.map(lambda x: split_len(x, cols-1)).flatten()
                line_bias = 0
                search_k = int(max(min(rows - 8, rows * 0.85), rows * 0.5)) + 1
                write_line(search_k, 0, "-" * int(0.8 * cols))
                write_line(search_k+1, 0, "# char: {}".format(str_length))
                write_line(search_k+2, 0, "# line: {}".format(line_number))
                write_line(search_k+3, 0, "prefix: " + prefix)
                while True:
                    display = display_str[line_bias: line_bias + search_k]
                    for index in range(len(display)):
                        write_line(index, 0, display[index])
                    for index in range(len(display), search_k):
                        stdscr.addstr(index, 0, "")
                        stdscr.clrtoeol()
                    stdscr.addstr(0, 0, "")
                    char = stdscr.get_wch()
                    if char == "\x1b" or char == curses.KEY_EXIT or char == "q" or char == "`":
                        return
                    elif char == curses.KEY_DOWN:
                        line_bias = max(min(line_bias + 1, len(display_str) - search_k), 0)
                    elif char == curses.KEY_UP:
                        line_bias = max(line_bias - 1, 0)
                    elif char == "G":
                        line_bias = max(0, len(display_str) - search_k)
                    elif char == "g":
                        char == stdscr.get_wch()
                        if char == "g":
                            line_bias = 0
                        if char == "\x1b" or char == curses.KEY_EXIT or char == "q" or char == "`":
                            return
                        else:
                            continue
                    elif isinstance(char, str) and char.isdigit():
                        num = 0
                        while isinstance(char, str) and char.isdigit():
                            num = num * 10 + int(char)
                            char = stdscr.get_wch()
                        if char == "G":
                            line_bias = max(min(num - search_k // 6, len(display_str) - search_k), 0)
                        else:
                            continue
                    else:
                        continue
                return

        if not inspect.isfunction(obj) and not inspect.ismethod(obj) and not inspect.ismodule(obj) and not inspect.isclass(obj):
            original_obj = obj
            obj = obj.__class__
        else:
            original_obj = None
        def testfunc(obj, x):
            eval("obj.{}".format(x))
        if original_obj is not None:
            class_temp = vector(dir(obj)).unique().filter(lambda x: len(x) > 0 and x[0] != "_").test(lambda x: testfunc(obj, x))
            extra_temp = vector(dir(original_obj)).unique().filter(lambda x: not x.startswith("_")).test(lambda x: original_obj.__getattribute__(x)) - class_temp
            if isinstance(original_obj, (list, vector, tuple, set, dict)):
                temp = vector(["content"]) + class_temp + extra_temp
            elif enhanced and list_like(original_obj):
                temp = vector(["content"]) + class_temp + extra_temp
            else:
                temp = class_temp + extra_temp
        else:
            extra_temp = vector()
            temp = vector(dir(obj)).unique().filter(lambda x: len(x) > 0 and x[0] != "_").test(lambda x: testfunc(obj, x))
        if len(temp) == 0:
            if original_obj is not None:
                help_doc = pydoc.render_doc(original_obj, "Help on %s")
            else:
                help_doc = pydoc.render_doc(obj, "Help on %s")

            if help_doc[:-1].split("\n")[-1].strip().startswith("See :func:`torch."):
                import torch
                see_doc = help_doc[:-1].split("\n")[-1].strip()
                m = re.fullmatch(r"See :func:`torch.(\w+)`", see_doc)
                if m:
                    extra_doc = pydoc.render_doc(torch.__getattribute__(m.group(1)))
                    help_doc = "\n".join([help_doc, "", "doc for torch.%s" %(m.group(1)), "-" * 30, "", extra_doc])

            raw_display_str = help_doc.replace("\t", "    ")
            str_length = len(raw_display_str)
            line_number = raw_display_str.count("\n") + 1

            def write_line(row, col=0, content=""):
                if col >= cols:
                    return
                if row >= rows:
                    return
                if len(content) + col >= cols:
                    stdscr.addstr(row, col, content[:cols - col])
                else:
                    stdscr.addstr(row, col, content)
                    stdscr.clrtoeol()
            stdscr.clear()
            rows, cols = stdscr.getmaxyx()
            for index in range(rows):
                stdscr.addstr(index, 0, "")
                stdscr.clrtoeol()
            display_str = vector(raw_display_str.split("\n"))
            # for index in range(len(display_str)):
            #     display_str[index] = "[{}] ".format(index+1) + display_str[index]
            def split_len(s, l):
                if len(s) <= l:
                    return vector([s])
                ret = vector()
                while s:
                    if len(s) > l:
                        ret.append(s[:l-1] + "\\")
                    else:
                        ret.append(s[:l])
                    s = s[l:]
                return ret
            display_str = display_str.map(lambda x: split_len(x, cols-1)).flatten()
            line_bias = 0
            search_k = rows
            while True:
                display = display_str[line_bias: line_bias + search_k]
                for index in range(len(display)):
                    write_line(index, 0, display[index])
                for index in range(len(display), search_k):
                    stdscr.addstr(index, 0, "")
                    stdscr.clrtoeol()
                stdscr.addstr(0, 0, "")
                char = stdscr.get_wch()
                if char == "\x1b" or char == curses.KEY_EXIT or char == "q" or char == "`":
                    return
                elif char == curses.KEY_DOWN:
                    line_bias = max(min(line_bias + 1, len(display_str) - search_k), 0)
                elif char == curses.KEY_UP:
                    line_bias = max(line_bias - 1, 0)
                elif char == "G":
                    line_bias = max(0, len(display_str) - search_k)
                elif char == "g":
                    char == stdscr.get_wch()
                    if char == "g":
                        line_bias = 0
                    if char == "\x1b" or char == curses.KEY_EXIT or char == "q" or char == "`":
                        return
                    else:
                        continue
                elif isinstance(char, str) and char.isdigit():
                    num = 0
                    while isinstance(char, str) and char.isdigit():
                        num = num * 10 + int(char)
                        char = stdscr.get_wch()
                    if char == "G":
                        line_bias = max(min(num - search_k // 6, len(display_str) - search_k), 0)
                    else:
                        continue
                else:
                    continue
            return
        else:
            str_display = dict()
            str_search = dict()
            sorted_key = dict()
            parent = touch(lambda: obj.__mro__[1], None)
            parent_dir = vector(dir(parent) if parent else [])
            def is_overridden(obj, parent, func_name):
                if parent is None:
                    return False
                if raw_function(eval("obj.{}".format(func_name))) != raw_function(eval("parent.{}".format(func_name))):
                    return True
                return False
            def is_property(func):
                if isinstance(raw_function(func), property):
                    return True
                return False

            space_parameter = max(10, int(stdscr.getmaxyx()[0] / 10))
            for item in temp:
                if isinstance(original_obj, content_type) and item == "content":
                    str_display[item] = "[new] [*] content" + " " *  max(1, space_parameter - len("content")) + "| " +  str(original_obj).replace("\n", " ")[:500]
                    str_search[item] = "[new] content"
                    sorted_key[item] = -1
                    continue
                elif enhanced and list_like(original_obj) and item == "content":
                    str_display[item] = "[new] [*] content" + " " *  max(1, space_parameter - len("content")) + "| " +  str(original_obj).replace("\n", " ")[:500]
                    str_search[item] = "[new] content"
                    sorted_key[item] = -1
                    continue
                if item in parent_dir:
                    if is_overridden(obj, parent, item):
                        str_display[item] = "[overridden] "
                        str_search[item] = "[overridden] "
                        sorted_key[item] = 10
                    else:
                        str_display[item] = "[inherited] "
                        str_search[item] = "[inherited] "
                        sorted_key[item] = 20
                else:
                    str_display[item] = "[new] "
                    str_search[item] = "[new] "
                    sorted_key[item] = 0

                if item in extra_temp:
                    str_display[item] = str_display[item] + "[A] "
                    str_search[item] = str_search[item] + "[A] " + item
                    sorted_key[item] += 0
                    try:
                        str_display[item] = str_display[item] + item + " " * max(1, space_parameter - len(item)) + "| [{}] ".format(class_name(original_obj.__getattribute__(item))) + str(original_obj.__getattribute__(item)).replace("\n", " ")
                    except:
                        try:
                            str_display[item] = str_display[item] + item + " " * max(1, space_parameter - len(item)) + "| [{}] ".format(class_name(original_obj.__getattribute__(item)))
                        except:
                            str_display[item] = str_display[item] + item
                    continue

                func = eval("obj.{}".format(item))
                if is_property(func):
                    str_display[item] = str_display[item] + "[P] " + item
                    str_search[item] = str_search[item] + "[P] " + item
                    sorted_key[item] += 1
                    if original_obj is not None:
                        try:
                            str_display[item] = str_display[item] + " " * max(1, space_parameter - len(item)) + "| [{}] ".format(class_name(original_obj.__getattribute__(item))) + str(original_obj.__getattribute__(item)).replace("\n", " ")
                        except:
                            try:
                                str_display[item] = str_display[item] + " " * max(1, space_parameter - len(item)) + "| [{}] ".format(class_name(original_obj.__getattribute__(item)))
                            except:
                                str_display[item] = str_display[item] + " " * max(1, space_parameter - len(item)) + "| [unk]"
                elif inspect.ismethod(func):
                    str_display[item] = str_display[item] + "[M] " + item
                    str_search[item] = str_search[item] + "[M] " + item
                    sorted_key[item] += 3
                elif inspect.isfunction(func) or inspect.isroutine(func):
                    str_display[item] = str_display[item] + "[F] " + item
                    str_search[item] = str_search[item] + "[F] " + item
                    str_display[item] = str_display[item] + " " * max(1, space_parameter - len(item)) + "| {}".format(get_args_str(func, item))
                    sorted_key[item] += 4
                # elif inspect.isroutine(func):
                #     str_display[item] = str_display[item] + "[F] " + item
                #     str_search[item] = str_search[item] + "[F] " + item
                #     sorted_key[item] += 4
                elif inspect.isclass(func):
                    str_display[item] = str_display[item] + "[C] " + item
                    str_search[item] = str_search[item] + "[C] " + item
                    sorted_key[item] += 5
                elif inspect.ismodule(func):
                    str_display[item] = str_display[item] + "[Module] " + item
                    str_search[item] = str_search[item] + "[Module] " + item
                    sorted_key[item] += 6
                elif inspect.isgenerator(func):
                    str_display[item] = str_display[item] + "[G] " + item
                    str_search[item] = str_search[item] + "[G] " + item
                    sorted_key[item] += 7
                elif original_obj is not None and item in class_temp:
                    str_display[item] = str_display[item] + "[D] " + item
                    str_search[item] = str_search[item] + "[D] " + item
                    try:
                        str_display[item] = str_display[item] + " " * max(1, space_parameter - len(item)) + "| [{}] ".format(class_name(original_obj.__getattribute__(item))) + str(original_obj.__getattribute__(item)).replace("\n", " ")
                    except:
                        try:
                            str_display[item] = str_display[item] + " " * max(1, space_parameter - len(item)) + "| [{}] ".format(class_name(original_obj.__getattribute__(item)))
                        except:
                            str_display[item] = str_display[item] + item
                    sorted_key[item] += 2
                elif isinstance(func, str):
                    str_display[item] = str_display[item] + "[S] " + item + " " * max(1, space_parameter - len(item)) + "| \"{}\"".format(func)
                    str_search[item] = str_search[item] + "[S] " + item
                elif isinstance(func, (int, float)):
                    str_display[item] = str_display[item] + "[N] " + item + " " * max(1, space_parameter - len(item)) + "| {}".format(func)
                    str_search[item] = str_search[item] + "[N] " + item
                else:
                    str_display[item] = str_display[item] + "[U] " + item + " " * max(1, space_parameter - len(item)) + "| [{}]".format(class_name(func))
                    str_search[item] = str_search[item] + "[U] " + item
                str_display[item] = str_display[item].replace("\n", " ")[:500]
                str_search[item] = str_search[item].replace("\n", " ")[:500]

            # return str_search, str_display, sorted_key

            # str_search, str_display, sorted_key = curses.wrapper(temp_c_main)

            def display_info(me, query: str, selected: str):
                result = me.map_index_from(selected).map(lambda x: sorted_key[x]).filter(lambda x: x > 0).map(lambda x: x // 10).count_all()
                ret = vector()
                ret.append("# new: {}".format(result.get(0,0)))
                ret.append("# overridden: {}".format(result.get(1,0)))
                ret.append("# inherited: {}".format(result.get(2,0)))
                ret.append("prefix: " + prefix)
                return ret
            if history is None:
                history = dict()
            f_ret = temp.fuzzy_search(str_func=lambda x: str_search[x], str_display=lambda x: str_display[x], pre_sorted_function=lambda x: (sorted_key[x], x), display_info=display_info, history=history, return_tuple=True, stdscr=stdscr)
            if f_ret is not None:
                if len(f_ret) == 2:
                    func = f_ret[-1]
                    if func == "content" and isinstance(original_obj, content_type):
                        ret = vhelp(original_obj, only_content=True, stdscr=stdscr, enhanced=enhanced)
                        if ret is not None:
                            return ret
                        vhelp(original_obj, history=history, only_content=False, prefix=prefix, stdscr=stdscr, enhanced=enhanced)
                        return
                    elif func == "content" and enhanced and list_like(original_obj):
                        ret = vhelp(original_obj, only_content=True, stdscr=stdscr, enhanced=True)
                        if ret is not None:
                            return ret
                        vhelp(original_obj, history=history, only_content=False, prefix=prefix, stdscr=stdscr, enhanced=True)
                        return
                    ret = None
                    if func in extra_temp:
                        searched = original_obj.__getattribute__(func)
                    else:
                        searched = eval("obj.{}".format(func))
                    ret = vhelp(searched, only_content=True, prefix=prefix + "." + func, stdscr=stdscr, enhanced=enhanced)
                    if ret:
                        return ".{}".format(func) + ret
                    if original_obj is not None:
                        ret = vhelp(original_obj, history=history, only_content=only_content, prefix=prefix, stdscr=stdscr, enhanced=enhanced)
                    else:
                        ret = vhelp(obj, history=history, only_content=only_content, prefix=prefix, stdscr=stdscr, enhanced=enhanced)
                    if ret:
                        return ret
                elif len(f_ret) == 3:
                    func = f_ret[-1]
                    return ".{}".format(func)
            else:
                return

def generator_wrapper(*args, **kwargs):
    if len(args) == 1 and callable(raw_function(args[0])):
        func = raw_function(args[0])
        @wraps(func)
        def wrapper(*args, **kwargs):
            ret = func(*args, **kwargs)
            return ctgenerator(ret)
        return wrapper
    else:
        raise TypeError("function is not callable")

class ctgenerator:

    @staticmethod
    def _generate(iterable):
        for x in iterable:
            yield x

    @staticmethod
    def _combine_generator(*args):
        for generator in args:
            for x in generator:
                yield x

    def __init__(self, generator):
        if isinstance(generator, GeneratorType):
            self.generator = generator
        elif isinstance(generator, ctgenerator):
            self.generator = generator.generator
        elif "__iter__" in generator.__dir__():
            self.generator = ctgenerator._generate(generator)
        else:
            raise TypeError("not a generator")

    @generator_wrapper
    def map(self, func: Callable, *args, default=NoDefault) -> "ctgenerator":
        if func is None:
            for x in self.generator:
                yield x
        if len(args) > 0:
            func = chain_function((func, *args))
        if not isinstance(default, EmptyClass):
            for x in self.generator:
                yield touch(lambda: func(x), default=default)
        else:
            for x in self.generator:
                yield func(x)

    @generator_wrapper
    def filter(self, func=None) -> "ctgenerator":
        if func is None:
            for x in self.generator:
                yield x
        for x in self.generator:
            if func(x):
                yield x

    def reduce(self, func, default=None):
        try:
            init_value = next(self.generator)
        except:
            return default
        ret = init_value
        for x in self.generator:
            ret = func(ret, x)
        return ret

    def apply(self, func) -> None:
        for x in self.generator:
            func(x)

    def __iter__(self):
        for x in self.generator:
            yield x

    def __next__(self):
        return next(self.generator)

    def __add__(self, other):
        if not isinstance(other, ctgenerator):
            other = ctgenerator(other)
        return ctgenerator(ctgenerator._combine_generator(self.generator, other.generator))

    def vector(self):
        return vector(self)

    def sum(self, default=None):
        return self.reduce(lambda x, y: x+y, default)

class fuzzy_obj:

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except:
            raise RuntimeError("{} is not a method/attribute of {}, the most similar name is {}".format(name, str(self.__class__)[8:-2].rpartition(".")[-1], vector(dir(self)).fuzzy_search(name, 3)))
