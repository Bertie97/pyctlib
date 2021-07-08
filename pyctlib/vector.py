#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package <main>
##############################
__all__ = """
    totuple
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
from .touch import touch, crash, once
import copy
import numpy as np
from pyoverload import iterable
from tqdm import tqdm, trange
# from fuzzywuzzy import fuzz
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
from .strtools import delete_surround
from .wrapper import empty_wrapper
import os.path
import time
import pydoc
from collections.abc import Hashable
from matplotlib.axes._subplots import Axes
try:
    import numba as nb
    jit = nb.jit
except:
    jit = empty_wrapper

"""
Usage:
from pyctlib.vector import *
from pyctlib import touch
"""

def list_like(obj):
    return "__getitem__" in dir(obj) and "__len__" in dir(obj) and "__iter__" in dir(obj)

def totuple(x, depth=1):
    if isinstance(x, types.GeneratorType):
        x = tuple(x)
    if not iterable(x):
        x = (x, )
    if depth == 1:
        if iterable(x) and len(x) == 1 and iterable(x[0]):
            return tuple(x[0])
        if iterable(x) and len(x) == 1 and isinstance(x, types.GeneratorType):
            return tuple(x[0])
        else:
            return tuple(x)
    if depth == 0:
        return tuple(x)
    temp = vector(x).map(lambda t: totuple(t, depth=depth-1))
    return temp.reduce(lambda x, y: x + y)

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
    return isinstance(x, Hashable)

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
    return np.clip(x, x_low, x_upper)

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
    def __init__(self, list, *, recursive=False, index_mapping=IndexMapping(), allow_undefined_value=False, content_type=NoDefault) -> "vector":
        ...

    @overload
    def __init__(self, tuple, *, recursive=False, index_mapping=IndexMapping(), allow_undefined_value=False, content_type=NoDefault) -> "vector":
        ...

    @overload
    def __init__(self, *data, recursive=False, index_mapping=IndexMapping(), allow_undefined_value=False, content_type=NoDefault) -> "vector":
        ...

    def __init__(self, *args, recursive=False, index_mapping=IndexMapping(), allow_undefined_value=False, content_type=NoDefault, str_function=None):
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
        # self.clear_appendix()
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
                return vector()
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

    def map(self, func: Callable, *args, func_self=None, default=NoDefault, processing_bar=False, register_result=False, split_tuple=False, filter_function=None) -> "vector":
        """
        generate a new vector with each element x are replaced with func(x)

        Parameters
        ----------
        func : callable
        args :
            more function
        func_self:
            if func_self is not None, then func_self(self) will be passed as another argument to func
            x -> func(x, func_self(self))
        default :
            default value used when func cause an error
        register_result:
            It can be True/False/<str>
            If it is set, second time you call the same map function, result will be retrieved from the buffer.
            warning:
            if register_result = True, plz not call map with lambda expression in the argument, for example:
                wrong: v.map(lambda x: x+1, register_result=True)
                right: 1. f = lambda x: x + 1
                          v.map(f, register_result)
                       2. v.map(lambda x: x+1, register_result="plus 1")
        filter_function:
            if filter_function(index, element) is False, map will not be executed.

        Example:
        ----------
        vector([0,1,2]).map(lambda x: x ** 2)
        will produce [0,1,4]
        """
        if func is None:
            return self
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
        for index, a in self.enumerate():
            if filter_function is not None and not filter_function(index, a):
                continue
            if touch(lambda: new_func(a)) is None:
                try:
                    error_information = "Error info: {}. ".format(error_info) + "\nException raised in map function at location [{}] for element [{}] with function [{}] and default value [{}]".format(index, a, new_func, default)
                except:
                    error_information = "Error info: {}. ".format(error_info) +"\nException raised in map function at location [{}] for element [{}] with function [{}] and default value [{}]".format(index, "<unknown>", new_func, default)
                error_information += "\n" + "-" * 50 + "\n" + error_trace + "-" * 50

                raise RuntimeError(error_information)
        return vector()

    def map_k(self, func, k, overlap=True, split_tuple=True) -> "vector":
        if self.length < k:
            return vector()
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

    def rmap(self, func, *args, default=NoDefault) -> "vector":
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
        if len(args) > 0:
            func = chain_function((func, *args))
        return self.map(lambda x: x.rmap(func, default=default) if isinstance(x, vector) else func(x), default=default)

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

    def __and__(self, other):
        if isinstance(other, vector):
            if self.length == other.length:
                return vector(zip(self, other)).map(lambda x: x[0] and x[1])
            raise RuntimeError("length of vector A [{}] isnot compatible with length of vector B [{}]".format(self.length, other.length))
        raise RuntimeError("can only support vector and vector")

    def __or__(self, other):
        if isinstance(other, vector):
            if self.length == other.length:
                return vector(zip(self, other)),map(lambda x: x[0] or x[1])
            raise RuntimeError("length of vector A [{}] isnot compatible with length of vector B [{}]".format(self.length, other.length))
        raise RuntimeError("can only support vector or vector")

    def __mul__(self, other):
        """__mul__.

        Usages
        -----------
        1. vector * n will repeat the vector n times
        2. vector * vector will zip the two vector
            vector([1,2,3]) * vector([4,5,6])
            will produce [(1,4),(2,5),(3,6)]
        """
        if isinstance(other, int):
            return vector(super().__mul__(times))
        if touch(lambda: self.check_type(tuple) and other.check_type(tuple)):
            return vector(zip(self, other)).map(lambda x: (*x[0], *x[1]))
        elif touch(lambda: self.check_type(tuple)):
            return vector(zip(self, other)).map(lambda x: (*x[0], x[1]))
        elif touch(lambda: other.check_type(tuple)):
            return vector(zip(self, other)).map(lambda x: (x[0], *x[1]))
        else:
            return vector(zip(self, other))

    @staticmethod
    def zip(*args, index_mapping=NoDefault) -> "vector":
        args = totuple(args)
        ret = vector(zip(*args)).map(lambda x: totuple(x), split_tuple=False)
        if isinstance(index_mapping, EmptyClass):
            ret._index_mapping = args[0].index_mapping
        else:
            ret._index_mapping = index_mapping
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

    def __pow__(self, other):
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
        """
        return vector([(i, j) for i in self for j in other])

    def __add__(self, other: list):
        """__add__.

        Parameters
        ----------
        other : list
            other
        """
        return vector(super().__add__(other))

    def _transform(self, element, func=None):
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

    def __eq__(self, other):
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

    def __neq__(self, other):
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

    def __lt__(self, element):
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

    def __gt__(self, element):
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

    def __le__(self, element):
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

    def __ge__(self, element):
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
            return self.map_index(IndexMapping(index, self.length, True))
        if isinstance(index, list):
            assert len(self) == len(index)
            return vector(zip(self, index), recursive=self._recursive, allow_undefined_value=self.allow_undefined_value).filter(lambda x: x[1]).map(lambda x: x[0])
        if isinstance(index, tuple):
            if len(index) == 1:
                return super().__getitem__(index[0])
            else:
                return super().__getitem__(index[0])[index[1:]]
        if isinstance(index, IndexMapping):
            return self.map_index(index)
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

    def __sub__(self, other):
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

    def unique(self) -> "vector":
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
        if self.ishashable():
            return vector(self.set(), recursive=False)
        explored = set() if hashable else list()
        pushfunc = explored.add if hashable else explored.append
        unique_elements = list()
        for x in self:
            if x not in explored:
                unique_elements.append(x)
                pushfunc(x)
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
        assert self.check_type(int) or self.check_type(float)
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

    def clip(self, *args, **kwargs):
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

    def sum(self, default=None):
        """sum.

        Parameters
        ----------
        default :
            default
        """
        if hasattr(self, "_vector__sum"):
            return self.__sum
        if self.check_type(int) or self.check_type(float):
            self.__sum = numba_sum(self.to_numpy())
        else:
            self.__sum = self.reduce(lambda x, y: x + y, default)
        return self.__sum

    def mean(self, default=NoDefault):
        if self.length == 0:
            if isinstance(default, EmptyClass):
                raise TypeError("vector is empty, plz set default to prevent error")
            return default
        return self.sum() / self.length

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

    def smooth(self, window_len=11, window='hanning'):
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

        if not self.check_type(int) and not self.check_type(float):
            raise ValueError("smooth only accepts 1 dimension arrays.")

        if window_len<3:
            return self

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

    def normalization(self, p=1):
        """
        normaize the vector using p-norm
        is equivalent to self.map(lambda x: x / self.norm(p))

        result is $\frac{x}{\|x\|_p}$
        """
        norm_p = self.norm(p)
        return self.map(lambda x: x / self.norm(p))

    def normalization_(self, p=1):
        """
        **inplace function:** normaize the vector using p-norm
        is equivalent to self.map(lambda x: x / self.norm(p))

        result is $\frac{x}{\|x\|_p}$
        """
        norm_p = self.norm(p)
        return self.map_(lambda x: x / self.norm(p))

    def softmax(self, beta=1):
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


    def join(self, sep: str):
        return sep.join(self)

    def flatten(self, depth=-1):
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
            if all(not isinstance(x, list) for x in array):
                return array
            return array.map(vector).map(lambda x: temp_flatten(x, depth - 1)).reduce(lambda x, y: x + y)
        return temp_flatten(self, depth)

    def reshape(self, *args):
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
    def length(self):
        """length.
        length of the vector
        """
        return len(self)

    def onehot(self, max_length: int=-1, default_dict: Dict[Any, int]={}):
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

    def sort(self, key: Callable=lambda x: x, reverse: bool=False):
        if key == None:
            return self
        if self.length == 0:
            return self
        temp = sorted(vector.zip(self, vector.range(self.length)), key=lambda x: key(x[0]), reverse=reverse)
        index_mapping_reverse = [x[1] for x in temp]
        index_mapping = IndexMapping(index_mapping_reverse, reverse=True)
        return self.map_index(index_mapping)

    def sort_(self, key=lambda x: x, reverse: bool=False):
        if key == None:
            return
        if self.length == 0:
            return
        temp = sorted(vector.zip(self, vector.range(self.length)), key=lambda x: key(x[0]), reverse=reverse)
        index_mapping_reverse = [x[1] for x in temp]
        index_mapping = IndexMapping(index_mapping_reverse, reverse=True)
        self.map_index_(index_mapping)

    def sort_by_index(self, key=lambda index: index):
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

    def sort_by_index_(self, key=lambda index: index):
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

    def sort_by_vector(self, other, func=lambda x: x):
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

    def sort_by_vector_(self, other, func=lambda x: x):
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
    def from_numpy(array):
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

    def to_numpy(self):
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

    def plot(self, ax: Optional[Axes]=None, title: Optional[str]=None, smooth: int=-1, saved_path: Optional[str]=None, legend: Optional[List[str]]=None, hline: Optional[List[str]]=None):
        """
        plot line graph for vector
        title: title of the graph
        smooth: windows size of smoothing
        saved_path: path to save the graph
        legend: list of legend string
        hline: list, can be None or contains "top" or/and "bottom", to plot a horizontal line corresponding to the biggest or smallest value
        """
        from matplotlib import pyplot as plt
        _has_ax = ax is not None
        if ax is None:
            ax = plt.gca()
            ax.clear()
        else:
            assert saved_path is None
        if self.check_type(float) or self.check_type(int):
            ax.plot(self.smooth(smooth))
        elif (self.check_type(list) or self.check_type(tuple)) and self.map(len).all_equal():
            splited_vector = self.zip_split()
            for sv in splited_vector:
                ax.plot(sv.smooth(smooth))
            if not legend:
                legend = vector.range(len(splited_vector)).map(str)
        else:
            raise ValueError
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        boundary_margin = 1 / 30 * (xmax - xmin)
        plt.xlim(-boundary_margin, self.length - 1 + boundary_margin)

        if title:
            ax.set_title(title)
        if legend:
            ax.legend(legend)
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
    def from_list(array):
        """from_list.

        Parameters
        ----------
        array :
            array
        """
        if not isinstance(array, list):
            return array
        return vector(vector.from_list(x) for x in array)

    def tolist(self):
        ret = list()
        for item in self:
            if isinstance(item, vector):
                ret.append(item.tolist())
            else:
                ret.append(item)
        return ret

    def numpy(self):
        return np.array(self)

    @overload
    @staticmethod
    def zeros(size: Iterable): ...

    @overload
    @staticmethod
    def zeros(*size): ...

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

    @overload
    @staticmethod
    def ones(size: Iterable): ...

    @overload
    @staticmethod
    def ones(*size): ...

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
    def linspace(low, high, nbins):
        return vector.from_numpy(np.linspace(low, high, nbins))

    @staticmethod
    def meshgrid(*args):
        args = totuple(args)
        if len(args) == 0:
            return vector()
        if isinstance(args[0], int):
            return vector.meshgrid(*[vector.range(d) for d in args])
        import itertools
        return vector(itertools.product(*args)).map(lambda x: x)

    @staticmethod
    def rand(*args):
        """rand.

        Parameters
        ----------
        args :
            args
        """
        args = totuple(args)
        ret = vector.from_numpy(np.random.rand(*args))
        ret._shape = args
        return ret

    @staticmethod
    def randint(low, high=None, size=(1, )):
        ret = vector.from_numpy(np.random.randint(low, high=high, size=size))
        ret._shape = totuple(size)
        return ret

    @staticmethod
    def randn(*args):
        """randn.

        Parameters
        ----------
        args :
            args
        """
        args = totuple(args)
        ret = vector.from_numpy(np.random.randn(*args))
        ret._shape = args
        return ret

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
    def from_randomwalk(start, transition_function, length):
        ret = vector([start])
        temp = start
        for index in range(length-1):
            temp = transition_function(start)
            ret.append(temp)
        return ret

    def iid(self, sample_func, length, args=()):
        return vector([sample_func(*args) for _ in range(length)])

    @property
    def shape(self):
        """shape.
        """
        if hasattr(self, "_vector__shape"):
            return self.__shape
        if all(not isinstance(x, vector) for x in self):
            self.__shape = (self.length, )
            return self.__shape
        if any(not isinstance(x, vector) for x in self):
            self.__shape = "undefined"
            return self.__shape
        if not self.map(lambda x: x.shape).all_equal():
            self.__shape = "undefined"
            return self.__shape
        if self[0].shape is None:
            self.__shape = "undefined"
            return self.__shape
        self.__shape = (self.length, *(self[0].shape))
        return self.__shape

    def append(self, element, refuse_value=NoDefault):
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

    def extend(self, other):
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

    def insert(self, location, element):
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

    def clear_appendix(self):
        touch(lambda: delattr(self, "_vector__shape"))
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
        touch(lambda: delattr(self, "_vector__shape"))
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

    def clear(self):
        """clear
        """
        self.clear_appendix()
        self._index_mapping = IndexMapping()
        super().clear()
        return self

    def remove(self, *args):
        """remove.

        Parameters
        ----------
        args :
            args
        """
        self.clear_appendix()
        super().remove(*args)
        return self

    def all_equal(self):
        """all_equal.
        """
        if self.length <= 1:
            return True
        return self.all(lambda x: x == self[0])

    @overload
    def sample(self, size: Iterable, replace=True, batch_size=1, p=None): ...

    @overload
    def sample(self, *size, replace=True, batch_size=1, p=None): ...

    def sample(self, *args, replace=True, batch_size=1, p=None):
        """sample.

        Parameters
        ----------
        args :
            args
        replace :
            replace
        p :
            p
        """
        args = totuple(args)
        if len(args) == 0:
            return vector()
        if isinstance(args[-1], bool):
            replace = args[-1]
            args = args[:-1]
        if len(args) >= 2 and isinstance(args[-2], bool) and isinstance(args[-1], (list, np.ndarray)):
            replace = args[-2]
            p = args[-1]
            args = args[:-2]
        if len(args) == 0:
            return vector()
        if batch_size > 1:
            args = (*args, batch_size)
        if len(args) == 1 and replace == False:
            index_mapping = IndexMapping(np.random.choice(vector.range(self.length), size = args, replace=False, p=p), range_size=self.length, reverse=True)
            return self.map_index(index_mapping)
        return vector(np.random.choice(self, size=args, replace=replace, p=p), recursive=False)

    def batch(self, batch_size=1, random=True, drop=True):
        if random:
            if self.length % batch_size == 0:
                return self.sample(self.length // batch_size, batch_size, replace=False)
            if drop:
                return self.sample(self.length // batch_size, batch_size, replace=False)
            else:
                return (self + self.sample(batch_size - self.length % batch_size)).batch(batch_size=batch_size, drop=True)
        else:
            return self[:(self.length - self.length % batch_size)].reshape(-1, batch_size)

    def shuffle(self):
        """
        shuffle the vector
        """
        index_mapping = IndexMapping(vector.range(self.length).sample(self.length, replace=False))
        return self.map_index(index_mapping)

    def reverse(self):
        """
        reverse the vector
        vector(0,1,2).reverse()
        will get
        [2,1,0]
        """
        index_mapping= IndexMapping(vector.range(self.length-1, -1, -1))
        return self.map_index(index_mapping)

    def reverse_(self):
        """
        **inplace function:** reverse the vector
        vector(0,1,2).reverse()
        will get
        [2,1,0]
        """
        index_mapping= IndexMapping(vector.range(self.length-1, -1, -1))
        return self.map_index_(index_mapping)

    def shuffle_(self):
        """
        **Inplace function:** shuffle the vector
        """
        index_mapping = IndexMapping(vector.range(self.length).sample(self.length, replace=False))
        self.map_index_(index_mapping)

    def split(self, *args):
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

    def split_index(self, *args):
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
            args.pop()
        assert split_num.sum() == self.length
        cumsum = split_num.cumsum()
        return tuple(self.shuffle().split_index(cumsum).map_index(sorted_index_mapping.reverse()))

    def copy(self, deep_copy=False):
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

    def map_index(self, index_mapping: "IndexMapping"):
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

    def map_index_(self, index_mapping: "IndexMapping"):
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

    def map_index_from(self, x):
        assert isinstance(x, vector)
        return self.map_index(x.index_mapping)

    def map_index_from_(self, x):
        assert isinstance(x, vector)
        self.map_index_(x.index_mapping)

    def map_reverse_index(self, reverse_index_mapping: "IndexMapping"):
        assert isinstance(reverse_index_mapping, IndexMapping)
        return self.map_index(reverse_index_mapping.reverse())

    def map_reverse_index_(self, reverse_index_mapping: "IndexMapping"):
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

    def roll(self, shift=1):
        index_mapping = IndexMapping([(index - shift) % self.length for index in range(self.length)], range_size=self.length, reverse=True)
        return self.map_index(index_mapping)

    def roll_(self, shift=1):
        index_mapping = IndexMapping([(index - shift) % self.length for index in range(self.length)], range_size=self.length, reverse=True)
        return self.map_index_(index_mapping)

    def __str__(self):
        if self.str_function is not None:
            return self.str_function(self)
        if self.shape != "undefined" and len(self.shape) > 1:
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
