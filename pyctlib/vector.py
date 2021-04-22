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
""".split()

from types import GeneratorType
from typing import List
from collections import Counter
from functools import wraps, reduce, partial
from .touch import touch, crash
import copy
import numpy as np
from pyoverload import iterable
from tqdm import tqdm, trange
from fuzzywuzzy import fuzz
from fuzzywuzzy.fuzz import WRatio
import curses
import re
import sys
import math
from typing import overload, Callable, Iterable, Union
import traceback
import inspect
import os
from .strtools import delete_surround

"""
Usage:
from pyctlib.vector import *
from pyctlib import touch
"""

def totuple(x, depth=1):
    if not iterable(x):
        x = (x, )
    if depth == 1:
        if iterable(x) and len(x) == 1 and iterable(x[0]):
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

class IndexMapping:

    def __init__(self, index_map: list=None, range_size: int=-1, reverse: bool=False):
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
        if index_map is None:
            assert range_size == -1
            self.__index_map = None
            self.__index_map_reverse = None
            self.__range_size = 0
            self.__domain_size = 0
            return
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

    def reverse(self):
        ret = IndexMapping()
        ret.__index_map = self._index_map_reverse
        ret.__index_map_reverse = self._index_map
        ret.__range_size = self.__domain_size
        ret.__domain_size = self.__range_size
        return ret

    @staticmethod
    def from_slice(index: slice, length):
        if index.start is None:
            start = 0
        else:
            start = index.start
            if start < 0:
                start = length + start
        if index.stop is None:
            stop = length
        else:
            stop = index.stop
            if stop < 0:
                stop = length + stop
            stop = min(max(stop, -1), length)
        if index.step is None:
            step = 1
        else:
            step = index.step
        assert step != 0
        if start < 0 and stop <= 0:
            return IndexMapping(vector(), range_size=length, reverse=True)
        if start >= length and stop >= length - 1:
            return IndexMapping(vector(), range_size=length, reverse=True)
        if (stop - start) * step <= 0:
            return IndexMapping(vector(), range_size=length, reverse=True)
        temp = list()
        current_index = start
        current_nu = 0
        while (stop - current_index) * step > 0:
            if 0 <= current_index < length:
                temp.append(current_index)
            current_index += step
        return IndexMapping(temp, range_size=length, reverse=True)

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
        return self.__index_map_reverse

    @property
    def index_map(self) -> Union[list, None]:
        if self._index_map is None and self._index_map_reverse is None:
            return None
        if self._index_map is not None:
            return self._index_map
        else:
            self.__index_map = self._reverse_mapping(self._index_map_reverse, range_size=self.domain_size)
            return self._index_map

    @property
    def index_map_reverse(self) -> Union[list, None]:
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
        if self._index_map is not None and other._index_map is not None:
            forward = True
        elif self._index_map_reverse is not None and other._index_map_reverse is not None:
            forward = False
        elif self._index_map is not None and other._index_map_reverse is not None:
            forward = True
        elif self._index_map_reverse is not None and other.index_map is not None:
            forward = False

        if forward:
            temp = [-1] * self.domain_size
            for index, to in enumerate(self.index_map):
                if 0 <= to < other.domain_size:
                    temp[index] = other.index_map[to]
            return IndexMapping(temp, range_size=other.range_size, reverse=False)
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
        return self._index_map is None and self._index_map_reverse is None

    def __str__(self):
        if self.isidentity:
            return "id()"
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
        if self._index_map is None:
            if self.domain_size > self.range_size * 4:
                return self._index_map_reverse.index(index)
        return self.index_map[index]

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

    def __init__(self, *args, recursive=False, index_mapping=IndexMapping(), allow_undefined_value=False, content_type=NoDefault):
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
        self.clear_appendix()
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
            else:
                try:
                    list.__init__(self, args[0])
                except:
                    list.__init__(self, args)
        else:
            list.__init__(self, args)

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

    def filter(self, func=None, ignore_error=True, func_self=None):
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
                return self.map_index(index_mapping)
            filtered_index = [index for index, a in enumerate(self) if new_func(a)]
            index_mapping = IndexMapping(filtered_index, reverse=True, range_size=self.length)
            return self.map_index(index_mapping)
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

    def filter_(self, func=None, ignore_error=True):
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

    def test(self, func, *args):
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

    def testnot(self, func, *args):
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

    # def map_with_self(self, func: Callable, func_self=lambda x: x, default=NoDefault, processing_bar=False):
    #     input_from_self = func_self(self)
    #     def temp(x):
    #         return func(x, input_from_self)
    #     return self.map(temp, default=default, processing_bar=False)

    def map(self, func: Callable, *args, func_self=None, default=NoDefault, processing_bar=False):
        """
        generate a new vector with each element x are replaced with func(x)

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
        else:
            input_from_self = func_self(self)
            func = chain_function((func, *args))
            def new_func(x):
                return func(x, input_from_self)
        if not isinstance(default, EmptyClass):
            if processing_bar:
                return vector([touch(lambda: new_func(a), default=default) for a in tqdm(super().__iter__())], recursive=self._recursive, index_mapping=self.index_mapping, allow_undefined_value=self.allow_undefined_value)
            else:
                return vector([touch(lambda: new_func(a), default=default) for a in super().__iter__()], recursive=self._recursive, index_mapping=self.index_mapping, allow_undefined_value=self.allow_undefined_value)
        try:
            if processing_bar:
                return vector([new_func(a) for a in tqdm(super().__iter__())], recursive=self._recursive, index_mapping=self.index_mapping, allow_undefined_value=self.allow_undefined_value)
            else:
                return vector([new_func(a) for a in super().__iter__()], recursive=self._recursive, index_mapping=self.index_mapping, allow_undefined_value=self.allow_undefined_value)
        except Exception as e:
            error_info = str(e)
            error_trace = traceback.format_exc()
        for index, a in self.enumerate():
            if touch(lambda: new_func(a)) is None:
                try:
                    error_information = "Error info: {}. ".format(error_info) + "\nException raised in map function at location [{}] for element [{}] with function [{}] and default value [{}]".format(index, a, new_func, default)
                except:
                    error_information = "Error info: {}. ".format(error_info) +"\nException raised in map function at location [{}] for element [{}] with function [{}] and default value [{}]".format(index, "<unknown>", new_func, default)
                error_information += "\n" + "-" * 50 + "\n" + error_trace + "-" * 50

                raise RuntimeError(error_information)

    def map_(self, func: Callable, *args, func_self=None, default=NoDefault, processing_bar=False):
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

    def rmap(self, func, *args, default=NoDefault):
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

    def replace(self, element, toelement=NoDefault):
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

    def replace_(self, element, toelement=NoDefault):
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

    def check_type(self, instance):
        """check_type
        check if all the element in the vector is of type instance

        Parameters
        ----------
        instance : Type
            instance
        """
        return all(self.map(lambda x: isinstance(x, instance)))

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
    def zip(*args, index_mapping=NoDefault):
        args = totuple(args)
        ret = vector(zip(*args)).map(lambda x: totuple(x))
        if isinstance(index_mapping, EmptyClass):
            ret._index_mapping = args[0].index_mapping
        else:
            ret._index_mapping = index_mapping
        return ret

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
            return self.map_index(IndexMapping.from_slice(index, self.length))
        if isinstance(index, list):
            assert len(self) == len(index)
            return vector(zip(self, index), recursive=self._recursive, allow_undefined_value=self.allow_undefined_value).filter(lambda x: x[1]).map(lambda x: x[0])
        if isinstance(index, tuple):
            return super().__getitem__(index[0])[index[1:]]
        if isinstance(index, IndexMapping):
            return self.map_index(index)
        return super().__getitem__(index)

    def getitem(self, index: int, index_mapping: IndexMapping=None, outboundary_value=OutBoundary):
        if index < 0 or index >= self.length:
            return outboundary_value
        if not index_mapping:
            return super().__getitem__(index)
        elif index_mapping.isidentity():
            return super().__getitem__(index)
        else:
            return self.getitem(index_mapping.index_map_reverse[index], outboundary_value=outboundary_value)

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


    def ishashable(self):
        """ishashable.
        chech whether every element in the vector is hashable
        """
        if not isinstance(touch(lambda: self._hashable, NoDefault), EmptyClass):
            return self._hashable
        self._hashable = self.all(lambda x: "__hash__" in x.__dir__())
        return self._hashable

    def __hash__(self):
        """__hash__.
        get the hash value of the vector if every element in the vector is hashable
        """
        if not self.ishashable():
            raise Exception("not all elements in the vector is hashable, the index of first unhashable element is %d" % self.index(lambda x: "__hash__" not in x.__dir__()))
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

    def count(self, *args):
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

    def index(self, element):
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

    def findall(self, element):
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

    def findall_crash(self, func):
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

    def all(self, func=lambda x: x):
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

    def any(self, func=lambda x: x):
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

    def max(self, key=None, with_index=False):
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
        m_index = 0
        m_key = self._transform(self[0], key)
        for index in range(1, len(self)):
            i_key = self._transform(self[index], key)
            if i_key > m_key:
                m_key = i_key
                m_index = index
        if with_index:
            return self[m_index], m_index
        return self[m_index]


    def min(self, key=None, with_index=False):
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

    def sum(self, default=None):
        """sum.

        Parameters
        ----------
        default :
            default
        """
        if not isinstance(touch(lambda: self._sum, NoDefault), EmptyClass):
            return self._sum
        self._sum = self.reduce(lambda x, y: x + y, default)
        return self._sum

    def cumsum(self):
        """
        cumulation summation of vector
        [a_1, a_2, \ldots, a_n]
        ->
        [a_1, a_1+a_2, \ldots, a_1+a_2+\ldots+a_n]
        """
        return self.cumulative_reduce(lambda x, y: x + y)

    def norm(self, p=2):
        """
        norm of vector
        is equivalent to
        self.map(lambda x: abs(x) ** p).sum() ** (1/p)
        """
        if touch(lambda: self._norm[p], None):
            return self._norm[p]
        self._norm[p] = math.pow(self.map(lambda x: math.pow(abs(x), p)).sum(), 1/p)
        return self._norm[p]

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

    def onehot(self, max_length=-1, default_dict={}):
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

    def sort(self, key=lambda x: x):
        if key == None:
            return self
        if self.length == 0:
            return self
        temp = sorted(vector.zip(self, vector.range(self.length)), key=lambda x: key(x[0]))
        index_mapping_reverse = [x[1] for x in temp]
        index_mapping = IndexMapping(index_mapping_reverse, reverse=True)
        return self.map_index(index_mapping)

    def sort_(self, key=lambda x: x):
        if key == None:
            return
        if self.length == 0:
            return
        temp = sorted(vector.zip(self, vector.range(self.length)), key=lambda x: key(x[0]))
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
            if len(array.shape) == 1:
                return vector(array.tolist())
            else:
                return vector(list(array)).map(lambda x: vector.from_numpy(x))
        except Exception as e:
            print("warning: input isn't pure np.ndarray")
            return vector(array.tolist())

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
    def range(stop): ...

    @overload
    def range(start, stop): ...

    @overload
    def range(start, stop, step): ...

    @staticmethod
    def range(*args):
        """range.

        Parameters
        ----------
        args :
            args
        """
        return vector(range(*args))

    @property
    def shape(self):
        """shape.
        """
        if not isinstance(touch(lambda: self._shape, NoDefault), EmptyClass):
            return self._shape
        if all(not isinstance(x, vector) for x in self):
            self._shape = (self.length, )
            return self._shape
        if any(not isinstance(x, vector) for x in self):
            self._shape = "undefined"
            return self._shape
        if not self.map(lambda x: x.shape).all_equal():
            self._shape = "undefined"
            return self._shape
        if self[0].shape is None:
            self._shape = "undefined"
            return self._shape
        self._shape = (self.length, *(self[0].shape))
        return self._shape

    def append(self, element):
        """append.

        Parameters
        ----------
        args :
            args
        """
        self.clear_appendix()
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
        if not isinstance(self.content_type, EmptyClass):
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
        self._shape = NoDefault
        self._hashable = NoDefault
        self._set = NoDefault
        self._sum = NoDefault
        self._norm = dict()

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
            return self
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

    def split_random(self, *args):
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
        return self.shuffle().split(cumsum).map_index(sorted_index_mapping.reverse())

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
        if not self.allow_undefined_value:
            assert all(0 <= index < self.length for index in index_mapping.index_map_reverse)
            ret = vector([self[index] for index in index_mapping.index_map_reverse], recursive=self._recursive, index_mapping=self.index_mapping.map(index_mapping), allow_undefined_value=False)
            return ret
        else:
            ret = vector([self[index] if index >= 0 else UnDefined for index in index_mapping.index_map_reverse], recursive=self._recursive, index_mapping=self.index_mapping.map(index_mapping), allow_undefined_value=True)
            return ret

    def map_index_(self, index_mapping: "IndexMapping"):
        """
        inplacement implementation of map_index
        """
        assert isinstance(index_mapping, IndexMapping)
        if index_mapping.isidentity:
            return self
        assert self.length == index_mapping.domain_size
        if not self.allow_undefined_value:
            assert all(0 <= index < self.length for index in index_mapping.index_map_reverse)
            ret = vector([self[index] for index in index_mapping.index_map_reverse], recursive=self._recursive, index_mapping=self.index_mapping.map(index_mapping), allow_undefined_value=False)
        else:
            ret = vector([self[index] if index >= 0 else UnDefined for index in index_mapping.index_map_reverse], recursive=self._recursive, index_mapping=self.index_mapping.map(index_mapping), allow_undefined_value=True)
        self.clear_appendix()
        super().clear()
        super().extend(ret)
        self._index_mapping = ret.index_mapping

    def original_index(self, index):
        if self.index_mapping.isidentity:
            return index
        return self.index_mapping.index_map_reverse[index]

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
        self.map_index(x.index_mapping)

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

    def __str__(self):
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
        if not isinstance(touch(lambda: self._set, NoDefault), EmptyClass):
            return self._set
        if not self.ishashable():
            raise RuntimeError("this vector is not hashable")
        self._set = set(self)
        return self._set

    def __contains__(self, item):
        if self.ishashable():
            return item in self.set()
        return super().__contains__(item)

    def __bool__(self):
        return self.length > 0

    def function_search(self, search_func, query="", max_k=NoDefault, str_func=str, str_display=None, display_info=None, sorted_function=None, pre_sorted_function=None, history=None, show_line_number=False, return_tuple=False):
        self = self.sort(key=pre_sorted_function)
        if str_display is None:
            str_display = str_func
        if len(query) > 0:
            candidate = self.clear_index_mapping().map(str_func)
            selected = search_func(candidate, query)
            if not selected:
                return None
            if isinstance(max_k, EmptyClass):
                return self.map_index_from(selected).sort(sorted_function)[0]
            else:
                return self.map_index_from(selected).sort(sorted_function)[:max_k]
        else:
            candidate = self.clear_index_mapping().map(str_func)
            def c_main(stdscr: "curses._CursesWindow"):
                def write_line(row, col=0, content=""):
                    content = " ".join(vector(content.split("\n")).map(lambda x: x.strip()))
                    if len(content) >= cols:
                        stdscr.addstr(row, col, content[:cols])
                    else:
                        stdscr.addstr(row, col, content)
                        stdscr.clrtoeol()
                stdscr.clear()
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
                result = self.map_index_from(selected).sort(key=sorted_function)[display_bias:display_bias + search_k]

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
                            display_str = "[{}] {}".format(result.original_index(index), str_display(result[index]))
                        else:
                            display_str = str_display(result[index])
                        if index == select_number:
                            write_line(1 + index, 0, "* " + display_str)
                        else:
                            write_line(1 + index, 0, display_str)
                        assert index < search_k
                    for index in range(len(result), search_k):
                        write_line(1 + index, 0, "")
                    write_line(rows-1, cols-5, content=str(char))

                    def new_len(x): return 1 + int(u'\u4e00' < x < u'\u9fff')
                    new_x_bias = sum([new_len(t) for t in query[:x_bias]])
                    stdscr.addstr(0, x_init + new_x_bias, "")
                    search_flag = False
                    char = stdscr.get_wch()
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
                    elif char == curses.KEY_UP:
                        if select_number == 2 and display_bias > 0:
                            display_bias -= 1
                            result = self.map_index_from(selected).sort(key=sorted_function)[display_bias:display_bias + search_k]
                        else:
                            select_number = max(select_number - 1, 0)
                    elif char == curses.KEY_DOWN:
                        if select_number == search_k - 3 and display_bias + search_k < selected.length:
                            display_bias += 1
                            result = self.map_index_from(selected).sort(key=sorted_function)[display_bias:display_bias + search_k]
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
                        result = self.map_index_from(selected).sort(key=sorted_function)[display_bias:display_bias + search_k]
                    elif char == 339:
                        # page up
                        decrease_amount = max(min(search_k, display_bias), 0)
                        display_bias -= decrease_amount
                        result = self.map_index_from(selected).sort(key=sorted_function)[display_bias:display_bias + search_k]
                    elif char == 262:
                        # home
                        display_bias = 0
                        select_number = 0
                        result = self.map_index_from(selected).sort(key=sorted_function)[display_bias:display_bias + search_k]
                    elif char == 360:
                        # end
                        display_bias = max(selected.length - search_k, 0)
                        result = self.map_index_from(selected).sort(key=sorted_function)[display_bias:display_bias + search_k]
                    else:
                        raise RuntimeError()
                        pass

                    try:
                        if search_flag:
                            selected = search_func(candidate, query)
                            result = self.map_index_from(selected).sort(key=sorted_function)[display_bias:display_bias + search_k]
                    except Exception as e:
                        error_info = str(e)
                    else:
                        error_info = ""


            return curses.wrapper(c_main)

    def regex_search(self, query="", max_k=NoDefault, str_func=str, str_display=None, display_info=None, sorted_function=None, pre_sorted_function=None, history=None, show_line_number=False, return_tuple=False):

        def regex_function(candidate, query):
            if len(query) == 0:
                return candidate
            regex = re.compile(query)
            selected = candidate.filter(lambda x: regex.search(x), ignore_error=False).sort(len)
            return selected

        return self.function_search(regex_function, query=query, max_k=max_k, str_func=str_func, str_display=str_display, display_info=display_info, sorted_function=sorted_function, pre_sorted_function=pre_sorted_function, history=history, show_line_number=show_line_number, return_tuple=return_tuple)

    def fuzzy_search(self, query="", max_k=NoDefault, str_func=str, str_display=None, display_info=None, sorted_function=None, pre_sorted_function=None, history=None, show_line_number=False, return_tuple=False):

        def fuzzy_function(candidate, query):
            upper = any(x.isupper() for x in query)
            if len(query) == 0:
                return candidate
            else:
                candidate = candidate.filter(lambda x: query[0] in x)
                if not upper:
                    candidate = candidate.map(lambda x: x.lower())
            if len(candidate) < 1000:
                partial_ratio = candidate.map(lambda x: (fuzz.partial_ratio(x, query), x))
                selected = partial_ratio.filter(lambda x: x[0] > 49)
            else:
                if len(query) == 1:
                    return candidate.map(lambda x: 100)
                if len(candidate) > 5000:
                    candidate = candidate.filter(lambda x: query[:2] in x)
                else:
                    candidate = candidate.filter(lambda x: query[0] in x and query[1] in x)
                if len(query) <= 3:
                    partial_ratio = candidate.map(lambda x: (fuzz.ratio(x, query) * (len(x) / len(query)) ** 0.8, x))
                elif len(query) <= 10:
                    partial_ratio = candidate.map(lambda x: (fuzz.ratio(x, query) * (len(x) / len(query)) ** 0.6, x))
                else:
                    partial_ratio = candidate.map(lambda x: (fuzz.ratio(x, query) * (len(x) / len(query)) ** 0.5, x))
                selected = partial_ratio.filter(lambda x: x[0] > 49)
            score = selected.map(lambda x: 100 * (x[0] == 100) + x[0] * min(1, len(x[1]) / len(query)) * min(1, len(query) / len(x[1])) ** 0.3, lambda x: round(x * 10) / 10).sort(lambda x: -x)
            return score

        return self.function_search(fuzzy_function, query=query, max_k=max_k, str_func=str_func, str_display=str_display, display_info=display_info, sorted_function=sorted_function, pre_sorted_function=pre_sorted_function, history=history, show_line_number=show_line_number, return_tuple=return_tuple)

    def get_size(self):
        return self.rmap(sys.getsizeof).reduce(lambda x, y: x + y, first=0)

    def __dir__(self):
        return vector(super().__dir__())

    @staticmethod
    def search_content(obj, history=None):
        assert isinstance(obj, (list, vector, set, dict, tuple))
        if isinstance(obj, (list, vector, set, tuple)):
            vector_obj = vector(obj)
            selected = vector_obj.fuzzy_search(history=history, show_line_number=True, return_tuple=True)
            return selected
        elif isinstance(obj, dict):
            vector_obj = vector(obj.keys())
            selected = vector_obj.fuzzy_search(str_display=lambda x: "[{}]: {}".format(x, obj[x]), history=history, return_tuple=True)
            return selected

    @staticmethod
    def help(obj=None, history=None, only_content=False):
        if obj is None:
            vector.help(vector)
        else:
            content_type = (list, vector, set, tuple, dict)
            if only_content:
                if isinstance(obj, content_type):
                    if history is None:
                        history = {}
                    selected = vector.search_content(obj, history=history)
                    if selected:
                        if len(selected) == 2:
                            if isinstance(obj, (list, vector, set, tuple)):
                                ret = vector.help(selected[1], only_content=True)
                                if ret is not None:
                                    if isinstance(obj, (list, vector, tuple)):
                                        return "[{}]".format(selected[0]) + ret
                                    else:
                                        return ".set<{}>".format(selected[1]) + ret
                                ret = vector.help(obj, history=history, only_content=True)
                                if ret is not None:
                                    return ret
                            else:
                                value = obj.get(selected[1], None)
                                if value is None:
                                    return
                                ret = vector.help(value, only_content=True)
                                if ret is not None:
                                    return "[\"{}\"]".format(selected[1]) if isinstance(selected[1], str) else "[{}]".format(selected) + ret
                                ret = vector.help(obj, history=history, only_content=True)
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
                            else:
                                return ".set<{}>".format(selected[1])
                    return
                if isinstance(obj, (int, str, float)):
                    raw_display_str = str(obj).replace("\t", "    ")
                    str_length = len(raw_display_str)
                    line_number = raw_display_str.count("\n") + 1
                    def c_main(stdscr: "curses._CursesWindow"):
                        def write_line(row, col=0, content=""):
                            if col >= cols:
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
                                return s
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
                    curses.wrapper(c_main)
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
                else:
                    temp = class_temp + extra_temp
            else:
                extra_temp = vector()
                temp = vector(dir(obj)).unique().filter(lambda x: len(x) > 0 and x[0] != "_").test(lambda x: testfunc(obj, x))
            if len(temp) == 0:
                help(obj)
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

                space_parameter = 15
                for item in temp:
                    if isinstance(original_obj, content_type) and item == "content":
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

                def display_info(me, query, selected):
                    result = me.map_index_from(selected).map(lambda x: sorted_key[x]).filter(lambda x: x > 0).map(lambda x: x // 10).count_all()
                    ret = vector()
                    ret.append("# new: {}".format(result.get(0,0)))
                    ret.append("# overridden: {}".format(result.get(1,0)))
                    ret.append("# inherited: {}".format(result.get(2,0)))
                    return ret
                if history is None:
                    history = dict()
                f_ret = temp.fuzzy_search(str_func=lambda x: str_search[x], str_display=lambda x: str_display[x], pre_sorted_function=lambda x: (sorted_key[x], x), display_info=display_info, history=history, return_tuple=True)
                if f_ret is not None:
                    if len(f_ret) == 2:
                        func = f_ret[-1]
                        if func == "content" and isinstance(original_obj, content_type):
                            ret = vector.help(original_obj, only_content=True)
                            if ret is not None:
                                return ret
                            vector.help(original_obj, history=history, only_content=False)
                            return
                        ret = None
                        if func in extra_temp:
                            searched = original_obj.__getattribute__(func)
                        else:
                            searched = eval("obj.{}".format(func))
                        if isinstance(searched, (list, vector, tuple, set, dict)):
                            ret = vector.help(searched, only_content=True)
                        elif func in extra_temp:
                            ret = vector.help(searched, only_content=True)
                        elif "module" in str(type(searched)):
                            ret = vector.help(searched, only_content=True)
                        elif inspect.isclass(searched):
                            ret = vector.help(searched, only_content=True)
                        elif isinstance(searched, vector):
                            ret = vector.help(searched, only_content=True)
                        elif isinstance(searched, (int, float, str)):
                            vector.help(searched, only_content=True)
                        else:
                            help(searched)
                            if os.name == "nt":
                                return
                        if ret:
                            return ".{}".format(func) + ret
                        if original_obj is not None:
                            ret = vector.help(original_obj, history=history, only_content=only_content)
                        else:
                            ret = vector.help(obj, history=history, only_content=only_content)
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
    def map(self, func: callable, *args, default=NoDefault) -> "ctgenerator":
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

vhelp = vector.help
