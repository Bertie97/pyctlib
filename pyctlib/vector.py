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
import curses
import re

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

def raw_function(func):
    if "__func__" in dir(func):
        return func.__func__
    return func

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

def chain_function(funcs):
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
    def ret(funcs, x):
        for func in funcs:
            x = func(x)
        return x
    return partial(ret, funcs)

class IndexMapping:

    def __init__(self, index_map=None, range_size=0, reverse=False):
        if index_map is None:
            self._index_map = None
            self._index_map_reverse = None
            return
        if not reverse:
            self._index_map = index_map
            self._index_map_reverse = self._reverse_mapping(index_map, range_size=range_size)
        else:
            self._index_map_reverse = index_map
            self._index_map = self._reverse_mapping(index_map, range_size=range_size)

    def reverse(self):
        ret = IndexMapping()
        ret._index_map = self._index_map_reverse
        ret._index_map_reverse = self._index_map
        return ret

    @property
    def domain_size(self) -> int:
        return len(self.index_map)

    @property
    def range_size(self) -> int:
        return len(self.index_map_reverse)

    @property
    def index_map(self) -> list:
        return self._index_map

    @property
    def index_map_reverse(self) -> list:
        return self._index_map_reverse

    def map(self, other):
        assert isinstance(other, IndexMapping)
        if self.isidentity:
            return copy.deepcopy(other)
        if other.isidentity:
            return copy.deepcopy(self)
        ret = IndexMapping()
        ret._index_map = [-1] * self.domain_size
        for index, to in enumerate(self.index_map):
            if 0 <= to < other.domain_size:
                ret._index_map[index] = other.index_map[to]
        ret._index_map_reverse = [-1] * other.range_size
        for index, to in enumerate(other.index_map_reverse):
            if 0 <= to < self.range_size:
                ret._index_map_reverse[index] = self.index_map_reverse[to]
        return ret

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
        return self.index_map is None

    def __str__(self):
        if self.isidentity:
            return "id()"
        return str(self.index_map_reverse)

    def __repr__(self):
        return self.__str__()

    def check_valid(self):
        if self.index_map is None:
            return self.index_map_reverse is None
        for index, to in enumerate(self.index_map):
            if to != -1:
                if self.index_map_reverse[to] != index:
                    return False
        for index, to in enumerate(self.index_map):
            if to != -1:
                if self.index_map[to] != index:
                    return False
        return True

class vector(list):
    """vector
    vector is actually list in python with advanced method like map, filter and reduce
    """


    def __init__(self, *args, recursive=False, index_mapping=IndexMapping(), allow_undefined_value=False):
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
        return touch(lambda: self._index_mapping, IndexMapping())

    def filter(self, func=None, ignore_error=True):
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
        try:
            if ignore_error:
                filtered_index = [index for index, a in enumerate(self) if touch(lambda: func(a), False)]
                index_mapping = IndexMapping(filtered_index, reverse=True, range_size=self.length)
                return self.map_index(index_mapping)
            filtered_index = [index for index, a in enumerate(self) if func(a)]
            index_mapping = IndexMapping(filtered_index, reverse=True, range_size=self.length)
            return self.map_index(index_mapping)
        except Exception as e:
            error_info = str(e)
        for index, a in enumerate(self):
            if touch(lambda: func(a)) is None:
                try:
                    error_information = "Error info: {}. Exception raised in filter function at location {} for element {}".format(error_info, index, a)
                except:
                    error_information = "Error info: {}. Exception raised in filter function at location {} for element {}".format(error_info, index, "<unknown>")
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

    def map(self, func, *args, default=NoDefault, processing_bar=False):
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
        if len(args) > 0:
            func = chain_function((func, *args))
        if default is not NoDefault:
            if processing_bar:
                return vector([touch(lambda: func(a), default=default) for a in tqdm(self)], recursive=self._recursive, index_mapping=self.index_mapping, allow_undefined_value=self.allow_undefined_value)
            else:
                return vector([touch(lambda: func(a), default=default) for a in self], recursive=self._recursive, index_mapping=self.index_mapping, allow_undefined_value=self.allow_undefined_value)
        try:
            if processing_bar:
                return vector([func(a) for a in tqdm(self)], recursive=self._recursive, index_mapping=self.index_mapping, allow_undefined_value=self.allow_undefined_value)
            else:
                return vector([func(a) for a in self], recursive=self._recursive, index_mapping=self.index_mapping, allow_undefined_value=self.allow_undefined_value)
        except:
            pass
        for index, a in enum_self:
            if touch(lambda: func(a)) is None:
                try:
                    error_information = "Exception raised in map function at location [{}] for element [{}] with function [{}] and default value [{}]".format(index, a, func, default)
                except:
                    error_information = "Exception raised in map function at location [{}] for element [{}] with function [{}] and default value [{}]".format(index, "<unknown>", func, default)
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
        return self

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
    def zip(*args):
        args = totuple(args)
        return vector(zip(*args)).map(lambda x: totuple(x))

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
            return vector(super().__getitem__(index), recursive=self._recursive, allow_undefined_value=self.allow_undefined_value)
        if isinstance(index, list):
            assert len(self) == len(index)
            return vector(zip(self, index), recursive=self._recursive, allow_undefined_value=self.allow_undefined_value).filter(lambda x: x[1]).map(lambda x: x[0])
        if isinstance(index, tuple):
            return super().__getitem__(index[0])[index[1:]]
        if isinstance(index, IndexMapping):
            return self.map_index(index)
        return super().__getitem__(index)

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
        if touch(lambda: self._hashable, NoDefault) is not NoDefault:
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

    def count_all(self):
        """count_all.
        count all the elements in the vector, sorted by occurance time

        Example
        -----------
        vector([1,2,3,2,3,1]).count_all()
        will produce Counter({1: 2, 2: 2, 3: 2})
        """
        if len(self) == 0:
            return vector([], recursive=False)
        hashable = self.ishashable()
        if hashable:
            return Counter(self)
        else:
            return dict(self.unique().map(lambda x: (x, self.count(x))))

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
        if touch(lambda: self._sum, NoDefault) is not NoDefault:
            return self._sum
        self._sum = self.reduce(lambda x, y: x + y, default)
        return self._sum

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

    def reduce(self, func, default=None):
        """reduce.
        reduce the vector will func (refer to map-reduce)

        Parameters
        ----------
        func :
            func
        default :
            default

        Example
        ----------
        vector(1,2,3,4).reduce(lambda x, y: x+y)
        will produce 10
        """
        if len(self) == 0:
            return default
        temp = self[0]
        for x in self[1:]:
            temp = func(temp, x)
        return temp

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
            args.replace(-1, size // abs(reduce(lambda x, y: x*y, args)))
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
                return value
            piece_length = len(value) // target_shape[0]
            ret = vector(_reshape(value[piece_length * index: piece_length * (index + 1)], target_shape[1:]) for index in range(target_shape[0]))
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
        temp = sorted(vector.zip(self, vector.range(self.length)), key=lambda x: key(x[0]))
        index_mapping_reverse = [x[1] for x in temp]
        index_mapping = IndexMapping(index_mapping_reverse, reverse=True)
        return self.map_index(index_mapping)

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
        if touch(lambda: self._shape, NoDefault) is not NoDefault:
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

    def insert(self, *args):
        """insert.

        Parameters
        ----------
        args :
            args
        """
        self.clear_appendix()
        self._index_mapping = IndexMapping()
        super().insert(*args)
        return self

    def clear_appendix(self):
        self._shape = NoDefault
        self._hashable = NoDefault
        self._set = NoDefault
        self._sum = NoDefault

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

    def batch(self, batch_size=1, drop=True):
        if self.length % batch_size == 0:
            return self.sample(self.length // batch_size, batch_size, replace=False)
        if drop:
            return self.sample(self.length // batch_size, batch_size, replace=False)
        else:
            return (self + self.sample(batch_size - self.length % batch_size)).batch(batch_size=batch_size, drop=True)

    def shuffle(self):
        index_mapping = IndexMapping(vector.range(self.length).sample(self.length, replace=False))
        return self.map_index(index_mapping)

    def split(self, *args):
        args = totuple(args)

    def copy(self, deep_copy=False):
        if not deep_copy:
            ret = vector(self)
        else:
            ret = vector(copy.deep_copy(self))
        if self.index_mapping is not None:
            ret._index_mapping = self.index_mapping.copy()
        else:
            ret._index_mapping = IndexMapping()
        return ret

    def map_index(self, index_mapping: "IndexMapping"):
        assert isinstance(index_mapping, IndexMapping)
        if index_mapping.isidentity:
            return self
        assert self.length == index_mapping.domain_size
        if not self.allow_undefined_value:
            ret = vector([self[index] for index in index_mapping.index_map_reverse], recursive=self._recursive, index_mapping=self.index_mapping.map(index_mapping), allow_undefined_value=False)
            return ret
        else:
            ret = vector([self[index] if index > 0 else UnDefined for index in index_mapping.index_map_reverse], recursive=self._recursive, index_mapping=self.index_mapping.map(index_mapping), allow_undefined_value=True)
            return ret

    def map_index_from(self, x):
        assert isinstance(x, vector)
        return self.map_index(x.index_mapping)

    def map_reverse_index(self, reverse_index_mapping: "IndexMapping"):
        assert isinstance(reverse_index_mapping, IndexMapping)
        return self.map_index(reverse_index_mapping.reverse())

    def clear_map_index(self):
        self._index_mapping = IndexMapping()
        return self

    def unmap_index(self):
        if self.index_mapping.isidentity:
            return self
        temp_flag = self.allow_undefined_value
        if self.index_mapping.domain_size > self.index_mapping.range_size:
            self.allow_undefined_value = True
        ret = self.map_index(self.index_mapping.reverse())
        self.allow_undefined_value = temp_flag
        return ret

    def __str__(self):
        if self.shape != "undefined" and len(self.shape) > 1:
            ret: List[str] = vector()
            for index, child in self.enumerate():
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
                ret += str(child)
                if index < self.length - 1:
                    ret += ", "
            ret += "]"
            return ret

    def __repr__(self):
        return self.__str__()

    def set(self):
        if touch(lambda: self._set, NoDefault) is not NoDefault:
            return self._set
        if not self.ishashable():
            raise RuntimeError("this vector is not hashable")
        self._set = set(self)
        return self._set

    def __contains__(self, item):
        if self.ishashable():
            return item in self.set()
        return super().__contains__(item)

    def regex_search(self, question="", k=NoDefault, str_func=str):

        if len(question) > 0:
            regex = re.compile(question)
            ratio = self.map(str_func).filter(lambda x: regex.search(x), ignore_error=False)
            return self.map_index_from(ratio)
        else:
            def c_main(stdscr: "curses._CursesWindow"):
                stdscr.clear()
                question = ""
                question_done = False
                select_number = 0
                result = vector()
                rows, cols = stdscr.getmaxyx()

                stdscr.addstr(0, 0, "token to search: ")
                search_k = k
                if search_k is NoDefault:
                    search_k = int(rows * 0.8)
                for index in range(len(self[:search_k])):
                    if index == 0:
                        stdscr.addstr(index + 1, 0, "* " + str(self[index])[:100])
                    else:
                        stdscr.addstr(index + 1, 0, str(self[index])[:100])

                while True:
                    stdscr.addstr(0, 0, "token to search: ")
                    stdscr.clrtoeol()
                    stdscr.addstr(question)

                    char = stdscr.get_wch()
                    ratio = vector()
                    if isinstance(char, str) and char.isprintable():
                        question += char
                        select_number = 0
                    elif char == curses.KEY_BACKSPACE or char == "\x7f":
                        question = question[:-1]
                        select_number = 0
                    elif char == "\n":
                        if len(result) > 0:
                            return result[select_number]
                    elif char == "\x1b":
                        return None
                    elif char == curses.KEY_UP:
                        select_number = max(select_number - 1, 0)
                    elif char == curses.KEY_DOWN:
                        select_number = max(min(select_number + 1, len(result) - 1), 0)
                    elif char == curses.KEY_LEFT or char == curses.KEY_RIGHT:
                        continue
                    else:
                        continue

                    if len(question) > 0:
                        regex = touch(lambda: re.compile(question), None)
                        if regex:
                            ratio = self.map(str_func).filter(lambda x: regex.search(x), ignore_error=False)
                            result = self.map_index(ratio.index_mapping)[:search_k]
                        stdscr.addstr(search_k + 1, 0, "regex " + str(regex))
                        stdscr.clrtoeol()
                        stdscr.addstr(search_k + 2, 0, "# match: " + str(ratio.length))
                        stdscr.clrtoeol()
                    else:
                        result = self[:search_k]
                        ratio = result.map(lambda x: 0)
                    for index in range(len(result)):
                        if index == select_number:
                            stdscr.addstr(1 + index, 0, "* " + str(result[index])[:cols-2])
                        else:
                            stdscr.addstr(1 + index, 0, str(result[index])[:cols])
                        stdscr.clrtoeol()
                    for index in range(len(result), search_k):
                        stdscr.addstr(1 + index, 0, "")
                        stdscr.clrtoeol()

            return curses.wrapper(c_main)

    def fuzzy_search(self, question="", k=NoDefault, str_func=str):
        if len(question) > 0:
            ratio = self.map(str_func).map(lambda x: fuzz.partial_ratio(x.lower(), question.lower()) * min(1, len(x) / len(question)) * min(1, len(question) / len(x)) ** 0.3).map(lambda x: round(x * 10) / 10).sort(lambda x: -x)
            if k is not NoDefault:
                return self.map_index(ratio.sort().index_mapping)[:k]
            else:
                return self.map_index(ratio.sort().index_mapping)[0]
        else:
            def c_main(stdscr: "curses._CursesWindow"):
                stdscr.clear()
                question = ""
                question_done = False
                select_number = 0
                result = vector()
                rows, cols = stdscr.getmaxyx()

                stdscr.addstr(0, 0, "token to search: ")
                search_k = k
                if search_k is NoDefault:
                    search_k = rows - 1
                for index in range(len(self[:search_k])):
                    if index == 0:
                        stdscr.addstr(index + 1, 0, "* " + str(self[index])[:100])
                    else:
                        stdscr.addstr(index + 1, 0, str(self[index])[:100])

                while True:
                    stdscr.addstr(0, 0, "token to search: ")
                    stdscr.clrtoeol()
                    stdscr.addstr(question)

                    char = stdscr.get_wch()
                    if isinstance(char, str) and char.isprintable():
                        question += char
                        select_number = 0
                    elif char == curses.KEY_BACKSPACE or char == "\x7f":
                        question = question[:-1]
                        select_number = 0
                    elif char == "\n":
                        if len(result) > 0:
                            return result[select_number]
                    elif char == "\x1b":
                        return None
                    elif char == curses.KEY_UP:
                        select_number = max(select_number - 1, 0)
                    elif char == curses.KEY_DOWN:
                        select_number = max(min(select_number + 1, len(result) - 1), 0)
                    elif char == curses.KEY_LEFT or char == curses.KEY_RIGHT:
                        continue
                    else:
                        raise AssertionError(repr(char))

                    if len(question) > 0:
                        ratio = self.map(str_func).map(lambda x: fuzz.partial_ratio(x.lower(), question.lower()) * min(1, len(x) / len(question)) * min(1, len(question) / len(x)) ** 0.3).map(lambda x: round(x * 10) / 10).sort(lambda x: -x)
                        result = self.map_index(ratio.index_mapping)[:search_k]
                        remain_len = ratio.filter(lambda x: x > 5).length
                        result = result[:remain_len]
                    else:
                        result = self[:search_k]
                        ratio = result.map(lambda x: 0)
                    for index in range(len(result)):
                        if index == select_number:
                            stdscr.addstr(1 + index, 0, "* " + str(result[index])[:100] + " " + str(ratio[index]))
                        else:
                            stdscr.addstr(1 + index, 0, str(result[index])[:100] + " " + str(ratio[index]))
                        stdscr.clrtoeol()
                    for index in range(len(result), search_k):
                        stdscr.addstr(1 + index, 0, "")
                        stdscr.clrtoeol()

            return curses.wrapper(c_main)

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
    def map(self, func, *args, default=NoDefault) -> "ctgenerator":
        if func is None:
            for x in self.generator:
                yield x
        if len(args) > 0:
            func = chain_function((func, *args))
        if default is not NoDefault:
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
