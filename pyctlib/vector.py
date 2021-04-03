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

"""
Usage:
from pyctlib.vector import *
from pyctlib import touch
"""

def totuple(x, depth=1):
    if not isinstance(x, tuple):
        x = (x, )
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

class vector(list):
    """vector
    vector is actually list in python with advanced method like map, filter and reduce
    """


    def __init__(self, *args, recursive=False):
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
                return vector([a for a in self if touch(lambda: func(a))], recursive=self._recursive)
            return vector([a for a in self if func(a)], recursive=self._recursive)
        except:
            pass
        for index, a in enumerate(self):
            if touch(lambda: func(a)) is None:
                try:
                    error_information = "Exception raised in filter function at location {} for element {}".format(index, a)
                except:
                    error_information = "Exception raised in filter function at location {} for element {}".format(index, "<unknown>")
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
        return vector([a for a in self if touch(lambda: (func(a), True)[-1])], recursive=self._recursive)

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
        return vector([a for a in self if not touch(lambda: (func(a), True)[-1])], recursive=self._recursive)

    def map(self, func, *args, default=NoDefault):
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
            return vector([touch(lambda: func(a), default=default) for a in self], recursive=self._recursive)
        try:
            return vector([func(a) for a in self], recursive=self._recursive)
        except:
            pass
        for index, a in enumerate(self):
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
            return vector(super().__getitem__(index))
        if isinstance(index, list):
            assert len(self) == len(index)
            return vector(zip(self, index)).filter(lambda x: x[1]).map(lambda x: x[0])
        if isinstance(index, tuple):
            return super().__getitem__(index[0])[index[1:]]
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

    def unique(self):
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
        afflicated_vector = vector(key(index) for index in range(self.length))
        temp = sorted(zip(self, afflicated_vector), key=lambda x: x[1])
        return vector(temp).map(lambda x: x[0])

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

    def append(self, *args):
        """append.

        Parameters
        ----------
        args :
            args
        """
        self.clear_appendix()
        super().append(*args)
        return self

    def extend(self, *args):
        """extend.

        Parameters
        ----------
        args :
            args
        """
        self.clear_appendix()
        super().extend(*args)
        return self

    def pop(self, *args):
        """pop.

        Parameters
        ----------
        args :
            args
        """
        self.clear_appendix()
        return super().pop(*args)

    def insert(self, *args):
        """insert.

        Parameters
        ----------
        args :
            args
        """
        self.clear_appendix()
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

    def sample(self, *args, replace=True, p=None):
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
        return vector(np.random.choice(self, size=args, replace=replace, p=p), recursive=False)

    def shuffle(self):
        return self.sample(self.length)

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
