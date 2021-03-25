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
from collections import Counter
from pyoverload import *
from functools import wraps, reduce
from .touch import touch, crash
import copy
import numpy as np

"""
Usage:
from pyctlib.vector import *
"""

def totuple(x, depth=1):
    if not iterable(x):
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

class vector(list):

    def __init__(self, *args):
        if len(args) == 0:
            list.__init__(self)
        elif len(args) == 1:
            if args[0] is None:
                return vector()
            elif isinstance(args[0], np.ndarray):
                temp = vector.from_numpy(args[0])
                list.__init__(self, temp)
            elif isinstance(args[0], list):
                def to_vector(array):
                    if isinstance(array, list):
                        return [vector.from_list(x) for x in array]
                temp = to_vector(args[0])
                list.__init__(self, temp)
            else:
                try:
                    list.__init__(self, args[0])
                except:
                    list.__init__(self, args)
        else:
            list.__init__(self, args)

    def filter(self, func=None, ignore_error=True):
        if func is None:
            return self
        try:
            if ignore_error:
                return vector([a for a in self if touch(lambda: func(a))])
            return vector([a for a in self if func(a)])
        except:
            pass
        for index, a in enumerate(self):
            if touch(lambda: func(a)) is None:
                try:
                    error_information = "Exception raised in filter function at location {} for element {}".format(index, a)
                except:
                    error_information = "Exception raised in filter function at location {} for element {}".format(index, "<unknown>")
                raise RuntimeError(error_information)

    def test(self, func):
        return vector([a for a in self if touch(lambda: func(a))])

    def testnot(self, func):
        return vector([a for a in self if not touch(lambda: func(a))])

    def map(self, func=None, default=NoDefault):
        """
        generate a new vector with each element x are replaced with func(x)
        """
        if func is None:
            return self
        if default is not NoDefault:
            return vector([touch(lambda: func(a), default=default) for a in self])
        try:
            return vector([func(a) for a in self])
        except:
            pass
        for index, a in enumerate(self):
            if touch(lambda: func(a)) is None:
                try:
                    error_information = "Exception raised in map function at location [{}] for element [{}] with function [{}] and default value [{}]".format(index, a, func, default)
                except:
                    error_information = "Exception raised in map function at location [{}] for element [{}] with function [{}] and default value [{}]".format(index, "<unknown>", func, default)
                raise RuntimeError(error_information)

    def rmap(self, func=None, default=NoDefault):
        if func is None:
            return self
        return self.map(lambda x: x.rmap(func, default) if isinstance(x, vector) else func(x), default)

    def replace(self, element, toelement=NoDefault):
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
        if isinstance(command, str):
            for x in self:
                exec(command.format(x))
        else:
            for x in self:
                command(x)

    def check_type(self, instance):
        return all(self.map(lambda x: isinstance(x, instance)))

    def __mul__(self, other):
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

    def __pow__(self, other):
        return vector([(i, j) for i in self for j in other])

    def __add__(self, other: list):
        return vector(super().__add__(other))

    def _transform(self, element, func=None):
        if not func:
            return element
        return func(element)

    def __eq__(self, other):
        if isinstance(other, list):
            return vector(zip(self, other)).map(lambda x: x[0] == x[1])
        else:
            return self.map(lambda x: x == other)

    def __neq__(self, other):
        if isinstance(self, list):
            return vector(zip(self, other)).map(lambda x: x[0] != x[1])
        else:
            return self.map(lambda x: x != other)

    def __lt__(self, element):
        if isinstance(element, list):
            return vector(zip(self, element)).map(lambda x: x[0] < x[1])
        else:
            return self.map(lambda x: x < element)

    def __gt__(self, element):
        if isin(element, list):
            return vector(zip(self, element)).map(lambda x: x[0] > x[1])
        else:
            return self.map(lambda x: x > element)

    def __le__(self, element):
        if isin(element, list):
            return vector(zip(self, element)).map(lambda x: x[0] <= x[1])
        else:
            return self.map(lambda x: x < element)

    def __ge__(self, element):
        if isin(element, list):
            return vector(zip(self, element)).map(lambda x: x[0] >= x[1])
        else:
            return self.map(lambda x: x >= element)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return vector(super().__getitem__(index))
        if isinstance(index, list):
            assert len(self) == len(index)
            return vector(zip(self, index)).filter(lambda x: x[1]).map(lambda x: x[0])
        if isinstance(index, tuple):
            return super().__getitem__(index[0])[index[1:]]
        return super().__getitem__(index)

    def __sub__(self, other):
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
        self._shape = None
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


    def _hashable(self):
        return all(self.filter(lambda x: "__hash__" in x.__dir__()))

    def __hash__(self):
        if not self._hashable():
            raise Exception("not all elements in the vector is hashable, the index of first unhashable element is %d" % self.index(lambda x: "__hash__" not in x.__dir__()))
        else:
            return hash(tuple(self))

    def unique(self):
        if len(self) == 0:
            return vector([])
        hashable = self._hashable()
        explored = set() if hashable else list()
        pushfunc = explored.add if hashable else explored.append
        unique_elements = list()
        for x in self:
            if x not in explored:
                unique_elements.append(x)
                pushfunc(x)
        return vector(unique_elements)

    def count_all(self):
        if len(self) == 0:
            return vector([])
        hashable = self._hashable()
        if hashable:
            return Counter(self)
        else:
            return self.unique().map(lambda x: (x, self.count(x)))

    def count(self, *args):
        if len(args) == 0:
            return len(self)
        if callable(args[0]):
            return len(self.filter(args[0]))
        return super().count(args[0])

    def index(self, element):
        if callable(element):
            for index in range(len(self)):
                if touch(lambda: element(self[index])):
                    return index
            return -1
        else:
            return super().index(element)

    def findall(self, element):
        if callable(element):
            return vector([index for index in range(len(self)) if touch(lambda: element(self[index]))])
        else:
            return vector([index for index in range(len(self)) if self[index] == element])

    def findall_crash(self, func):
        assert callable(func)
        return vector([index for index in range(len(self)) if crash(lambda: func(self[index]))])

    def all(self, func=lambda x: x):
        for t in self:
            if not touch(lambda: func(t)):
                return False
        return True

    def any(self, func=lambda x: x):
        for t in self:
            if touch(lambda: func(t)):
                return True
        return False

    def max(self, key=None, with_index=False):
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
        return self.reduce(lambda x, y: x + y, default)

    def prod(self, default=None):
        return self.reduce(lambda x, y: x * y, default)

    def group_by(self, key=lambda x: x[0]):
        result = _Vector_Dict()
        for x in self:
            k_x = key(x)
            if k_x not in result:
                result[k_x] = vector([x])
            else:
                result[k_x].append(x)
        return result

    def reduce(self, func, default=None):
        if len(self) == 0:
            return default
        temp = self[0]
        for x in self[1:]:
            temp = func(temp, x)
        return temp

    def enumerate(self):
        return enumerate(self)

    def flatten(self, depth=-1):
        def temp_flatten(array, depth=-1):
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
            if len(target_shape) == 1:
                return value
            piece_length = len(value) // target_shape[0]
            ret = vector(_reshape(value[piece_length * index: piece_length * (index + 1)], target_shape[1:]) for index in range(target_shape[0]))
            return ret
        return _reshape(self.flatten(), args)

    def generator(self):
        return ctgenerator(self)

    @property
    def length(self):
        return len(self)

    def onehot(self, max_length=-1, default_dict={}):
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
            ret = np.zeros(length)
            ret[index] = 1.
            return ret
        return temp_list.map(lambda x: create_onehot_vector(x, max_length))

    def sort_by_index(self, key=lambda index: index):
        afflicated_vector = vector(key(index) for index in range(self.length))
        temp = sorted(zip(self, afflicated_vector), key=lambda x: x[1])
        return vector(temp).map(lambda x: x[0])

    def sort_by_vector(self, other, func=lambda x: x):
        assert isinstance(other, list)
        assert self.length == len(other)
        return self.sort_by_index(lambda index: func(other[index]))

    @staticmethod
    def from_numpy(array):
        try:
            assert isinstance(array, np.ndarray)
            if len(array.shape) == 1:
                return vector(list(array))
            else:
                return vector(list(array)).map(lambda x: vector.from_numpy(x))
        except Exception as e:
            print("warning: input isn't pure np.ndarray")
            return vector(list(array))

    @staticmethod
    def from_list(array):
        if not isinstance(array, list):
            return array
        return vector(vector.from_list(x) for x in array)

    @staticmethod
    def zeros(*args):
        args = totuple(args)
        return vector.from_numpy(np.zeros(args))

    @staticmethod
    def ones(*args):
        args = totuple(args)
        ret = vector.from_numpy(np.ones(args))
        ret._shape = args
        return ret

    @staticmethod
    def rand(*args):
        args = totuple(args)
        ret = vector.from_numpy(np.random.rand(*args))
        ret._shape = args
        return ret

    @staticmethod
    def randn(*args):
        args = totuple(args)
        ret = vector.from_numpy(np.random.randn(*args))
        ret._shape = args
        return ret

    @staticmethod
    def range(*args):
        return vector(range(*args))

    @property
    def shape(self):
        if touch(lambda: self._shape) is not None:
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
        self._shape = None
        super().append(*args)
        return self

    def extend(self, *args):
        self._shape = None
        super().extend(*args)
        return self

    def pop(self, *args):
        self._shape = None
        return super().pop(*args)

    def insert(self, *args):
        self._shape = None
        super().insert(*args)
        return self

    def clear(self):
        self._shape = None
        super().clear()
        return self

    def remove(self, *args):
        self._shape = None
        super().remove(*args)
        return self

    def all_equal(self):
        if self.length <= 1:
            return True
        return self.all(lambda x: x == self[0])

    def sample(self, *args, replace=True, p=None):
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
        return vector(np.random.choice(self, size=args, replace=replace, p=p))

def generator_wrapper(*args, **kwargs):
    if len(args) == 1 and callable(raw_function(args[0])):
        func = raw_function(args[0])
        @wraps(func)
        def wrapper(*args, **kwargs):
            return ctgenerator(func(*args, **kwargs))
        return wrapper
    else:
        raise TypeError("function is not callable")

class ctgenerator:

    @staticmethod
    def _generate(iterable):
        for x in iterable:
            yield x

    @override
    def __init__(self, generator):
        if "__iter__" in generator.__dir__():
            self.generator = ctgenerator._generate(generator)

    @__init__
    def _(self, generator: "ctgenerator"):
        self.generator = generator

    @__init__
    def _(self, generator: GeneratorType):
        self.generator = generator

    @generator_wrapper
    def map(self, func) -> "ctgenerator":
        for x in self.generator:
            yield func(x)

    @generator_wrapper
    def filter(self, func=None) -> "ctgenerator":
        for x in self.generator:
            if func(x):
                yield x

    def reduce(self, func, initial_value=None):
        if not initial_value:
            initial_value = next(self.generator)
        result = initial_value
        for x in self.generator:
            result = func(initial_value, x)
        return result

    def apply(self, func) -> None:
        for x in self.generator:
            func(x)

    def __iter__(self):
        for x in self.generator:
            yield x

    def __next__(self):
        return next(self.generator)

    def vector(self):
        return vector(self)
