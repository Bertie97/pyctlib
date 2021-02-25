#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package <main>
##############################
__all__ = """
    vector
    generator_wrapper
    ctgenerator
""".split()

from types import GeneratorType
from collections import Counter
from pyoverload import *
from functools import wraps

"""
Usage:
from pyctlib.vector import *
"""

def touch(f: Callable):
    try: return f()
    except: return None

def raw_function(func):
    if "__func__" in dir(func):
        return func.__func__
    return func

class _Vector_Dict(dict):

    def values(self):
        return vector(super().values())

    def keys(self):
        return vector(super().keys())

class vector(list):

    def __init__(self, *args):
        if len(args) == 0:
            list.__init__(self)
        elif len(args) == 1:
            list.__init__(self, args[0])
        else:
            list.__init__(self, args)

    def filter(self, func=None):
        if func is None:
            return self
        try:
            return vector([a for a in self if func(a)])
        except:
            pass
        for index, a in enumerate(self):
            if touch(lambda: func(a)) is None:
                raise RuntimeError("Exception raised in filter function at location {} for element {}".format(index, a))

    def test(self, func):
        return vector([a for a in self if touch(lambda: func(a))])

    def testnot(self, func):
        return vector([a for a in self if not touch(lambda: func(a))])

    def map(self, func=None):
        """
        generate a new vector with each element x are replaced with func(x)
        """
        if func is None:
            return self
        try:
            return vector([func(a) for a in self])
        except:
            pass
        for index, a in enumerate(self):
            if touch(lambda: func(a)) is None:
                raise RuntimeError("Exception raised in map function at location {} for element {}".format(index, a))

    def apply(self, func) -> None:
        for x in self:
            func(x)

    def check_type(self, instance):
        return all(self.map(lambda x: isinstance(x, instance)))

    @override
    def __mul__(self, other):
        if touch(lambda: self.check_type(tuple) and other.check_type(tuple)):
            return vector(zip(self, other)).map(lambda x: (*x[0], *x[1]))
        elif touch(lambda: self.check_type(tuple)):
            return vector(zip(self, other)).map(lambda x: (*x[0], x[1]))
        elif touch(lambda: other.check_type(tuple)):
            return vector(zip(self, other)).map(lambda x: (x[0], *x[1]))
        else:
            return vector(zip(self, other))

    @__mul__
    def _(self, times: int):
        return vector(super().__mul__(times))

    def __pow__(self, other):
        return vector([(i, j) for i in self for j in other])

    def __add__(self, other: list):
        return vector(super().__add__(other))

    def _transform(self, element, func=None):
        if not func:
            return element
        return func(element)

    @override
    def __eq__(self, element):
        return self.map(lambda x: x == element)

    @__eq__
    def _(self, other: list):
        return vector(zip(self, other)).map(lambda x: x[0] == x[1])

    @override
    def __neq__(self, element):
        return self.map(lambda x: x != element)

    @__neq__
    def _(self, other: list):
        return vector(zip(self, other)).map(lambda x: x[0] != x[1])

    @override
    def __lt__(self, element):
        return self.map(lambda x: x < element)

    @__lt__
    def _(self, other: list):
        return vector(zip(self, other)).map(lambda x: x[0] < x[1])

    @override
    def __gt__(self, element):
        return self.map(lambda x: x > element)

    @__gt__
    def _(self, other: list):
        return vector(zip(self, other)).map(lambda x: x[0] > x[1])

    @override
    def __le__(self, element):
        return self.map(lambda x: x < element)

    @__le__
    def _(self, other: list):
        return vector(zip(self, other)).map(lambda x: x[0] <= x[1])

    @override
    def __ge__(self, element):
        return self.map(lambda x: x >= element)

    @__ge__
    def _(self, other: list):
        return vector(zip(self, other)).map(lambda x: x[0] >= x[1])

    @override
    def __getitem__(self, index):
        if isinstance(index, slice):
            return vector(super().__getitem__(index))
        return super().__getitem__(index)

    @__getitem__
    def _(self, index_list: list):
        assert len(self) == len(index_list)
        return vector(zip(self, index_list)).filter(lambda x: x[1]).map(lambda x: x[0])

    @overload
    def __sub__(self, other: Iterable):
        try:
            other = set(other)
        except:
            other = list(other)
        finally:
            return self.filter(lambda x: x not in other)

    @overload
    def __sub__(self, other):
        return self.filter(lambda x: x != other)

    def __setitem__(self, i, t):
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

    def count(self, *args):
        if len(args) == 0:
            return len(self)
        return super().count(args[0])

    # @overload
    # def index(self, element: int):
    #     return super().index(element)

    # @overload
    def index(self, element):
        if isinstance(element, int):
            return super().index(element)
        elif callable(element):
            for index in range(len(self)):
                if element(self[index]):
                    return index
            return -1
        else:
            raise RuntimeError("error input for index")

    def all(self, func=lambda x: x):
        for t in self:
            if not func(t):
                return False
        return True

    def any(self, func=lambda x: x):
        for t in self:
            if func(t):
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

    def sum(self):
        return self.reduce(lambda x, y: x + y)

    def group_by(self, key=lambda x: x[0]):
        result = _Vector_Dict()
        for x in self:
            k_x = key(x)
            if k_x not in result:
                result[k_x] = vector([x])
            else:
                result[k_x].append(x)
        return result

    def reduce(self, func):
        if len(self) == 0:
            return None
        temp = self[0]
        for x in self[1:]:
            temp = func(temp, x)
        return temp

    def flatten(self):
        return self.reduce(lambda x, y: x + y)

    def generator(self):
        return ctgenerator(self)

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
