import sys
import os
from sys import getsizeof

sys.path.append(os.path.abspath("."))
import pyctlib
import pathlib
import numpy as np
from pyctlib import vector, IndexMapping, scope, vhelp
from pyctlib.vector import chain_function
from fuzzywuzzy import fuzz
with scope("import"):
    from pyctlib import path, get_relative_path, file
    from pyctlib import touch
    from pyctlib.wrapper import generate_typehint_wrapper

print(pyctlib.__file__)

# from pyctlib import Logger
# logger = Logger(True)
# # logger.test_info(1)
# try:
#     1 / 0
# except Exception as e:
#     logger.exception(e)
#     exit(0)

with scope("all"):
    vec = vector(1, 2, 3)
    assert list(vec) == [1, 2, 3]
    vec = vector([1, 2, 3])
    assert list(vec) == [1, 2, 3]
    vec = vector((1, 2, 3))
    assert list(vec) == [1, 2, 3]
    assert list(vector([1, 2, 3, 4, 5, 6]).filter(lambda x: x > 3)) == [4, 5, 6]
    assert list(vector(0, 1, 2, 3).test(lambda x: 1 / x)) == [1, 2, 3]
    assert list(vector(0, 1, 2, 3).testnot(lambda x: 1 / x)) == [0]
    assert list(vector([0, 1, 2]).map(lambda x: x ** 2)) == [0, 1, 4]
    assert list(vector([[0, 1], [2, 3]], recursive=True).rmap(lambda x: x + 1).flatten()) == [1, 2, 3, 4]
    assert list(vector(0, 1, 2, 3, 1).replace(1, -1)) == [0, -1, 2, 3, -1]
    assert list(vector(0, 1, 2, 3, 4).replace(lambda x: x > 2, 2)) == [0, 1, 2, 2, 2]
    assert list(vector(0, 1, 2, 3, 4).replace(lambda x: x > 2, lambda x: x + 2)) == [0, 1, 2, 5, 6]
    assert list(vector([1, 2, 3]) * vector([4, 5, 6])) == [(1, 4), (2, 5), (3, 6)]
    assert list(vector([1, 2, 3]) ** vector([2, 3, 4])) == [(1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4)]
    assert list(vector([1, 2, 3, 2, 3, 1]).unique()) == [1, 2, 3]
    assert list(vector(0, 1, 2, 3, 4, 0, 1).findall_crash(lambda x: 1 / x)) == [0, 5]
    assert list(vector([1, 2, 3, 4, 2, 3]).findall(lambda x: x > 2)) == [2, 3, 5]
    assert list(vector.zip(vector(1, 2, 3), vector(1, 2, 3), vector(1, 2, 3))) == [(1, 1, 1), (2, 2, 2), (3, 3, 3)]
    assert list(vector([1, 2, 3, 4, 1]).sort_by_index(key=lambda x: -x)) == [1, 4, 3, 2, 1]
    x = vector.range(100).shuffle()
    assert x == vector.range(100).map_index_from(x)
    x = vector.range(100).sample(10, replace=False)
    assert x == vector.range(100).map_index_from(x)
    assert repr(vector.range(10).filter(lambda x: x < 5).unmap_index()) == "[0, 1, 2, 3, 4, Not Defined, Not Defined, Not Defined, Not Defined, Not Defined]"
    vector.range(5).map(lambda x, y: x / y, func_self=lambda x: x.sum())
    assert vector.range(100).reshape(2, -1).shape == (2, 50)
    x = vector.range(100)[:10]
    y = vector.range(100).map_index_from(x)
    assert x == y
    assert list(vector.range(100)[::-1]) == list(range(100)[::-1])
    x = vector.rand(10)
    assert all(x[::-2][::-1] == x[1::2])

with scope("IndexMapping"):
    t1 = IndexMapping(slice(0, 15, 2), 10, True)
    t2 = IndexMapping([4,3,2,1,0], 5, True)
    assert list(t1.map(t2).index_map) == [4, -1, 3, -1, 2, -1, 1, -1, 0, -1]
    t3 = IndexMapping(slice(0, 2, 1), range_size=5, reverse=True)
    assert list(t1.map(t3).index_map) == [0, -1, 1, -1, -1, -1, -1, -1, -1, -1]
