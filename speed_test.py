import numba as nb
import numpy as np
import sys
import os
sys.path.append(os.path.abspath("."))
from pyctlib import vector, IndexMapping, scope, vhelp

@nb.jit(nb.f8[:](nb.f8[:]), nopython=True, cache=True)
def numba_cumsum1(x):
    return np.cumsum(x)

@nb.jit(nb.f8[:](nb.f8[:]), nopython=True, cache=True)
def numba_cumsum2(x):
    ret = np.zeros(x.shape[0])
    ret[0] = x[0]
    for index in range(1, x.shape[0]):
        ret[index] = ret[index-1] + x[index]
    return ret

def numpy_cumsum2(x):
    return np.cumsum(x)

def cumsum(x):
    ret = [0] * len(x)
    ret[0] = x[0]
    for index in range(1, len(x)):
        ret[index] = ret[index - 1] + x[index]
    return ret

t = vector.randn(100000)
t_list = list(t)
t_np = np.array(t)

%timeit t.cumsum()
%timeit t_np.cumsum()
%timeit cumsum(t_list)

%timeit t.to_numpy()
%timeit vector(t_np)
%timeit t_np.tolist()

