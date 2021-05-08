import numba as nb
import numpy as np

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
