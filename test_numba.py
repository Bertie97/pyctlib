import numba
import numpy as np
import time

@numba.njit
def dummy():
    return None

@numba.njit
def sum2d_nocache(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return result

@numba.njit(cache=True)
def sum2d_cache(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return result

start = time.time()
dummy()
end = time.time()
print(f'Dummy timing {end - start}')

a=np.random.random((1000,100))
start = time.time()
sum2d_nocache(a)
end = time.time()
print(f'No cache 1st timing {end - start}')

a=np.random.random((1000,100))
start = time.time()
sum2d_nocache(a)
end = time.time()
print(f'No cache 2nd timing {end - start}')

a=np.random.random((1000,100))
start = time.time()
sum2d_cache(a)
end = time.time()
print(f'Cache 1st timing {end - start}')

a=np.random.random((1000,100))
start = time.time()
sum2d_cache(a)
end = time.time()
print(f'Cache 2nd timing {end - start}')

@numba.jit(nopython=False)
def test_function(func, a):
    length = a.shape[0]
    for index in range(length):
        a[index] = func(a[index])
    return a
