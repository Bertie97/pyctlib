# pyctlib
## Introduction

Package [`pyctlib`](https://github.com/Bertie97/pyctlib/tree/main/pyctlib) is the fundamental package for project [`PyCTLib`](https://github.com/Bertie97/pyctlib), a powerful toolkit for python development. This package provides basic functions for fast `python v3.6+` programming. It provides easy communication with the inputs and outputs for the operating systems. It provides quick try-catch function `touch`, useful types like `vector`, timing tools like `timethis` and `scope` as long as manipulation of file path, reading & writing of text/binary files and command line tools. 

## Installation

This package can be installed by `pip install pyctlib` or moving the source code to the directory of python libraries (the source code can be downloaded on [github](https://github.com/Bertie97/pyctlib) or [PyPI](https://pypi.org/project/pyctlib/)). 

```shell
pip install pyctlib
```

## Basic Types

### vector

`vector` is an improved implement of `list` for python. In addition to original function for `list`, it has the following advanced usage:

```python
In [1]: from pyctlib.basictype import vector                                         

In [2]: a = vector([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])                                  

In [3]: a.map(lambda x: x * 2)                                                       
Out[3]: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

In [4]: a.filter(lambda x: x % 2 == 0)                                               
Out[4]: [2, 4, 6, 8, 10]

In [5]: a.all(lambda x: x > 1)                                                       
Out[5]: False

In [6]: a.any(lambda x: x == 2)                                                      
Out[6]: True

In [7]: a[a > 7] = 7                                                                 

In [8]: a                                                                            
Out[8]: [1, 2, 3, 4, 5, 6, 7, 7, 7, 7]

In [9]: a.reduce(lambda x, y: x + y)                                                 
Out[9]: 49

In [10]: a.index(lambda x: x % 4 == 0)                                               
Out[10]: 3
```

## File Management

## Timing Tools

### `scope`

```python
>>> from pyctlib import scope, jump
>>> with scope("name1"):
...     print("timed1")
...
timed1
[name1 takes 0.000048s]
>>> with scope("name2"), jump:
...     print("timed2")
...
>>>
```

### `timethis`

```python
>>> from pyctlib import timethis
>>> @timethis
... def func(): print('timed')
... 
>>> func()
timed
[func takes 0.000050s]
```

## Touch function

```python
>>> from pyctlib import touch
>>> touch(lambda: 1/0, 'error')
'error'
>>> touch('a')
>>> a = 4
>>> touch('a')
4
```



## Acknowledgment

@Yiteng Zhang, Yuncheng Zhou: Developers