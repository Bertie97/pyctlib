# pyctlib
A package provides easy communication with the inputs and outputs for the operating systems. It provides manipulation of file path, reading & writing of text/binary files and command line tools. A powerful toolkit for python development.

## BasicType

### vector

`vector` is an improved implement of `list` for python. In addition to original function for `list`, it has the following advanced usage:

```
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
