# pyctlib
A powerful toolkit for python development.

## BasicType

### vector

`vector` is an improved implement of `list` for python. In addition to original function for `list`, it has the following advanced usage:

```
In [1]: from pyctlib.basictype import vector                                             

In [2]: a = vector([1,2,3,4,5,6,7,8])                                                    

In [3]: a                                                                                
Out[3]: [1, 2, 3, 4, 5, 6, 7, 8]

In [4]: a.map(lambda x: x + 1)                                                           
Out[4]: [2, 3, 4, 5, 6, 7, 8, 9]

In [5]: a.filter(lambda x: x % 2 == 0)                                                   
Out[5]: [2, 4, 6, 8]

In [6]: a[2 < a < 5]                                                                     
Out[6]: [1, 2, 3, 4]
```
