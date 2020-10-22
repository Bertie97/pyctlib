# pyoverload

`pyoverload` is a package affiliated to project [`pyctlib`](https://github.com/Bertie97/pyctlib). It is a powerful overloading tools to provide easy overload for `python v3.6+`. `pyoverload` provide multiple usages. The simplest one, however, can be easily implemented as follows. 

```python
>>> from pyoverload import overload
>>> @overload
... def func(x: int):
...     print("func1", x)
...
>>> @overload
... def func(x: str):
...     print("func2", x)
...
>>> func(1)
func1 1
>>> func("1")
func2 1
```

`pyoverload` has all of following appealing features:

1. Support of **`Jedi` auto-completion** by keyword decorator "overload". This means all main-stream python IDE can hint you the overloaded functions you have defined. 
2. **Multiple usages** that are user friendly for all kinds of users, including `C/Java` language system users and those who are used to `singledispatch` based overload. Also, easy collector of ordinary python functions is also provided. 
3. Support of **all kinds of functions**, including functions, methods, class methods and static methods. One simple implementation for all.
4. **String types** supported. This means that one can use `"numpy.ndarray"` to mark a numpy array without importing the whole package. 
5. Sufficient **built-in types are provided** for easy representations such as `List[Int]`, `Dict@{str: int}` or `List<<int>>[10]`. 
6. **Available usage listing** when no overload function matches the input arguments. 
7. **Type constraint for an ordinary function** using `@params` decorator. 

