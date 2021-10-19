#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package <main>
##############################
__all__ = """
    restore_type_wrapper
    generate_typehint_wrapper
    empty_wrapper
""".split()

from pyoverload import *
from .basicwrapper import *
import inspect
from functools import wraps
from .strtools import delete_surround

def _restore_type_wrapper(func: Callable, special_attr: List[str]):
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        if len(args) == 0: return ret
        if str(type(args[0])) in func.__qualname__ and len(args) > 1: totype = type(args[1])
        else: totype = type(args[0])
        constructor = totype
        if "numpy.ndarray" in str(totype):
            import numpy as np
            constructor = np.array
        elif "torchplus" in str(totype):
            import torchplus as tp
            constructor = tp.tensor
        elif "torch.Tensor" in str(totype):
            import torch
            constructor = lambda x: x.as_subclass(torch.Tensor) if isinstance(x, torch.Tensor) else torch.tensor(x)
        if not isinstance(ret, tuple): ret = (ret,)
        output = tuple()
        for r in ret:
            try: new_r = constructor(r)
            except: new_r = r
            for a in special_attr:
                if a in dir(r): exec(f"new_r.{a} = r.{a}")
            output += (new_r,)
        if len(output) == 1: output = output[0]
        return output
    return wrapper

@overload
@decorator
def restore_type_wrapper(func: Callable):
    return _restore_type_wrapper(func, [])

@overload
def restore_type_wrapper(*special_attr: str):
    @decorator
    def restore_type_decorator(func: Callable):
        return _restore_type_wrapper(func, special_attr)
    return restore_type_decorator

def type_str(obj):
    if isinstance(obj, list):
        if len(obj) > 0 and all(isinstance(t, type(obj[0])) for t in obj):
            return delete_surround(str(type(obj)), "<class '", "'>") + "[{}]".format(type_str(obj[0]))
    return delete_surround(str(type(obj)), "<class '", "'>")

def generate_typehint_wrapper(func):
    assert callable(raw_function(func))
    func = raw_function(func)
    @wraps(func)
    def wrapper(*args, **kwargs):
        args_name = inspect.getfullargspec(func)[0]
        ret = func(*args, **kwargs)
        from .vector import vector
        typehint = vector()
        default_dict = dict()
        default = inspect.getfullargspec(func).defaults
        if default:
            for index in range(len(default)):
                name = args_name[len(args_name) - len(default) + index]
                default_dict[name] = default[index]
        for name, arg in zip(args_name, args):
            typehint.append("@type {}: {}".format(name, type_str(arg)))
        for name in args_name[len(args):]:
            if name in kwargs:
                typehint.append("@type {}: {}".format(name, type_str(kwargs[name])))
            else:
                typehint.append("@type {}: {}".format(name, type_str(default_dict[name])))
        print("\n".join(typehint))
        return ret
    return wrapper

def empty_wrapper(*args, **kwargs):
    if len(kwargs) > 0 or len(args) > 1 or (len(args) == 1 and not callable(args[0])):
        return empty_wrapper
    elif len(args) == 1:
        func = args[0]
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    else:
        return
