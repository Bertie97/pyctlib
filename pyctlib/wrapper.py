#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package <main>
##############################
__all__ = """
    restore_type_wrapper
""".split()

from pyoverload import *
from .basicwrapper import *

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
