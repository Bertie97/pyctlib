#! python3 -u
#  -*- coding: utf-8 -*-

##############################
## Author: Yuncheng Zhou
##############################

from functools import wraps

def raw_function(func):
    if "__func__" in dir(func):
        return func.__func__
    return func

def decorator(wrapper_func):
    if not callable(wrapper_func): raise TypeError("function is not callable")
    def wrapper(func):
        func = raw_function(func)
        func_name = f"{func.__name__}[{wrapper_func.__qualname__.split('.')[0]}]"
        if not callable(func): raise TypeError("function is not callable")
        wrapped_func = wraps(func)(wrapper_func(func))
        wrapped_func.__name__ = func_name
        return wrapped_func
    wrapper.__name__ = wrapper(lambda: 0).__name__
    return wrapper

# def value(func): return func()