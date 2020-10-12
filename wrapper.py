#! python3 -u
#  -*- coding: utf-8 -*-

##############################
## Author: Yuncheng Zhou
##############################

from functools import wraps

def raw_function(func):
    if "__func__" in func.__dir__():
        return func.__func__
    return func

def decorator(wrapper_func):
    if not callable(wrapper_func): raise TypeError("function is not callable")
    def wrapper(func):
        func = raw_function(func)
        if not callable(func): raise TypeError("function is not callable")
        return wraps(func)(wrapper_func(func))
    return wrapper

# def value(func): return func()