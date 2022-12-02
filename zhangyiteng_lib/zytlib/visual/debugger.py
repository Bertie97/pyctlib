#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package visual
##############################

__all__ = """
    profile
""".split()

import sys
from functools import wraps
try:
    from line_profiler import LineProfiler
except ImportError:
    raise ImportError("'pyctlib.watch.debugger' cannot be used without dependency 'line_profiler'. ")

def profile(*args, **kwargs):
    """
    Usage 1:
    @profile
    def function()

    Usage 2:
    @profile(2)
    def function()

    Usage 3:
    @profile(max_cnt, filename=<path>)
    def function
    """
    if len(args) == 1 and callable(args[0]):
        func = args[0]

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not hasattr(wrapper, 'cnt'):
                wrapper.cnt = 1
            else:
                wrapper.cnt += 1
            if not sys._getframe().f_back.f_code.co_name == func.__name__:
                prof = LineProfiler()
                try:
                    return prof(func)(*args, **kwargs)
                finally:
                    prof.print_stats()
            else:
                return func(*args, **kwargs)

        return wrapper
    elif len(kwargs) > 0 and "filename" in kwargs:
        if len(args) > 0:
            max_cnt = args[0]
        else:
            max_cnt = -1
        filename = kwargs["filename"]

        def profile_decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not hasattr(wrapper, 'cnt'):
                    wrapper.cnt = 1
                else:
                    wrapper.cnt += 1
                if not (max_cnt >= 0 and wrapper.cnt > max_cnt) and not sys._getframe().f_back.f_code.co_name == func.__name__:
                    prof = LineProfiler()
                    try:
                        return prof(func)(*args, **kwargs)
                    finally:
                        prof.dump_stats(filename)
                else:
                    return func(*args, **kwargs)

            return wrapper
        return profile_decorator
    else:
        if len(args) > 0:
            max_cnt = args[0]
        else:
            max_cnt = -1
        def profile_decorator(func):

            @wraps(func)
            def wrapper(*args, **kwargs):
                if not hasattr(wrapper, 'cnt'):
                    wrapper.cnt = 1
                else:
                    wrapper.cnt += 1
                if not (max_cnt >= 0 and wrapper.cnt > max_cnt) and not sys._getframe().f_back.f_code.co_name == func.__name__:
                    prof = LineProfiler()
                    try:
                        return prof(func)(*args, **kwargs)
                    finally:
                        prof.print_stats()
                else:
                    return func(*args, **kwargs)

            return wrapper
        return profile_decorator
