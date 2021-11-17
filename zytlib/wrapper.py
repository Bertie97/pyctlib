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
import inspect
from functools import wraps
from .strtools import delete_surround
from functools import wraps
from .touch import crash
import signal
import types

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

def decorator(*wrapper_func, use_raw = True):
    if len(wrapper_func) > 2:
        raise TypeError("Too many arguments for @decorator")
    elif len(wrapper_func) == 1:
        wrapper_func = wrapper_func[0]
    else:
        return decorator(lambda x: decorator(x, use_raw = use_raw), use_raw = use_raw)
    if not isinstance(wrapper_func, type(decorator)):
        raise TypeError("@decorator wrapping a non-wrapper")
    def wrapper(*args, **kwargs):
        if not kwargs and len(args) == 1:
            func = args[0]
            raw_func = raw_function(func)
            if isinstance(raw_func, type(decorator)):
                func_name = f"{raw_func.__name__}[{wrapper_func.__qualname__.split('.')[0]}]"
                wrapped_func = wraps(raw_func)(wrapper_func(raw_func if use_raw else func))
                wrapped_func.__name__ = func_name
                wrapped_func.__doc__ = raw_func.__doc__
                return wrapped_func
        return decorator(wrapper_func(*args, **kwargs))
    return wraps(wrapper_func)(wrapper)

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


def raw_function(func):
    if "__func__" in dir(func):
        return func.__func__
    return func

class return_type_wrapper:

    def __init__(self, _type):
        self._type = _type

    def __call__(self, func):
        func = raw_function(func)
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self._type(func(*args, **kwargs))
        return wrapper


def second_argument(*args):

    if len(args) == 1:
        second_arg = args[0]
        def wrapper(func):
            @wraps(func)
            def temp_func(first_arg):
                return func(first_arg, second_arg)
            return temp_func
        return wrapper
    elif len(args) == 2:
        second_arg = args[0]
        func = args[1]
        @wraps(func)
        def wrapper(first_arg):
            return func(first_arg, second_arg)
        return wrapper
    else:
        raise ValueError()

def destory_registered_property(func):
    """
    class A:

        def __init__(self):
            return

        @destory_registered_property
        def destory(self):
            return

        @registered_property
        def test(self):
            print("hello")
            return 1
    """
    @wraps(func)
    def wrapper(self, *args, **kwars):
        self.__registered_property = dict()
        return func(self, *args, **kwars)
    return wrapper

def registered_property(func):
    """
    class A:

        def __init__(self):
            return

        @destory_registered_property
        def destory(self):
            return

        @registered_property
        def test(self):
            print("hello")
            return 1
    """

    @wraps(func)
    def wrapper(self):
        if not hasattr(self, "__registered_property"):
            exec("self.__registered_property = dict()")
        register = eval("self.__registered_property")
        if func.__name__ in register:
            return register[func.__name__]
        register[func.__name__] = func(self)
        return register[func.__name__]
    return property(wrapper)

class A:

    def __init__(self):
        return

    @registered_property
    def test(self):
        print("hello")
        return 1

class TimeoutException(Exception):
    pass

def timeout(seconds_before_timeout):
    if seconds_before_timeout == -1:
        def wrapper(func):
            @wraps(func)
            def temp_func(*args, **kwargs):
                return func(*args, **kwargs)
            return temp_func
        return wrapper
    def decorate(f):
        def handler(signum, frame):
            raise TimeoutException
        def new_f(*args, **kwargs):
            old = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds_before_timeout)
            try:
                result = f(*args, **kwargs)
            finally:
                # reinstall the old signal handler
                signal.signal(signal.SIGALRM, old)
                # cancel the alarm
                # this line should be inside the "finally" block (per Sam Kortchmar)
                signal.alarm(0)
            return result
        # new_f.func_name = f.func_name
        return new_f
    return decorate

def repeat_trigger(func, n=1, start=1, with_input=True):
    """
    @repeat_trigger(func, n=1, start=1)
    def call():
        pass

    then the(start + n * k) time function 'call' is called, func will be automatically run.

    if with_input
        func will be passed a argument input.
        input.ret = call()
        input.num_calls = # function `call` is called
        input.k = # function `func` is called
    """
    def wrapper(f):
        @wraps(f)
        def temp_func(*args, **kwargs):
            temp_func.num_calls += 1
            ret = f(*args, **kwargs)
            if n > 0 and (diff:= (temp_func.num_calls - start)) >= 0 and diff % n == 0:
                if with_input:
                    input = types.SimpleNamespace()
                    input.ret = ret
                    input.num_calls = temp_func.num_calls
                    input.k = (diff % n) + 1
                    func(input)
                else:
                    func()
            return ret
        temp_func.num_calls = 0
        return temp_func
    return wrapper
