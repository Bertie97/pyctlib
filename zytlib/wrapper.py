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
import time
from collections import namedtuple

def wrapper_template(func, **hyper):

    @wraps(func)
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        return ret

    return wrapper

def flexible_wrapper(wrapper):

    def f_wrapper(*args, **hyper):
        if len(hyper) == 0 and len(args) == 0:
            return wrapper
        if len(hyper) == 0:
            assert len(args) == 1
            assert callable(args[0])
            func = args[0]
            return wrapper(func)
        assert len(args) == 0
        def n_wrapper(func):
            return wrapper(func, **hyper)
        return n_wrapper

    return f_wrapper

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

def partial_function(func, fixed_n=tuple(), fixed_value=tuple(), **new_kwargs):
    if isinstance(fixed_n, int):
        fixed_n = tuple([fixed_n])
        fixed_value = tuple([fixed_value])
    assert len(fixed_n) == len(fixed_value)

    @wraps(func)
    def wrapper(*args, **kwargs):
        f_args = tuple()
        fixed_n = wrapper.fixed_n
        fixed_value = wrapper.fixed_value
        new_kwargs = wrapper.new_kwargs.copy()
        new_kwargs.update(kwargs)
        index = 0
        while len(fixed_n) > 0:
            while index < fixed_n[0]:
                if len(args) == 0:
                    raise RuntimeError("no enough input")
                f_args = (*f_args, args[0])
                args = args[1:]
                index += 1
            f_args = (*f_args, fixed_value[0])
            fixed_value = fixed_value[1:]
            fixed_n = fixed_n[1:]
            index += 1
        f_args = f_args + args
        return func(*f_args, **new_kwargs)

    wrapper.fixed_n = fixed_n
    wrapper.fixed_value = fixed_value
    wrapper.new_kwargs = new_kwargs
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

def registered_method(*args, **kwargs):
    if len(args) == 1 and callable(func:= args[0]):
        @wraps(func)
        def wrapper(self):
            if not hasattr(self, "_{}__registered_property".format(self.__class__.__name__)):
                exec("self._{}__registered_property = dict()".format(self.__class__.__name__))
            register = eval("self._{}__registered_property".format(self.__class__.__name__))
            if func.__name__ in register:
                return register[func.__name__]
            register[func.__name__] = func(self)
            return register[func.__name__]
        return wrapper
    elif len(args) == 1 and isinstance(n:= args[0], int) and n > 0:
        pass
    elif len(args) == 0 and (n:= kwargs.pop("n", -1) > 0):
        pass
    else:
        raise RuntimeError

    def o_wrapper(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, "_{}__registered_property".format(self.__class__.__name__)):
                exec("self._{}__registered_property = dict()".format(self.__class__.__name__))
            register = eval("self._{}__registered_property".format(self.__class__.__name__))
            if func.__name__ in register and args[:n] in register[func.__name__]:
                return register[func.__name__][args[:n]]
            if func.__name__ not in register:
                register[func.__name__] = dict()
            register[func.__name__][args[:n]] = (ret:= func(self, *args, **kwargs))
            return ret
        return wrapper
    o_wrapper.n = n
    return o_wrapper

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

        @registered_method
        def func_1(self):
            print("hello from func_1")
            return 2

        @registered(n=1)
        def func_2(self, a):
            print("hello from func_2")
            return a
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "_{}__registered_property".format(self.__class__.__name__)):
            exec("self._{}__registered_property = dict()".format(self.__class__.__name__))
        register = eval("self._{}__registered_property".format(self.__class__.__name__))
        if func.__name__ in register:
            return register[func.__name__]
        register[func.__name__] = func(self, )
        return register[func.__name__]
    return property(wrapper)

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

class FunctionTimer:

    def __init__(self, fast_threshold=-1, disable=False):
        from .table import table
        self.fast_threshold = fast_threshold
        self.__funcs = {}
        self.disable = disable

    def timer(self, func):
        if self.disable:
            return func

        @wraps(func)
        def wrapper(*args, **kwargs):
            old = wrapper.record
            if self.fast_threshold > 0 and old[0] > self.fast_threshold:
                ret = func(*args, **kwargs)
                elapse = old[1] / old[0]
            else:
                start = time.time()
                ret = func(*args, **kwargs)
                end = time.time()
                elapse = end - start
            wrapper.record = (old[0] + 1, old[1] + elapse)
            return ret

        wrapper.record = (0, 0)
        self.__funcs[func] = wrapper
        return wrapper

    @property
    def funcs(self):
        from .table import table
        ret = table()
        recored_type = namedtuple("func_recorder", ["calls", "total", "mean"])
        for f, w in self.__funcs.items():
            c = w.record
            if c[0] == 0:
                mean = 0
            else:
                mean = c[1] / c[0]
            ret[f] = recored_type(c[0], c[1], mean)
        return ret

    def __str__(self):
        return f"Timer: {self.funcs}"

    def __repr__(self):
        return str(self)
