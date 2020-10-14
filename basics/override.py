#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Package PyCTLib
##############################
__all__ = """
    override
    params

    iterable
    isarray
    isdtype
    isatype
    isoftype

    Type
    Bool
    Int
    Float
    Str
    Set
    List
    Tuple
    Dict
    Callable
    Function
    Method
    Lambda
    Func
    Real real
    Iterable
    Null null
""".split()

from functools import wraps
from pyctlib.basics.wrapper import *
from pyctlib.basics.typehint import *
from pyctlib.basics.typehint import TypeHintError

def _wrap_params(f):
    if '__wrapped__' in dir(f) and '[params]' in f.__name__: return f
    return params(f)

def _collect_declarations(func, collection, place_first=False):
    try_func = _wrap_params(func)
    if try_func.__doc__:
        lines = try_func.__doc__.split('\n')
        if len(lines) > 1: toadd = lines[1]
        else: toadd = lines[0]
        if place_first: collection.insert(0, toadd)
        else: collection.append(toadd)
    return try_func

@decorator(use_raw = False)
def override(*arg):
    """
    Usage 1:

    """

    if len(arg) == 1: arg = arg[0]
    if not iterable(arg):
        func = raw_function(arg)
        if not callable(func): raise SyntaxError("Wrong usage of @override. ")
        class override_wrapper:
            def __init__(self, f): self.default = f; self.func_list = []

            def __call__(self, *args, **kwargs):
                f = raw_function(args[0])
                if not kwargs and len(args) == 1 and Func(f):
                    fname = f.__name__.split('[')[0]
                    if fname == "_" or func.__name__ in fname:
                        self.func_list.append(_wrap_params(f)); return
                if '__func__' in dir(arg) and func.__qualname__.split('.')[0] in str(type(args[0])): args = args[1:]
                dec_list = []
                for f in self.func_list:
                    try: return _collect_declarations(f, dec_list)(*args, **kwargs)
                    except TypeHintError: continue
                try:
                    ret = _collect_declarations(self.default, dec_list, place_first=True)(*args, *kwargs)
                    if len(self.func_list) == 0 and (
                        Func(ret) and ret.__name__.endswith('[params]') or 
                        iterable(ret) and all([Func(x) and x.__name__.endswith('[params]') for x in ret])
                    ):
                        dec_list.clear()
                        if callable(ret): ret = (ret,)
                        if iterable(ret):
                            for f in ret:
                                try: return _collect_declarations(f, dec_list)(*args, **kwargs)
                                except TypeHintError: continue
                    else: return ret
                except TypeHintError: pass
                for name, value in func.__globals__.items():
                    if callable(value) and (name.startswith(func.__name__) or
                                            name.endswith(func.__name__)) and name != func.__name__:
                        try: return _collect_declarations(value, dec_list)(*args, **kwargs)
                        except TypeHintError: continue
                func_name = self.default.__name__.split('[')[0]
                dec_list = [func_name + x[x.index('('):] for x in dec_list]
                raise NameError("No {func}() matches arguments {args}. ".format(
                    func=func_name, 
                    args=', '.join([repr(x) for x in args] + 
                        ['='.join((repr(x[0]), repr(x[1]))) for x in kwargs.items()])
                ) + "All available usages are:\n{}".format('\n'.join(dec_list)))

        owrapper = override_wrapper(func)
        @wraps(func)
        def final_wrapper(*x): return owrapper(*x)
        return final_wrapper
    else:
        functionlist = arg
        @decorator
        def wrap(func):
            def wrapper(*args, **kwargs):
                dec_list = []
                for function in functionlist:
                    if not callable(function): function = eval(function)
                    try: return _collect_declarations(function, dec_list)(*args, **kwargs)
                    except TypeHintError: continue
                func_name = self.default.__name__.split('[')[0]
                dec_list = [func_name + x[x.index('('):] for x in dec_list]
                raise NameError("No {func}() matches arguments {args}. ".format(
                    func=func_name, 
                    args=', '.join([repr(x) for x in args] + 
                        ['='.join((repr(x[0]), repr(x[1]))) for x in kwargs.items()])
                ) + "All available usages are:\n{}".format('\n'.join(dec_list)))
            return wrapper
        return wrap

if __name__ == "__main__":
    # @params # (float, int, str, +list_[3])
    # def f(x:float, y:int, z=3, *k:list_[3], **b:float): return x, y, z, k, b

    # print(f(1.0, 1.0, 'a', [1, 1, 1], [2, 3, 4], t=3))

    @params#(float, float, str, t=int)
    def f(x:float, y:int, z=3, *, t): return x, y, z, t

    print(f(1.0, 1, 'a', t=3))

    import sys
    @params(+Str, [str, null], [str, null])
    def pyprint(*s, sep=' ', end='\n'):
        if sep is None: sep = ' '
        if end is None: end = '\n'
        sys.stdout.write(sep.join(s) + end)

    print('s', 'l', sep=None)

    isint = lambda x: abs(x - round(x)) < 1e-4
    rint = lambda x: int(round(x))
    from rational import GCD
    class rat:

        @params(int, int)
        def __init__numdenint(self, numerator, denominator):
            self.numerator, self.denominator = numerator, denominator
            self.value = self.numerator / self.denominator
            self.cancelation()

        @params([int, float], [int, float])
        def __init__numdenfloat(self, numerator, denominator):
            if isint(numerator) and isint(denominator):
                self.__init__numdenint(rint(numerator), rint(denominator))
            else: self.__init__float(numerator / denominator)

        @params(str)
        def __init__str(self, string):
            try: arg = [float(x) for x in string.split('/')]
            except Exception: raise SyntaxError("Invalid Format")
            self.__init__numdenfloat(*arg)

        @params(int)
        def __init__int(self, num): self.__init__numdenint(num, 1)

        @params(float)
        def __init__float(self, num):
            if isint(num): self.__init__int(rint(num))
            else:
                self.value = num
                self.__init__numdenint(*rat.nearest(num))

        @override(__init__int, __init__float, __init__str, __init__numdenint, __init__numdenfloat)
        def __init__(): pass

        def tuple(self): return self.numerator, self.denominator

        def cancelation(self):
            d = GCD(*self.tuple())
            self.numerator //= d
            self.denominator //= d
            if self.denominator < 0:
                self.numerator = - self.numerator
                self.denominator = - self.denominator

        def __add__(self, other):
            return rat(self.numerator * other.denominator +
                       self.denominator * other.numerator,
                       self.denominator * other.denominator)

        def __mul__(self, other):
            return rat(self.numerator * other.numerator,
                       self.denominator * other.denominator)

        def __str__(self):
            if self.denominator == 1: return str(self.numerator)
            return str(self.numerator)+'/'+str(self.denominator)

        @staticmethod
        def nearest(num, maxiter=100):
            def iter(x, d):
                if isint(x) or d >= maxiter: return int(round(x)), 1
                niter = iter(1 / (x - int(x)), d+1)
                return int(x) * niter[0] + niter[1], niter[0]
            if num >= 0: return iter(num, 0)
            num = iter(-num, 0)
            return -num[0], num[1]

    print(rat(126/270) + rat(25, 14))
    print(rat(126, 270) * rat(25, 14))
