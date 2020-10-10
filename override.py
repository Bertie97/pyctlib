#! python3 -u
#  -*- coding: utf-8 -*-

from pyctlib.typehint import *

def override(*arg):
    """
    Usage 1:

    """
    if len(arg) == 1: arg = arg[0]
    if not iterable(arg):
        func = arg
        if not callable(func): raise SyntaxError("Wrong usage of @override. ")
        class override_wrapper:
            def __init__(self, f): self.default = f; self.func_list = []

            def __call__(self, *args, **kwargs):
                if not kwargs and len(args) == 1 and callable(args[0]):
                    if (args[0].__name__ == "_" or func.__name__ in args[0].__name__) and \
                        not isoftype(args, func.__annotations__.get(func.__code__.co_varnames[0], int)):
                        self.func_list.append(params(args[0])); return
                for f in self.func_list:
                    try: return f(*args, **kwargs)
                    except TypeHintError: continue
                try:
                    ret = self.default(*args, *kwargs)
                    if len(self.func_list) == 0 and (callable(ret) or iterable(ret) and all([callable(x) for x in ret])):
                        if callable(ret): ret = (ret,)
                        if iterable(ret):
                            for f in ret:
                                try: return f(*args, **kwargs)
                                except TypeHintError: continue
                    else: return ret
                except TypeHintError: pass
                for name, value in func.__globals__.items():
                    if callable(value) and (name.startswith(func.__name__) or
                                            name.endswith(func.__name__)) and name != func.__name__:
                        try: return value(*args, **kwargs)
                        except TypeHintError: continue
                raise NameError("No {func}() matches arguments {args}.  "
                                .format(func=func.__name__, args=', '.join([repr(x) for x in args] + ['='.join((repr(x[0]), repr(x[1]))) for x in kwargs.items()])))

        owrapper = override_wrapper(func)
        @wraps(func)
        def final_wrapper(*x): return owrapper(*x)
        return final_wrapper
    else:
        functionlist = arg
        def wrap(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                for function in functionlist:
                    # print(function.__name__, args, kwargs)
                    if not callable(function): function = eval(function)
                    try: return function(*args, **kwargs)
                    except TypeHintError: continue
                raise NameError("No {func}() matches arguments {args}. "
                                .format(func=func.__name__, args=', '.join([repr(x) for x in args] + ['='.join((repr(x[0]), repr(x[1]))) for x in kwargs.items()])))
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
