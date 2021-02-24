#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package pyoverload
##############################
__all__ = """
    override
    overload
    params
""".split()

from functools import wraps
from .typehint import *
from .typehint import __all__ as typehint_all
from .typehint import TypeHintError, _getDeclaration
from .utils import get_environ_vars, _get_wrapped, raw_function, decorator

__all__ += typehint_all

_debug = False

def set_debug_mode(m): global _debug; _debug = m

def _wrap_params(f):
    if hasattr(f, '__wrapped__'):
        if '[params]' in f.__name__: return None, f
        else: return f, params(run=False)(_get_wrapped(f))
    return f, params(run=False)(f)

def _collect_declarations(func, collection, place_first=False, error=''):
    f = _get_wrapped(raw_function(func))
    toadd = '\t'.join([_getDeclaration(f)] + ([error] if error else []))
    if place_first: collection.insert(0, toadd)
    else: collection.append(toadd)
    # if f.__doc__:
    #     lines = f.__doc__.split('\n')
    #     toadd = ''
    #     for l in lines:
    #         if not l.strip(): continue
    #         if l.startswith(_get_func_name(f)) or l[0] == '(': toadd = l
    #         else: break
    #     if toadd:
    #         toadd = '\t'.join([toadd, error])
    #         if place_first: collection.insert(0, toadd)
    #         else: collection.append(toadd)

def _get_func_name(f, change_name = True):
    f = raw_function(f)
    fname = f.__name__.split('[')[0]
    if fname.endswith('__0__') or fname.endswith('__default__'):
        fname = '__'.join(fname.split('__')[:-2])
        if change_name: f.__name__ = fname + '['.join(f.__name__.split('[')[1:])
    return fname

@decorator
def overload(func):
    local_vars = get_environ_vars()
    fname = raw_function(func).__name__.split('[')[0]
    rawfname = _get_func_name(func, change_name = False)
    key = f"__overload_{rawfname}__"
    overrider = f"__override_{rawfname}__"
    if key in local_vars:
        local_vars[key] = local_vars[key] + 1
        new_name = f"__{fname}_overload_{local_vars[key]}"
        local_vars[new_name] = local_vars[overrider](func)
    else:
        local_vars[key] = 0
        local_vars[overrider] = override(func)
    exec(f"def {rawfname}(*args, **kwargs): return {overrider}(*args, **kwargs)", local_vars)
    # func.__name__ = '['.join([new_name] + func.__name__.split('[')[1:])
    return local_vars[rawfname]

def _try_imp(f, by, collect, place_first=False):
    args, kwargs = by
    run_func, test_func = _wrap_params(f)
    try:
        ret = test_func(*args, **kwargs)
        if run_func is not None: ret = run_func(*args, **kwargs)
        return ret
    except TypeHintError as e:
        _collect_declarations(f, collect, place_first=place_first, error=str(e))
        if _debug: print(str(e))
        return "__try_imp_failed__"

@decorator
def override(*arg):
    """
    Usage 1:

    """

    if len(arg) == 1: arg = arg[0]
    if not iterable(arg):
        if not Functional(arg): raise SyntaxError("Wrong usage of @override. ")
        class override_wrapper:
            def __init__(self, argf):
                self.mainfunc = argf
                self.func_list = [argf]
                fname = raw_function(argf).__name__.split('[')[0]
                if fname.endswith('__0__') or fname.endswith("__default__"): self.default = 0
                else: self.default = None
                self.must_default = False

            def __call__(self, *args, **kwargs):
                if len(args) > 0:
                    argf = args[0]
                    if not kwargs and len(args) == 1 and Functional(argf):
                        fname = raw_function(argf).__name__.split('[')[0]
                        funcname = _get_func_name(self.mainfunc)
                        if fname == "_" or funcname in fname:
                            if fname.endswith('__0__') or fname.endswith("__default__"):
                                if self.default is not None: raise TypeError("Only one default function is acceptable. ")
                                self.default = len(self.func_list)
                            self.func_list.append(argf); return
                            self.must_default = True
                dec_list = []
                if len(self.func_list) == 1:
                    run_func, test_func = _wrap_params(self.func_list[0])
                    try:
                        ret = test_func(*args, **kwargs)
                        if run_func is not None: ret = run_func(*args, **kwargs)
                        if (
                            Functional(ret) and raw_function(ret).__name__.endswith('[params]') or 
                            Sequence(ret) and len(ret) > 0 and all([Functional(x) and raw_function(x).__name__.endswith('[params]') for x in ret])
                        ):
                            dec_list.clear()
                            if Functional(ret): ret = (ret,)
                            for f in ret:
                                ret = _try_imp(f, by=(args, kwargs), collect=dec_list)
                                if isinstance(ret, str) and ret == "__try_imp_failed__": pass
                                else: return ret
                        else: return ret
                    except TypeHintError as e:
                        _collect_declarations(self.func_list[0], dec_list, error=str(e))
                        if _debug: print(str(e))
                elif len(self.func_list) > 1:
                    if self.must_default and self.default is None: self.default = 0
                    for i, f in list(enumerate(self.func_list)) + ([(-1, self.func_list[self.default])] if self.default is not None else []):
                        if i == self.default: continue
                        ret = _try_imp(f, by=(args, kwargs), collect=dec_list, place_first=i==-1)
                        if isinstance(ret, str) and ret == "__try_imp_failed__": pass
                        else: return ret
                else:
                    for name, value in arg.__globals__.items():
                        name = name.replace('override', '').strip('_')
                        if Functional(value) and (name.startswith(raw_function(self.mainfunc).__name__ + '_') or
                                                name.endswith('_' + raw_function(self.mainfunc).__name__)) and \
                                                name != raw_function(self.mainfunc).__name__:
                            ret = _try_imp(value, by=(args, kwargs), collect=dec_list)
                            if isinstance(ret, str) and ret != "__try_imp_failed__": return ret
                func_name = _get_func_name(_get_wrapped(raw_function(self.func_list[self.default if self.default else 0]))).split("_overload")[0]
                dec_list = [func_name + x[x.index('('):] for x in dec_list]
                raise NameError("No {func}() matches arguments {args}. ".format(
                    func=func_name, 
                    args=', '.join([repr(x) for x in args] + 
                        ['='.join((str(x[0]), repr(x[1]))) for x in kwargs.items()])
                ) + "All available usages are:\n{}".format('\n'.join(dec_list)))

        owrapper = override_wrapper(arg)
        @wraps(arg)
        def final_wrapper(*args, **kwargs): return owrapper(*args, **kwargs)
        return final_wrapper
    else:
        functionlist = arg
        @decorator
        def wrap(func):
            def wrapper(*args, **kwargs):
                dec_list = []
                for f in functionlist:
                    if isinstance(f, str): f = eval(f)
                    ret = _try_imp(f, by=(args, kwargs), collect=dec_list)
                    if isinstance(ret, str) and ret != "__try_imp_failed__": return ret
                func_name = _get_func_name(func)
                dec_list = [func_name + x[x.index('('):] for x in dec_list]
                raise NameError("No {func}() matches arguments {args}. ".format(
                    func=func_name, 
                    args=', '.join([repr(x) for x in args] + 
                        ['='.join((str(x[0]), repr(x[1]))) for x in kwargs.items()])
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
