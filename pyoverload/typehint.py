#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package pyoverload
##############################
__all__ = """
    inheritable
    iterable
    isarray
    isdtype
    isatype
    isoftype
    isclassmethod

    Type
    listargs
    params

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
    Array
    Scalar
    IntScalar
    FloatScalar
""".split()

import re, sys
from copy import deepcopy
from .utils import decorator, get_environ_vars

class TypeHintError(Exception): pass

_mid = lambda x: x[1] if len(x) > 1 else x[0]
_rawname = lambda s: _mid(str(s).split("'")).split('.')[-1]

def inheritable(x):
    """
    inheritable(x) -> bool

    Returns whether a class type can be inherited. 

    Args:
        x (type): the input class name.

    Example::

        >>> inheritable(int)
        True
        >>> inheritable(bool)
        False
    """
    try:
        class tmp(x): pass
        return True
    except TypeError: return False

def iterable(x):
    """
    iterable(x) -> bool

    Returns whether a type or instance can be iterated. 

    Args:
        x (any): the input variable.

    Example::

        >>> iterable(x for x in range(4))
        True
        >>> inheritable({2, 3})
        True
        >>> inheritable("12")
        False
    """
    # Test types
    try:
        if isinstance(x, Type):
            x = Type.extractType([x])
            if all([iterable(i) for i in x]): return True
            return False
    except NameError: pass
    if isinstance(x, type):
        try: x = x()
        except Exception: return False
    # Test instances
    if isinstance(x, str): return False
    try:
        iter(x); len(x)
        return not callable(x)
    except Exception: return False
    return False

def isarray(x):
    """
    isarray(x) -> bool

    Returns whether an instance is an array. 

    Args:
        x (any): the input variable.

    Example::

        >>> isarray(np.array([2, 3]))
        True
        >>> isarray(torch.tensor([2, 3]))
        True
        >>> isarray([1, 2])
        False
    """
    if not isatype(x) and 'shape' in dir(x): return True
    return False

def isdtype(x):
    """
    isdtype(x) -> bool

    Returns whether an instance is a data type of numpy or pytorch or tensorflow etc. . 

    Args:
        x (any): the input variable.

    Example::

        >>> isdtype(np.int32)
        True
        >>> isdtype(torch.float64)
        True
        >>> isdtype(int)
        False
    """
    return 'dtype' in repr(type(x)).lower()

def isatype(x):
    """
    isatype(x) -> bool

    Returns whether an instance is a type. 
    
    Note:
        Types detected includes all python classes and pyctlib.Type as well as:
        1. None representing any type
        2. a list or iterable set of types like [int, float]

    Args:
        x (any): the input variable.

    Example::

        >>> isatype(np.array)
        False
        >>> isatype(np.ndarray)
        True
        >>> isatype(None)
        True
        >>> isatype([int, np.ndarray])
        True
    """
    if x is None: return True
    if isinstance(x, type): return True
    if isinstance(x, Type): return True
    if iterable(x):
        if len(x) <= 0: return False
        for xt in x:
            if isinstance(x, type): continue
            if isinstance(x, Type): continue
            break
        else: return True
    return False

def isoftype(x, xtype, environ_func = None):
    """
    isoftype(x, xtype) -> bool

    Returns whether an instance 'x' is of type 'xtype'. 
    
    Note:
        'xtype' can be provided in one of the following fashions:
        1. a pyctlib.Type like Int, Dict[2]@{int: str} or List<<Int>>[]
        2. a str representing a type, either the full name of the required type or a name that can be computed when 
        1. None representing any type
        2. a list or iterable set of types like [int, float]

    Args:
        x (any): the input variable.

    Example::

        >>> isatype(np.array)
        False
        >>> isatype(np.ndarray)
        True
        >>> isatype(None)
        True
        >>> isatype([int, np.ndarray])
        True
    """
    if isinstance(xtype, str):
        local_vars = get_environ_vars(isoftype if environ_func is None else environ_func)
        local_vars.update(locals())
        locals().update(local_vars.simplify())
        try: xtype = eval(xtype)
        except: return xtype in [_rawname(t) for t in type(x).__mro__]
    if xtype == type: return isatype(x)
    if xtype is None: return True
    if isinstance(xtype, type):
        if isinstance(xtype, Type): return xtype(x)
        if isinstance(x, xtype): return True
        return False
    if type(xtype) in (list, tuple, set):
        if type(x) in xtype: return True
        for xt in xtype:
            if isoftype(x, xt): return True
        return False
    if callable(xtype) and xtype(x): return True
    return False

def isclassmethod(x):
    if not callable(x): return False
    if '__qualname__' not in dir(x): return False
    p = x.__qualname__.split('.')
    if len(p) <= 1: return False
    try:
        class_name = '.'.join(p[:-1])
        if isinstance(x.__globals__.get(class_name, int), type): return True
        else: return False
    except: return False

class _strIO:
    def __init__(self): self._str_ = ''
    def write(self, s): self._str_ += s
    def __str__(self): return self._str_
    def split(self, c=None): return self._str_.split(c)

def _getDeclaration(func):
    ss = _strIO()
    oldout = sys.stdout
    sys.stdout = ss
    help(func)
    sys.stdout = oldout
    dec = [l for l in ss.split('\n') if len(l) > 0 and 'Help' not in l][0].replace("def ", '').strip().strip(':')
    return dec[dec.index('('):]

def _get_func_name(f):
    fname = f.__name__.split('[')[0]
    if fname.endswith('__0__') or fname.endswith('__default__'):
        fname = '__'.join(fname.split('__')[:-2])
        f.__name__ = fname + '['.join(f.__name__.split('[')[1:])
    return fname

def listargs(*_T):
    '''
    Expand the args to arguments if they are in a list.
    '''
    if len(_T) == 1 and iterable(_T[0]): _T = _T[0]
    @decorator
    def wrap(func):
        def wrapper(*args, **kwargs):
            if isclassmethod(func):
                pre = args[:1]
                args = args[1:]
            else: pre = tuple()
            if len(args) == 1 and type(args[0]) in (list, tuple, set):
                for item in args[0]:
                    if not isoftype(item, _T):
                        if _T not in (list, tuple, dict, set):
                            raise TypeHintError("Unsupported types of arguments for " +
                                                _get_func_name(func) + "(): " + str(item) + '. ')
                        break
                else: args = args[0]
            return func(*pre, *args, **kwargs)
        return wrapper
    return wrap

class Type(type):

    @staticmethod
    def extractType(array):
        output = []
        for x in array:
            if isinstance(x, Type):
                output.extend(Type.extractType(x.types))
            else: output.append(x)
        return output

    @listargs(type)
    def __new__(cls, *_T, name=None, shape=tuple(), inv=False, ext=False, itypes=None):
        if len([0 for t in _T if not isatype(t)]) > 0:
            raise SyntaxError("Wrong parameter type. ")
        if len(_T) == 1 and type(_T[0]) == Type:
            self = super().__new__(cls, _T[0].__name__, _T[0].__bases__, {})
            self.copyfrom(_T[0])
        else:
            _T = Type.extractType(_T)
            tmpT = []
            have_basic = False
            for t in _T:
                while True:
                    if inheritable(t):
                        if t.__bases__ and t.__bases__[0] == object:
                            if have_basic: break
                            else: have_basic = True
                        if t in tmpT: break
                        tmpT.append(t); break
                    t = t.__bases__[0]
            self = super().__new__(cls, cls.__name__ + '.' + Type.strT(*_T), tuple(tmpT), {})
            self.inv = inv
            self.name = name
            self.types = _T
            self.shape = shape
            self.extendable = ext
            self.itemtypes = itypes
        return self

    def __init__(*args, **kwargs): pass

    @listargs(type)
    @staticmethod
    def strT(*_T):
        if isoftype(_T, (list, tuple, set)) and len(_T) > 1:
            return '(' + ', '.join([Type.strT(t) for t in _T]) + ')'
        if iterable(_T) and len(_T) == 1 and isoftype(_T[0], dict): _T = _T[0]
        if isoftype(_T, dict):
            return '{' + ', '.join([':'.join((Type.strT(k), Type.strT(v))) for k, v in _T.items()]) + '}'
        if len(_T) > 0: return _rawname(_T[0])
        return ''

    def copyfrom(self, other):
        self.types = other.types
        if len(other.shape) > 0: self = self[other.shape]
        if other.extendable: self = +self
        if other.inv: self = ~self
        self @= other.itemtypes
        self.name = other.name

    def __getitem__(self, i=None):
        if i == 0 or i is None: i = tuple()
        if isatype(i): return self@i
        if isoftype(i, int): i = (i,)
        elif isoftype(i, [list, tuple]): pass
        else: raise TypeHintError("Wrong size specifier. ")
        if all([x is None or iterable(x) for x in self.types]):
            return Type(self.types, name=self.name, shape=i, inv=self.inv, ext=self.extendable, itypes=self.itemtypes)
        else: return Type([list, tuple, set], name=f"List", shape=i, inv=self.inv, ext=self.extendable, itypes=self)

    def __lshift__(self, i):
        if all([issubclass(x, dict) for x in self.types]) and len(i) == 2: return self@{i[0]: i[1]}
        elif all([iterable(x) for x in self.types]) or all([isarray(x) for x in self.types]): return self@i
        raise TypeHintError("Only tensors, iterable or dict types can use <> representations.")

    def __rshift__(self, l):
        if not isinstance(l, list): raise TypeHintError("Please use iterable<type>[length] to specify a type.")
        if len(l) == 0: return self[-1]
        if len(l) == 1: return self[l[0]]
        return self[tuple(l)]

    def __matmul__(self, other):
        if other is None: return
        if not iterable(self): raise TypeError("Only iterable Type can use @ to specify the items. ")
        if isinstance(other, Type):
            return Type(self.types, name=self.name, shape=self.shape, inv=self.inv, ext=self.extendable, itypes=other.itemtypes if iterable(other) else other)
        if all([isarray(x) for x in self.types]) and isdtype(other):
            return Type(self.types, name=self.name, shape=self.shape, inv=self.inv, ext=self.extendable, itypes=other)
        shape = self.shape
        if len(shape) == 0 and isoftype(other, (list, tuple)) and len(other) > 0: shape = [len(other)]
        if not iterable(other): other = (other,)
        if len(shape) == 1 and len(other) not in (1, shape[0]): raise TypeHintError("Wrong number of item types for @. ")
        if isinstance(other, dict) and not all([issubclass(x, dict) for x in self.types]): raise TypeHintError("Please do use {a:b} typehint for dicts.")
        return Type(self.types, name=self.name, shape=shape, inv=self.inv, ext=self.extendable, itypes=other)

    def __pos__(self):
        return Type(self.types, name=self.name, shape=self.shape, inv=self.inv, ext=True, itypes=self.itemtypes)

    def __invert__(self):
        return Type(self.types, name=self.name, shape=self.shape, inv=not self.inv, ext=self.extendable, itypes=self.itemtypes)

    def __str__(self):
        string = f"<Type '{'*' * self.extendable}{'non-' * self.inv}{self.name if self.name else _rawname(self.types[0]).capitalize()}"
        if self.itemtypes != None:
            if isinstance(self.itemtypes, Type):
                string += f"<<{self.itemtypes.name}>>"
            elif isinstance(self.itemtypes, type):
                string += f"<<{_rawname(self.itemtypes)}>>"
            else: string += str(self.itemtypes)
        if self.shape: string += str(list(self.shape))
        string += "'>"
        return string

    __repr__ = __str__

    @property
    def len(self):
        if len(self.shape) == 0: return -1
        prod = 1; unsure = False
        for s in self.shape:
            if s in (None, -1): unsure = True; continue
            prod *= s
        if unsure: return -prod
        else: return prod

    def __len__(self): return abs(self.len)

    def __iter__(self):
        if len(self.shape) == 0: raise TypeHintError(self.__str__() + " is not iterable. ")

    def __call__(self, arg):
        true = not self.inv
        false = self.inv
        if isoftype(arg, self.types):
            if len(self.shape) == 0: pass
            elif len(self.shape) == 1 and (len(arg) == self.shape[0] or self.shape[0] in (None, -1)): pass
            elif len(self.shape) > 1 and shape in arg.__dict__ and \
                all([a==b or b in (None, -1) for a, b in zip(arg.shape, self.shape)]): pass
            else: return false
            if self.itemtypes != None:
                if isinstance(self.itemtypes, dict):
                    if len(self.itemtypes) > 1 or len(self.itemtypes) <= 0:
                        raise TypeHintError("Wrong item types after @, only one set of dict annotation is allowed. ")
                    keytype = list(self.itemtypes.keys())[0]
                    valtype = list(self.itemtypes.values())[0]
                    for k, v in arg.items():
                        if not isoftype(k, keytype) or not isoftype(v, valtype): return false
                elif iterable(self.itemtypes):
                    if len(self.itemtypes) == 1: self.itemtypes *= len(arg)
                    for x, xt in zip(arg, self.itemtypes):
                        if not isoftype(x, xt): return false
                elif isarray(arg):
                    dname = lambda dt: re.findall(r'\b\w+\b', repr(dt))[-1]
                    if not iterable(self.itemtypes): itypes = [self.itemtypes]
                    else: itypes = self.itemtypes
                    for dt in Type.extractType(itypes):
                        if isdtype(dt): dt = dname(dt)
                        if isatype(dt): dt = _rawname(dt)
                        if dname(arg.dtype) == dt: return true
                    return false
                elif isatype(self.itemtypes):
                    for x in arg:
                        if not isoftype(x, self.itemtypes): return false
                else: raise TypeHintError("Invalid item types after @, please use iterable or types.")
            return true
        return false

def T(main_type, *args, name=None):
    class Default_Type(Type):
        def __call__(self, arg):
            if super().__call__(arg): return main_type
            else: return None
    return Default_Type(main_type, *args, name=name)

Bool = T(bool)
Int = T(int)
Float = T(float)
Str = T(str)
Set = T(set)
List = T(list)
Dict = T(dict)
Tuple = T(tuple)

class CallableType(Type):
    def __call__(self, x): return callable(x)
Callable = CallableType(name="Callable")
Function = T(type(iterable))
Method = T(type(Bool.copyfrom), type("".split), type(Int.__str__), name="Method")
Lambda = T(type(lambda: None), name="Lambda")
Func = T(type(iterable), Method, Lambda, name="FunctionType")
Real = T(float, int, name="Real")
Iterable = T(tuple, list, dict, set, name="Iterable")
null = type(None)
Null = T(null)
real = [int, float]

class ArrayType(Type):
    @property
    def Numpy(self):
        try:
            import numpy as np
            return Type(np.ndarray)
        except ImportError: pass

    @property
    def Torch(self):
        try:
            import torch
            return Type(torch.Tensor)
        except ImportError: pass

    @property
    def TensorFlow(self):
        try:
            ss = strIO()
            olderr = sys.stderr
            sys.stderr = ss
            import tensorflow as tf
            sys.stderr = olderr
            return Type(tf.Tensor)
        except ImportError: pass

    @property
    def TensorPlus(self):
        try:
            import pyctlib.torchplus as tp
            return Type(tp.Tensor)
        except ImportError: pass

    def __call__(self, x): return isarray(x)

Array = ArrayType(name="Array")

class ScalarType(Type):
    def __call__(self, x):
        if isarray(x):
            if not (len(x.shape) == 0 or all([i==1 for i in x.shape])): return False
            res = False
            for t in self.types:
                res = res or _rawname(t) in str(x.dtype)
            return res
        else: return super().__call__(x)

Scalar = ScalarType(Float, Int, name="Scalar")
IntScalar = ScalarType(Int, name="IntScalar")
FloatScalar = ScalarType(Float, name="FloatScalar")

def getArgs(func, *args, **kwargs):
    getTypes = False
    allargs = {}
    if "__type__" in kwargs: getTypes = kwargs.pop('__type__')
    if "__return__" in kwargs: allargs['return'] = kwargs.pop('__return__')
    inputargs = list(args)
    normalargs = func.__code__.co_varnames[:func.__code__.co_argcount]
    extargs = func.__code__.co_varnames[func.__code__.co_argcount:]
    dec = _getDeclaration(func)
    eargs = ''.join(re.findall(r"[^*]\*{1} *(\w+)\b", dec))
    ekwargs = ''.join(re.findall(r"[^*]\*{2} *(\w+)\b", dec))
    lendefault = 0 if not func.__defaults__ else len(func.__defaults__)
    rg = lendefault - func.__code__.co_argcount, lendefault
    for var, idefault in zip(normalargs, range(*rg)):
        if len(inputargs) > 0:
            if var in kwargs:
                raise TypeHintError(_get_func_name(func) + "() got multiple values for argument " + repr(var))
            allargs[var] = inputargs.pop(0); continue
        if var in kwargs:
            if getTypes: allargs[var] = type(kwargs.pop(var))
            else: allargs[var] = kwargs.pop(var)
            continue
        if idefault < 0:
            error = _get_func_name(func) + "() missing required positional argument: " + repr(var)
            raise TypeHintError(error)
        if not func.__defaults__ or not 0 <= idefault < len(func.__defaults__): continue
        if getTypes: allargs[var] = type(func.__defaults__[idefault])
        else: allargs[var] = func.__defaults__[idefault]
    if eargs: allargs[eargs] = tuple(inputargs); inputargs = []
    if ekwargs: allargs[ekwargs] = inputargs.pop(0) if len(inputargs) == 1 else kwargs; kwargs = {}
    for addkwarg in extargs:
        if addkwarg in (eargs, ekwargs): continue
        if addkwarg in kwargs:
            if getTypes: allargs[addkwarg] = type(kwargs.pop(addkwarg))
            else: allargs[addkwarg] = kwargs.pop(addkwarg)
        else:
            if not func.__kwdefaults__ or addkwarg not in func.__kwdefaults__: continue
                # raise TypeHintError(func.__name__ + "() got an undefined parameter " + addkwarg + ". ")
            if getTypes: allargs[addkwarg] = type(func.__kwdefaults__[addkwarg])
            else: allargs[addkwarg] = func.__kwdefaults__[addkwarg]
    if len(kwargs) > 0: raise TypeHintError(_get_func_name(func) + "() got an unexpected " + 
        "keyword argument " + repr(list(kwargs.keys())[0]))
    if len(inputargs) > 0: raise TypeHintError(_get_func_name(func) +
        "() takes from {lower} to {upper} positional arguments but {real} were given. "
        .format(lower=-rg[0], upper=rg[1] - rg[0], real=len(inputargs) + rg[1] - rg[0]))
    return allargs

@decorator
def params(*types, run=True, **kwtypes):
    israw = len(kwtypes) == 0 and len(types) == 1 and \
            callable(types[0]) and not isatype(types[0])
    @decorator
    def wrap(func):
        org_dec = _getDeclaration(func)
        if isclassmethod(func): alltypes = (None,) + types
        else: alltypes = types
        if israw or (len(kwtypes) == 0 and len(types) == 0):
            annotations = {k: v.strip('\'"') if isinstance(v, str) else v for k, v in func.__annotations__.items()}
        else: annotations = getArgs(func, *alltypes, **kwtypes, __type__=True)
        def wrapper(*args, **kwargs):
            if israw or (len(kwtypes) == 0 and len(types) == 0):
                _annotations = {k: v.strip('\'"') if isinstance(v, str) else v for k, v in func.__annotations__.items()}
            else: _annotations = getArgs(func, *alltypes, **kwtypes, __type__=True)
            eargs = ''.join(re.findall(r"[^*]\*{1} *(\w+)\b", org_dec))
            ekwargs = ''.join(re.findall(r"[^*]\*{2} *(\w+)\b", org_dec))
            _values = getArgs(func, *args, **kwargs)
            if ekwargs:
                assert ekwargs in _values
                if ekwargs in _annotations:
                    if iterable(_annotations[ekwargs]):
                        _values.update(_values[ekwargs]); _values.pop(ekwargs)
                        if len(_annotations[ekwargs]) == 0: _annotations.pop(ekwargs)
                        else:
                            _annotations.update(_annotations[ekwargs])
                            _annotations.pop(ekwargs)
                    else:
                        if not extendable(_annotations[ekwargs]):
                            print("Warning: auto extended non-extendable type. Please check your typehint. ")
                        _annotations[ekwargs] = Type(dict)@{str:_annotations[ekwargs]}
            if eargs:
                assert eargs in _values
                if eargs in _annotations:
                    if iterable(_annotations[eargs]):
                        if len(_annotations[eargs]) == 0: _annotations.pop(eargs)
                        elif len(_annotations[eargs]) > 1:
                            raise TypeHintError(_get_func_name(func) + "() has too many type restraints. ")
                        else:
                            if not extendable(_annotations[eargs][0]):
                                print("Warning: auto extended non-extendable type. Please check your typehint. ")
                            _annotations[eargs] = _annotations[eargs][0]
                    if eargs in _annotations:
                        _annotations[eargs] = Type(list, tuple)@_annotations[eargs]
            for arg in _values:
                if arg in _annotations:
                    if isoftype(_values[arg], _annotations[arg], environ_func=func): continue
                    break
            else:
                if run:
                    retval = func(*args, **kwargs)
                    if 'return' not in _annotations or isoftype(retval, _annotations['return'], environ_func=func): return retval
                    raise TypeHintError(_get_func_name(func) + "() returns an invalid value.")
                else: return None
            raise TypeHintError(_get_func_name(func) + "() has argument " + arg + " of wrong type. \
Expect type {type} but get {value}.".format(type=repr(_annotations[arg]), value=repr(_values[arg])))
        org_dec = _getDeclaration(func)
        dec = org_dec
        for arg in annotations:
            if arg == 'return':
                dec = dec[:dec.rindex(')')] + f") -> {_rawname(annotations[arg])}"
                continue
            res = re.search(rf"\b{arg}\b", dec)
            if res:
                idx = res.span()[0]
                pairs = {')': '(', '}': '{', ']': '['}
                count = {}
                for i in range(idx, len(dec)):
                    if dec[i] in pairs:
                        if count.get(pairs[dec[i]], 0) <= 0: break
                        count[pairs[dec[i]]] -= 1; continue
                    if dec[i] in "'\"" and dec[i] in count and count[dec[i]] > 0: count[dec[i]] -= 1; continue
                    if dec[i] in "({['\"": count[dec[i]] = count.get(dec[i], 0) + 1; continue
                    if dec[i] == ',' and (len(count) == 0 or max(count.values()) == 0): break
                dec = dec[:idx] + f"{arg}:{_rawname(annotations[arg])}" + dec[i:]
        if func.__doc__: lines = func.__doc__.strip().split('\n')
        else: lines = []
        if len(lines) < 1 or org_dec != lines[0]: lines.insert(0, org_dec)
        if len(lines) < 2 or dec != lines[1]: lines.insert(1, dec)
        if len(lines) >= 3: lines.insert(2, '')
        func.__doc__ = '\n'.join(lines)
        return wrapper
    if israw: return wrap(types[0])
    else: return wrap

if __name__ == "__main__":
    print(isoftype(
        {1:0.2, '2': 4},
        Dict[2]@{(Int, str):Real}
    ))

    @params(Func, Int, +Int, __return__ = Real[2])
    def test_func(a, b=2, *k):
        return k

    print(isclassmethod(extendable))
    print(isclassmethod(Float.__len__))
    print(isclassmethod(Type.__len__))

    print(Int)
    print(List)
    print(Real)
    print(T(Real))
    print(+Real)
    print(+Tuple[2])
    print(Iterable[2])
    print(isoftype((2.0, 3, 'a', None), Tuple[3]))
    print(List[2]@[int, Tuple[3]])
    print((List[2]@[int, Tuple[3]])([1, (2.0, 3, 'a')]))
    print(isoftype(0.1, Int))
    test_func(lambda x: 1, 3, 4, 5)
