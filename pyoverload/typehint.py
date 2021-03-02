#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package pyoverload
##############################
__all__ = """
    isofsubclass
    inheritable
    isitertype
    iterable
    isarray
    isdtype
    isatype
    isoftype
    isclassmethod
    typename

    Type
    params

    Bool
    Int
    Float
    Str
    Set
    List
    Tuple
    Dict
    Class
    Callable
    Function
    Method
    Lambda
    Functional
    Real real
    Null null
    Sequence
    sequence
    Array
    Iterable
    Scalar
    IntScalar
    FloatScalar
""".split()

import re, sys
from pyoverload.utils import decorator, get_environ_vars, raw_function, _get_wrapped

try:
    import inspect
    have_inspect = True
except ImportError: have_inspect = False

class TypeHintError(Exception): pass

_mid = lambda x: x[1] if len(x) > 1 else x[0]
_rawname = lambda s: _mid(str(s).split("'")).split('.')[-1]

def equals(x, y):
    return (isinstance(x, type(y)) or isinstance(y, type(x))) and x == y

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
    if isinstance(x, Type) or callable(x): return False
    try:
        class tmp(x): pass
        return True
    except TypeError: return False

def iterable(x):
    """
    iterable(x) -> bool

    Returns whether an instance can be iterated. Strings are excluded. 

    Args:
        x (any): the input variable.

    Example::

        >>> iterable(x for x in range(4))
        True
        >>> iterable({2, 3})
        True
        >>> iterable("12")
        False
    """
    if isinstance(x, str): return False
    if isinstance(x, type): return False
    if callable(x): return False
    return hasattr(x, '__iter__') and hasattr(x, '__len__')

def isitertype(x):
    """
    isitertype(x) -> bool

    Returns whether a type can be iterated. Str type is excluded. 

    Args:
        x (type): the input variable.

    Example::

        >>> isitertype(list)
        True
        >>> isitertype(str)
        False
    """
    if isinstance(x, str):
        local_vars = get_environ_vars()
        local_vars.update(locals())
        locals().update(local_vars.simplify())
        try: x = eval(x)
        except: return False
    if isinstance(x, Type):
        if x.itemtypes is not None: return True
        return all([isitertype(i) for i in x.types])
    if isinstance(x, type):
        return hasattr(x, '__iter__') and hasattr(x, '__len__')
    if callable(x):
        try: return equals(x('iterable check'), '')
        except Exception: return False
    raise SystemError("Conflict between functions 'isatype' and 'isitertype', please contact the developer for more information. ")

def isofsubclass(x, px):
    """
    isofsubclass(x, px) -> bool

    Returns whether a type 'x' is a subclass of type 'px'.

    Args:
        x (type): the input type.
        px (type): the parent type to check.
    """
    try: return issubclass(x, px)
    except: return False

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
    if equals(x, 'iterable check'): return ''
    if not isinstance(x, type) and hasattr(x, 'shape'): return True
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
    if isinstance(x, str): return True
    if callable(x):
        name = _get_func_name(x)
        if not name: return False
        name = name.split('.')[-1]
        if name.startswith('is') or name.endswith('able'): return True
    if iterable(x):
        if len(x) <= 0: return False
        for xt in x:
            if isatype(xt): continue
            break
        else: return True
    return False

def typename(x):
    """
    typename(x) -> str

    Returns the name of a type. 

    Args:
        x (any): the input variable.
    """
    name = ''
    if callable(x): name = _get_func_name(x)
    if name: return name
    try: name = _rawname(x).capitalize()
    except: pass
    return name

def isoftype(x, xtype):
    """
    isoftype(x, xtype) -> bool

    Returns whether an instance 'x' is of type 'xtype'. 
    
    Note:
        'xtype' can be provided in one of the following fashions:
        1. a pyctlib.Type like Int, Dict[2]@{int: str} or List<<Int>>[]
        2. a str representing a type, either the full name of the required 
            type or a name that can be computed when the package is used.
        3. None representing any type
        4. a list or iterable set of types like [int, float]

    Args:
        x (any): the input variable.
        xtype (type): the type to check.

    Example::
        >>> isoftype(1, None)
        True
        >>> isoftype(1., [int, 'np.ndarray'])
        False
    """
    if isinstance(xtype, str):
        local_vars = get_environ_vars()
        local_vars.update(locals())
        locals().update(local_vars.simplify())
        try: xtype = eval(xtype)
        except: return xtype in [_rawname(t) for t in type(x).__mro__]
    if xtype == type: return isatype(x)
    if xtype is None: return True
    if not iterable(xtype):
        if isinstance(xtype, Type):
            if xtype.isunion and type(x) in xtype.types: return True
            return xtype(x) not in (None, False)
        if isinstance(xtype, type): return isinstance(x, xtype)
        if callable(xtype):
            name = _get_func_name(xtype)
            if not name: return False
            name = name.split('.')[-1]
            if name.startswith('is') or name.endswith('able'):
                try: return xtype(x) not in (None, False)
                except: pass
            return False
    if type(x) in xtype: return True
    for xt in xtype:
        if isoftype(x, xt): return True
    return False

def isclassmethod(x):
    if not callable(x): return False
    if hasattr(x, '__qualname__'): return False
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
    if have_inspect and '__file__' in func.__globals__:
        try:
            dec = next(l for l in inspect.getsource(func).split('\n') if l.strip() and not l.strip().startswith('@')).strip().rstrip(':')
            return dec[dec.index('('):]
        except OSError: pass
    ss = _strIO()
    oldout = sys.stdout
    sys.stdout = ss
    help(func)
    sys.stdout = oldout
    dec = [l for l in ss.split('\n') if len(l) > 0 and 'Help' not in l][0].replace("def ", '').strip().strip(':')
    return dec[dec.index('('):]

def _get_func_name(f):
    try: rawname = f.__name__
    except: rawname = ''
    if not rawname:
        try: rawname = f.__class__.__name__
        except: rawname = ''
    if not rawname: return None
    fname = rawname.split('[')[0]
    if fname.endswith('__0__') or fname.endswith('__default__'):
        fname = '__'.join(fname.split('__')[:-2])
        rawname = fname + '['.join(rawname.split('[')[1:])
    return fname

@decorator
def list_type_args(func):
    '''
    Expand the args to arguments if they are in a list.
    '''
    def wrapper(*args, **kwargs):
        if isclassmethod(func):
            pre = args[:1]
            args = args[1:]
        else: pre = tuple()
        if len(args) == 1 and type(args[0]) in (list, tuple):
            for item in args[0]:
                if not isatype(item):
                    raise TypeHintError("Unsupported types of arguments for " +
                                        _get_func_name(func) + "(): " + str(item) + '. ')
                    break
            else: args = args[0]
        return func(*pre, *args, **kwargs)
    return wrapper

class Type(type):

    @staticmethod
    def extractType(array):
        output = []
        for x in array:
            if iterable(x): output.extend(Type.extractType(list(x)))
            elif isinstance(x, Type) and x.isunion:
                output.extend(Type.extractType(x.types))
            elif isinstance(x, type) or callable(x) or isinstance(x, str): output.append(x)
        return output

    def __new__(cls, *_T, name=None, shape=tuple(), inv=False, ext=False, itypes=None):
        if len(_T) == 1 and isinstance(_T[0], Type):
            inv = _T[0].inv
            name = _T[0].name
            shape = _T[0].shape
            ext = _T[0].extendable
            itypes = _T[0].itemtypes
            inheritableT = _T[0].inheritableT
            _T = _T[0].types
        else:
            _T = Type.extractType(_T)
            if name == None: name = cls.__name__ + '.' + '/'.join(map(typename, _T))
            inheritableT = tuple(t for t in _T if inheritable(t))

        self = super().__new__(cls, name, inheritableT, {})
        self.inv = inv
        self.types = _T
        self.name = name
        self.shape = shape
        self.extendable = ext
        self.itemtypes = itypes
        self.inheritableT = inheritableT
        return self

    def __init__(*args, **kwargs): pass

    def __getitem__(self, i=None):
        if i == 0 or i is None: i = tuple()
        if isoftype(i, sequence): pass
        elif isinstance(i, slice): return self@{i.start:i.stop}
        elif isinstance(i, int): i = (i,)
        elif isatype(i): return self@i
        else: raise TypeHintError("Wrong size specifier. ")
        if all(x is None or isitertype(x) for x in self.types):
            if not all([isoftype(x, int) for x in i]):
                if isinstance(i, list): return self@i
                if all([isinstance(x, slice) for x in i]): return self@{x.start:x.stop for x in i}
                if all([isinstance(x, str) or isatype(x) for x in i]): return self@Type(*i)
                raise TypeHintError("Wrong size specifier. ")
            T = Type(self)
            T.shape = i
            return T
        else:
            T = Type(self)
            T.types = sequence
            T.name = "Iterable"
            T.shape = i
            T.itemtypes = self
            return T

    def __lshift__(self, i):
        if all([isofsubclass(x, dict) for x in self.types]) and len(i) == 2: return self@{i[0]: i[1]}
        elif all([isitertype(x) for x in self.types]): return self@i
        raise TypeHintError("Only tensors, iterable or dict types can use <> representations.")

    def __rshift__(self, l):
        if not isinstance(l, list): raise TypeHintError("Please use iterable<type>[length] to specify a type.")
        if len(l) == 0: return self[-1]
        if len(l) == 1: return self[l[0]]
        return self[tuple(l)]

    def __or__(self, other): return Type(self, other, name=f"{self.rawname}|{other.rawname if isinstance(other, Type) else _rawname(other)}")
    def __ror__(self, other): return Type(other, self, name=f"{other.rawname if isinstance(other, Type) else _rawname(other)}|{self.rawname}")
    def __and__(self, other): return Type(~Type(~self, ~other), name=f"{self.rawname}&{other.rawname}")
    def __rand__(self, other): return Type(~Type(~other, ~self), name=f"{other.rawname}&{self.rawname}")

    def __matmul__(self, other):
        if other is None: return
        if not all(isitertype(x) for x in self.types): raise TypeError("Only iterable Type can use @ to specify the items. ")
        if isinstance(other, Type) or all(isarray(x) for x in self.types) and isdtype(other):
            T = Type(self)
            T.itemtypes = other
            return T
        shape = self.shape
        if len(shape) == 0 and isoftype(other, (list, tuple)) and len(other) > 0: shape = [len(other)]
        if not iterable(other): other = (other,)
        if len(shape) == 1 and len(other) not in (1, shape[0]): raise TypeHintError("Wrong number of item types for @. ")
        if isinstance(other, dict) and not all([isofsubclass(x, dict) for x in self.types]): raise TypeHintError("Please do use {a:b} typehint for dicts.")
        T = Type(self)
        T.itemtypes = other
        return T

    def __pos__(self):
        T = Type(self)
        T.extendable = True
        return T

    def __invert__(self):
        T = Type(self)
        T.inv = not self.inv
        return T

    def __str__(self):
        string = f"<Type '{'*' * self.extendable}{'non-' * self.inv}{self.rawname}"
        if self.itemtypes != None:
            if isinstance(self.itemtypes, Type):
                string += f"<{self.itemtypes.rawname}>"
            elif isinstance(self.itemtypes, type):
                string += f"<{_rawname(self.itemtypes)}>"
            elif iterable(self.itemtypes):
                string += f"<{'/'.join([_rawname(t).capitalize() for t in self.itemtypes])}>"
            else: string += f"<{self.itemtypes}>"
        if self.shape: string += str(list(self.shape))
        string += "'>"
        return string

    __repr__ = __str__

    @property
    def isunion(self):
        return self.name is None and len(self.shape) == 0 and not self.inv \
            and not self.extendable and self.itemtypes is None

    @property
    def rawname(self):
        if self.name: return self.name
        return '/'.join([typename(t) for t in self.types])

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
        if len(self.shape) == 0: raise TypeHintError(str(self) + " is not iterable. ")

    def __call__(self, arg):
        true = not self.inv
        false = self.inv

        if self.isunion: return isoftype(arg, self.types)
        if isoftype(arg, self.types):
            if len(self.shape) == 0: pass
            elif len(self.shape) == 1 and (len(arg) == self.shape[0] or self.shape[0] in (None, -1)): pass
            elif len(self.shape) > 1 and isarray(arg) and \
                all([a==b or b in (None, -1) for a, b in zip(arg.shape, self.shape)]): pass
            else: return false
            if self.itemtypes != None:
                if not iterable(arg): raise TypeHintError(str(arg) + " is not iterable. ")
                if isinstance(self.itemtypes, dict):
                    if len(self.itemtypes) <= 0:
                        raise TypeHintError("Wrong item types after @, {} is not allowed. ")
                    for k, v in arg.items():
                        for ktype, vtype in self.itemtypes.items():
                            if isoftype(k, ktype) and isoftype(v, vtype): break
                        else: return false
                elif iterable(self.itemtypes):
                    if len(self.itemtypes) == 1: self.itemtypes *= len(arg)
                    for x, xt in zip(arg, self.itemtypes):
                        if not isoftype(x, xt): return false
                elif isarray(arg):
                    dname = lambda dt: re.findall(r'\b\w+\b', repr(dt))[-1]
                    if not iterable(self.itemtypes): itypes = [self.itemtypes]
                    else: itypes = self.itemtypes
                    for dt in Type.extractType(itypes):
                        if isinstance(dt, Type):
                            x = arg[(0,) * len(arg.shape)]
                            return isoftype(x, dt)
                        if isdtype(dt): dt = dname(dt)
                        if isatype(dt): dt = _rawname(dt)
                        if dname(arg.dtype) == dt: return true
                    return false
                elif isatype(self.itemtypes):
                    import random
                    for x in random.sample(list(arg), min(len(arg), 5)):
                        if not isoftype(x, self.itemtypes): return false
                else: raise TypeHintError("Invalid item types after @, please use iterable or types.")
            return true
        return false
    
    def checktype(self, *args):
        raise NotImplementedError("Please implement the check function 'checktype' first.")

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

Class = T(type, name="Class")
Callable = T(callable)
Function = T(type(iterable))
Method = T(type(_strIO().split), type("".split), type(Int.__str__), type(staticmethod(lambda:None)), type(classmethod(lambda:None)), name="Method")
Lambda = T(type(lambda: None), name="Lambda")
Functional = T(Function, Method, Lambda, name="FunctionType")
Real = T(float, int, name="Real")
sequence = [tuple, list]
Sequence = T(tuple, list, name="Sequence")
null = type(None)
Null = T(null)
real = [int, float]

class ArrayType(Type):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.types = [isarray]

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

Array = ArrayType(name="Array")
Iterable = T(Array, Sequence, set, dict, name="Iterable")

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
    if ekwargs: allargs[ekwargs] = inputargs[0] if len(inputargs) == 1 and isinstance(inputargs[0], dict) else kwargs; kwargs = {}
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
    if len(types) == 1 and len(kwtypes) == 0 and Functional(types[0]): return params()(types[0])
    @decorator
    def induced_decorator(func):
        declaration = _getDeclaration(_get_wrapped(raw_function(func)))
        eargs = ''.join(re.findall(r"[^*]\*{1} *(\w+)\b", declaration))
        ekwargs = ''.join(re.findall(r"[^*]\*{2} *(\w+)\b", declaration))
        depth = []
        d = 0;
        for c in declaration:
            if c in '([{': d += 1; depth.append(d)
            elif c in '}])': depth.append(d); d -= 1
            else: depth.append(d)
            if d == 0: break
        declaration = ''.join(c for x, c in zip(depth, declaration) if x == 1)
        declaration = declaration.strip('()')
        declaration = ','.join(p.split(':')[0].split('=')[0].strip() + ("='__default__'" if re.search('[^=]=[^=]', p) else '') for p in declaration.split(','))
        exec(f"def tmp({declaration}): return locals()")
        fetch = eval('tmp')
        if len(types) == len(kwtypes) == 0:
            annotations = func.__annotations__
            if 'return' in annotations:
                rtype = annotations.pop('return')
            else: rtype = None
        else:
            if '__return__' in kwtypes:
                rtype = kwtypes.pop('__return__')
            else: rtype = None
            annotations = fetch(*types, **kwtypes)
        def wrapper_func(*args, **kwargs):
            try: values = fetch(*args, **kwargs)
            except Exception as e: raise TypeHintError(str(e).replace('tmp()', f'{_get_func_name(func)}()'))
            for arg in values:
                if equals(values[arg], "__default__") or arg not in annotations: continue
                if arg == eargs:
                    if Sequence(annotations[arg]) and len(annotations[arg]) == len(values[arg]):
                        if not all(isoftype(v, t) for v, t in zip(values[arg], annotations[arg])): break
                    if any(not isoftype(x, annotations[arg]) for x in values[arg]): break
                elif arg == ekwargs:
                    if Dict(annotations[args]):
                        if not all(isoftype(values[arg][k], annotations[arg][k]) for k in annotations[args]): break
                    if any(not isoftype(x, annotations[arg]) for x in values[arg].values()): break
                elif not isoftype(values[arg], annotations[arg]): break
            else:
                if run:
                    r = func(*args, **kwargs)
                    if isoftype(r, rtype): return r
                    else: raise TypeHintError(f"{_get_func_name(func)}() has return value of wrong type. Expect type {repr(rtype)} but got {repr(r)}.")
                else: return None
            raise TypeHintError(f"{_get_func_name(func)}() has argument {arg} of wrong type. Expect type {repr(annotations[arg])} but got {repr(values[arg])}.")
        return wrapper_func
    return induced_decorator
        

@decorator
def params_old(*types, run=True, **kwtypes):
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
                        if not _annotations[ekwargs].extendable:
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
                            if not _annotations[eargs][0].extendable:
                                print("Warning: auto extended non-extendable type. Please check your typehint. ")
                            _annotations[eargs] = _annotations[eargs][0]
                    if eargs in _annotations:
                        _annotations[eargs] = Type(list, tuple)@_annotations[eargs]
            for arg in _values:
                if arg in _annotations:
                    if isoftype(_values[arg], _annotations[arg]): continue
                    break
            else:
                if run:
                    retval = func(*args, **kwargs)
                    if 'return' not in _annotations or isoftype(retval, _annotations['return']): return retval
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

    @params(Functional, Int, +Int, __return__ = Real[2])
    def test_func(a, b=2, *k):
        return k

    print(isclassmethod(inheritable))
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
    print(test_func(lambda x: 1, 3, 4, 5))
