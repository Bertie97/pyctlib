#! python3 -u
#  -*- coding: utf-8 -*-

import re, sys
from functools import wraps

class TypeHintError(Exception): pass

def iterable(x):
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
    if isatype(x) or not iterable(x): return False
    if isoftype(x, [list, dict, tuple, set, str]): return False
    if 'shape' in x.__dict__: return True
    raise TypeError("Currently unrecognized array type, please contact the developers.")

def isdtype(x):
    return 'dtype' in repr(type(x)).lower()

def isatype(x):
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

def isoftype(x, xtype):
    if isinstance(xtype, str):
        try: xtype = eval(xtype)
        except: return False
    if xtype == type: return isatype(x)
    if xtype is None: return True
    if isinstance(xtype, type):
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
    p = x.__qualname__.split('.')
    if len(p) <= 1: return False
    try:
        class_name = '.'.join(p[:-1])
        if isinstance(x.__globals__.get(class_name, int), type): return True
        else: return False
    except: return False

def listargs(*_T):
    '''
    Expand the args to arguments if they are in a list.
    '''
    if len(_T) == 1 and iterable(_T[0]): _T = _T[0]
    def wrap(func):
        @wraps(func)
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
                                                func.__name__ + "(): " + str(item) + '. ')
                        break
                else: args = args[0]
            return func(*pre, *args, **kwargs)
        return wrapper
    return wrap

class Type:

    @staticmethod
    def extractType(array):
        output = []
        for x in array:
            if isinstance(x, Type):
                output.extend(Type.extractType(x.types))
            else: output.append(x)
        return output

    @listargs(type)
    def __init__(self, *_T, shape=tuple(), ext=False, itypes=None):
        if len([0 for t in _T if not isatype(t)]) > 0 or len(_T) <= 0:
            raise SyntaxError("Wrong parameter type. ")
        self.inv = False
        self.types = Type.extractType(_T)
        self.shape = shape
        self.extendable = ext
        self.itemtypes = itypes
        if len(_T) == 1 and type(_T[0]) == Type: self.copyfrom(_T[0])

    @listargs(type)
    def strT(self, *_T):
        mid = lambda x: x[1] if len(x) > 1 else x[0]
        rawname = lambda s: mid(str(s).split("'"))
        if isoftype(_T, (list, tuple, set)) and len(_T) > 1:
            return '(' + ', '.join([self.strT(t) for t in _T]) + ')'
        if iterable(_T) and len(_T) == 1 and isoftype(_T[0], dict): _T = _T[0]
        if isoftype(_T, dict):
            return '{' + ', '.join([':'.join((self.strT(k), self.strT(v))) for k, v in _T.items()]) + '}'
        return rawname(_T[0])

    def isextendable(self): return self.extendable

    def copyfrom(self, other):
        self.types = other.types
        if len(other.shape) > 0: self = self[other.shape]
        if other.extendable: self = +self
        self @= other.itemtypes

    def __getitem__(self, i=None):
        if i == 0 or i is None: i = tuple()
        if isatype(i): return self@i
        if isoftype(i, int): i = (i,)
        elif isoftype(i, [list, tuple]): pass
        else: raise TypeHintError("Wrong size specifier. ")
        if all([x is None or iterable(x) for x in self.types]):
            return Type(self.types, shape=i, ext=self.extendable, itypes=self.itemtypes)
        else: return Type([list, tuple, dict, set], shape=i, ext=self.extendable, itypes=self)
        # else: raise TypeHintError(("Type " if len(self.types) == 1 else "Types ") + self.strT(self.types) +
                            #   (" is" if len(self.types) == 1 else " are") + " not iterable. ")

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
        if isinstance(other, Type):
            if iterable(other): return Type(self.types, shape=other.shape, ext=self.extendable, itypes=other.itemtypes)
            else: return Type(self.types, shape=other.shape, ext=self.extendable, itypes=other)
        if all([isarray(x) for x in self.types]) and isdtype(other):
            return Type(self.types, shape=self.shape, ext=self.extendable, itypes=other)
        shape = self.shape
        if len(shape) == 0 and isoftype(other, (list, tuple)) and len(other) > 0: shape = [len(other)]
        if not iterable(other): other = (other,)
        if len(shape) == 1 and len(other) not in (1, shape[0]): raise TypeHintError("Wrong number of item types for @. ")
        if isinstance(other, dict) and not all([issubclass(x, dict) for x in self.types]): raise TypeHintError("Please do use {a:b} typehint for dicts.")
        return Type(self.types, shape=shape, ext=self.extendable, itypes=other)

    def __pos__(self):
        return Type(self.types, shape=self.shape, ext=True, itypes=self.itemtypes)

    def __invert__(self):
        self.inv = True
        return self

    def __str__(self):
        string = '<'
        if len(self.shape) > 0:
            string += "extendable " if self.extendable else ''
            string += self.strT(self.types)
            string += str(list(self.shape))
            if self.itemtypes != None: string += " of types " + self.strT(self.itemtypes)
        else:
            string += "Extendable t" if self.extendable else 'T'
            string += "ype " + ('' if len(self.types) == 1 else "in ") + self.strT(self.types)
            if self.itemtypes != None: string += " of type " + self.strT(self.itemtypes)
        string += '>'
        return string

    __repr__ = __str__

    def __len__(self):
        if len(self.shape) == 0: return "N"
        prod = 1; unsure = False
        for s in self.shape:
            if s in (None, -1): unsure = True; continue
            prod *= s
        if unsure: return str(prod) + " x N"
        else: return prod

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
                    rawname = lambda s: mid(str(s).split("'"))
                    dname = lambda dt: re.findall(r'\b\w+\b', repr(dt))[-1]
                    if not iterable(self.itemtypes): itypes = [self.itemtypes]
                    else: itypes = self.itemtypes
                    for dt in Type.extractType(itypes):
                        if isdtype(dt): dt = dname(dt)
                        if isatype(dt): dt = rawname(dt)
                        if dname(arg.dtype) == dt: return true
                    return false
                elif isatype(self.itemtypes):
                    for x in arg:
                        if not isoftype(x, self.itemtypes): return false
                else: raise TypeHintError("Invalid item types after @, please use iterable or types.")
            return true
        return false

# def T(*types):
#     all_available_types = [x for x in types if isinstance(x, type)]
#     if len(all_available_types) > 1: all_available_types = all_available_types[:1]
#     try:
#         class defaults(Type, *all_available_types):
#             def __init__(self):
#                 Type.__init__(self, *types)
#                 if all_available_types:
#                     all_available_types
#         return defaults()
#     except TypeError: return Type(*types)

Bool = Type(bool)
class Int(int):
    def __new__(cls): return Type(int) 
class Float(float):
    def __new__(cls): return Type(float)
class Str(str):
    def __new__(cls): return Type(str)
class Set(set):
    def __new__(cls): return Type(set)
class List(list):
    def __new__(cls): return Type(list)
class Dict(dict):
    def __new__(cls): return Type(dict)
class Tuple(tuple):
    def __new__(cls): return Type(tuple)


Callable = callable
Func = Type(type(iterable))
Method = Type(type(Bool.isextendable), type("".split), type(Int.__str__))
Lambda = Type(type(lambda: None))
class Real(float):
    def __new__(cls): return Type(int, float)
class Iterable(tuple):
    def __new__(cls): return Type(list, tuple, dict, set)
null = type(None)
Null = Type(null)
real = [int, float]

def extendable(t): return type(t) == Type and t.isextendable()

class strIO:
    def __init__(self): self._str_ = ''
    def write(self, s): self._str_ += s
    def __str__(self): return self._str_
    def split(self, c=None): return self._str_.split(c)

try:
    def Numpy(x):
        import numpy as np
        return T(np.ndarray)(x)
except ImportError: pass
try:
    def Torch(x):
        import torch
        return T(torch.Tensor)(x)
except ImportError: pass
try:
    def TensorFlow(x):
        ss = strIO()
        olderr = sys.stderr
        sys.stderr = ss
        import tensorflow as tf
        sys.stderr = olderr
        return T(tf.Tensor)(x)
except ImportError: pass

def getDeclaration(func):
    ss = strIO()
    oldout = sys.stdout
    sys.stdout = ss
    help(func)
    sys.stdout = oldout
    return [l for l in ss.split('\n') if len(l) > 0 and 'Help' not in l][0]

def getArgs(func, *args, **kwargs):
    getTypes = False
    allargs = {}; missing = []
    if "__type__" in kwargs: getTypes = kwargs.pop('__type__')
    if "__return__" in kwargs: allargs['return'] = kwargs.pop('__return__')
    inputargs = list(args)
    normalargs = func.__code__.co_varnames[:func.__code__.co_argcount]
    extargs = func.__code__.co_varnames[func.__code__.co_argcount:]
    dec = getDeclaration(func)
    eargs = ''.join(re.findall(r"[^*]\*{1} *(\w+)\b", dec))
    ekwargs = ''.join(re.findall(r"[^*]\*{2} *(\w+)\b", dec))
    lendefault = 0 if not func.__defaults__ else len(func.__defaults__)
    rg = lendefault - func.__code__.co_argcount, lendefault
    for var, idefault in zip(normalargs, range(*rg)):
        if len(inputargs) > 0:
            if var in kwargs:
                raise TypeHintError(func.__name__ + "() got multiple values for \
argument " + repr(var))
            allargs[var] = inputargs.pop(0); continue
        if var in kwargs:
            if getTypes: allargs[var] = type(kwargs.pop(var))
            else: allargs[var] = kwargs.pop(var)
            continue
        if idefault < 0:
            missing.append(var)
            if idefault == lendefault - 1:
                error = func.__name__ + "() missing " + str(-idefault)
                error += " required positional arguments: "
                error += ', '.join([repr(v) for v in missing[:-1]])
                error += ", and " + repr(missing[-1])
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
    if len(kwargs) > 0: raise TypeHintError(func.__name__ + "() got an unexpected " + 
        "keyword argument " + repr(list(kwargs.keys())[0]))
    if len(inputargs) > 0: raise TypeHintError(func.__name__ +
        "() takes from {lower} to {upper} positional arguments but {real} were given)"
        .format(lower=-rg[0], upper=rg[1] - rg[0], real=len(inputargs) + rg[1] - rg[0]))
    return allargs

def params(*types, **kwtypes):
    israw = len(kwtypes) == 0 and len(types) == 1 and \
            callable(types[0]) and not isatype(types[0])
    def wrap(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            dec = getDeclaration(func)
            eargs = ''.join(re.findall(r"[^*]\*{1} *(\w+)\b", dec))
            ekwargs = ''.join(re.findall(r"[^*]\*{2} *(\w+)\b", dec))
            if isclassmethod(func): alltypes = (None,) + types
            else: alltypes = types
            if israw: annotations = func.__annotations__
            else: annotations = getArgs(func, *alltypes, **kwtypes, __type__=True)
            values = getArgs(func, *args, **kwargs)
            if ekwargs:
                assert ekwargs in values
                if ekwargs in annotations:
                    if iterable(annotations[ekwargs]):
                        values.update(values[ekwargs]); values.pop(ekwargs)
                        if len(annotations[ekwargs]) == 0: annotations.pop(ekwargs)
                        else:
                            annotations.update(annotations[ekwargs])
                            annotations.pop(ekwargs)
                    else:
                        if not extendable(annotations[ekwargs]):
                            print("Warning: auto extended non-extendable type. Please check your typehint. ")
                        annotations[ekwargs] = Type(dict)@{str:annotations[ekwargs]}
            if eargs:
                assert eargs in values
                if eargs in annotations:
                    if iterable(annotations[eargs]):
                        if len(annotations[eargs]) == 0: annotations.pop(eargs)
                        if len(annotations[eargs]) > 1:
                            raise TypeHintError(func.__name__ + "() has too many type restraints. ")
                        if not extendable(annotations[eargs][0]):
                            print("Warning: auto extended non-extendable type. Please check your typehint. ")
                        annotations[eargs] = annotations[eargs][0]
                    annotations[eargs] = Type(list, tuple)@annotations[eargs]
            for arg in values:
                if arg in annotations:
                    if isoftype(values[arg], annotations[arg]): continue
                    break
            else:
                retval = func(*args, **kwargs)
                if 'return' not in annotations or isoftype(retval, annotations['return']): return retval
                raise TypeHintError(func.__name__ + "() returns an invalid value.")
            raise TypeHintError(func.__name__ + "() has argument " + arg + " of wrong type. \
Expect type {type} but get {value}.".format(type=repr(annotations[arg]), value=repr(values[arg])))
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
