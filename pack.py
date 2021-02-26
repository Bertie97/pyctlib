#! python3 -u
#  -*- coding: utf-8 -*-

##############################
## Project: PyCTLib
## File: Package packer
##############################

import sys, os, re
from pyoverload import *
from functools import wraps
from types import GeneratorType

assert len(sys.argv) > 1
package_names = sys.argv[1:]

if 'all' in package_names:
    package_names = []
    for path in os.listdir('.'):
        if path.startswith('setup_') and path.endswith('.py'):
            package_names.append(path[6:-3])

def main():
    for p in package_names:
        if '==' in p: package_name, version = p.split('==')
        else: package_name = p; version = None
        version_match = None
        with open(f"setup_{package_name}.py") as fp:
            file_str = fp.read()
            lines = []
            for line in file_str.split('\n'):
                if line.strip().startswith('#'): continue
                for match in re.findall(r'open\(.+\).read\(\)', line):
                    line = line.replace(match, '"""' + eval(match) + '"""')
                for match in re.findall(r'version *= *"[\d.]+"', line):
                    v = match.split('"')[1]
                    if version is None: version = '.'.join(v.split('.')[:-1] + [str(eval(v.split('.')[-1]) + 1)])
                    version = f'version = "{version}"'
                    line = line.replace(match, version)
                    version_match = match
                lines.append(line)
            with open("setup.py", 'w') as outfp:
                outfp.write('\n'.join(lines))

        with open(f"setup_{package_name}.py", 'w') as fp:
            if version_match is not None: fp.write(file_str.replace(version_match, version))

        ppath = path('.')/"packing_package"
        if ppath.exists(): os.system(f"rm -r {ppath}")
        os.system(f"cp setup.py {ppath.mkdir()/'setup.py'}")
        os.system(f"cp -r {package_name} packing_package/{package_name}")
        os.system(f"cd packing_package; python3 setup.py sdist --dist-dir dist_{package_name}")
        os.system(f"cd packing_package; twine upload {str(vector(ppath/('dist_' + package_name)).max()[-2:])}")
        os.system("rm setup.py")
        os.system("rm -r packing_package")

def totuple(num):
    if isinstance(num, str): return (num,)
    try: return tuple(num)
    except: return (num,)

def touch(f: Callable):
    try: return f()
    except: return None

def raw_function(func):
    if "__func__" in dir(func):
        return func.__func__
    return func

class _Vector_Dict(dict):

    def values(self):
        return vector(super().values())

    def keys(self):
        return vector(super().keys())

class vector(list):

    def __init__(self, *args):
        if len(args) == 0:
            list.__init__(self)
        elif len(args) == 1:
            list.__init__(self, args[0])
        else:
            list.__init__(self, args)

    def filter(self, func=None):
        if func is None:
            return self
        try:
            return vector([a for a in self if func(a)])
        except:
            pass
        for index, a in enumerate(self):
            if touch(lambda: func(a)) is None:
                raise RuntimeError("Exception raised in filter function at location {} for element {}".format(index, a))

    def test(self, func):
        return vector([a for a in self if touch(lambda: func(a))])

    def testnot(self, func):
        return vector([a for a in self if not touch(lambda: func(a))])

    def map(self, func=None):
        """
        generate a new vector with each element x are replaced with func(x)
        """
        if func is None:
            return self
        try:
            return vector([func(a) for a in self])
        except:
            pass
        for index, a in enumerate(self):
            if touch(lambda: func(a)) is None:
                raise RuntimeError("Exception raised in map function at location {} for element {}".format(index, a))

    def apply(self, func) -> None:
        for x in self:
            func(x)

    def check_type(self, instance):
        return all(self.map(lambda x: isinstance(x, instance)))

    @override
    def __mul__(self, other):
        if touch(lambda: self.check_type(tuple) and other.check_type(tuple)):
            return vector(zip(self, other)).map(lambda x: (*x[0], *x[1]))
        elif touch(lambda: self.check_type(tuple)):
            return vector(zip(self, other)).map(lambda x: (*x[0], x[1]))
        elif touch(lambda: other.check_type(tuple)):
            return vector(zip(self, other)).map(lambda x: (x[0], *x[1]))
        else:
            return vector(zip(self, other))

    @__mul__
    def _(self, times: int):
        return vector(super().__mul__(times))

    def __pow__(self, other):
        return vector([(i, j) for i in self for j in other])

    def __add__(self, other: list):
        return vector(super().__add__(other))

    def _transform(self, element, func=None):
        if not func:
            return element
        return func(element)

    @override
    def __eq__(self, element):
        return self.map(lambda x: x == element)

    @__eq__
    def _(self, other: list):
        return vector(zip(self, other)).map(lambda x: x[0] == x[1])

    @override
    def __neq__(self, element):
        return self.map(lambda x: x != element)

    @__neq__
    def _(self, other: list):
        return vector(zip(self, other)).map(lambda x: x[0] != x[1])

    @override
    def __lt__(self, element):
        return self.map(lambda x: x < element)

    @__lt__
    def _(self, other: list):
        return vector(zip(self, other)).map(lambda x: x[0] < x[1])

    @override
    def __gt__(self, element):
        return self.map(lambda x: x > element)

    @__gt__
    def _(self, other: list):
        return vector(zip(self, other)).map(lambda x: x[0] > x[1])

    @override
    def __le__(self, element):
        return self.map(lambda x: x < element)

    @__le__
    def _(self, other: list):
        return vector(zip(self, other)).map(lambda x: x[0] <= x[1])

    @override
    def __ge__(self, element):
        return self.map(lambda x: x >= element)

    @__ge__
    def _(self, other: list):
        return vector(zip(self, other)).map(lambda x: x[0] >= x[1])

    @override
    def __getitem__(self, index):
        if isinstance(index, slice):
            return vector(super().__getitem__(index))
        return super().__getitem__(index)

    @__getitem__
    def _(self, index_list: list):
        assert len(self) == len(index_list)
        return vector(zip(self, index_list)).filter(lambda x: x[1]).map(lambda x: x[0])

    @overload
    def __sub__(self, other: Iterable):
        try:
            other = set(other)
        except:
            other = list(other)
        finally:
            return self.filter(lambda x: x not in other)

    @overload
    def __sub__(self, other):
        return self.filter(lambda x: x != other)

    def __setitem__(self, i, t):
        if isinstance(i, int):
            super().__setitem__(i, t)
        elif isinstance(i, slice):
            super().__setitem__(i, t)
        elif isinstance(i, list):
            if all([isinstance(index, bool) for index in i]):
                if iterable(t):
                    p_index = 0
                    for value in t:
                        while i[p_index] == False:
                            p_index += 1
                        super().__setitem__(p_index, value)
                else:
                    for index in range(len(self)):
                        if i[index] == True:
                            super().__setitem__(index, t)
            elif all([isinstance(index, int) for index in i]):
                if iterable(t):
                    p_index = 0
                    for p_index, value in enumerate(t):
                        super().__setitem__(i[p_index], value)
                else:
                    for index in i:
                        super().__setitem__(index, t)
            else:
                raise TypeError("only support the following usages: \n [int] = \n [slice] = \n [list] = ")
        else:
            raise TypeError("only support the following usages: \n [int] = \n [slice] = \n [list] = ")


    def _hashable(self):
        return all(self.filter(lambda x: "__hash__" in x.__dir__()))

    def __hash__(self):
        if not self._hashable():
            raise Exception("not all elements in the vector is hashable, the index of first unhashable element is %d" % self.index(lambda x: "__hash__" not in x.__dir__()))
        else:
            return hash(tuple(self))

    def unique(self):
        if len(self) == 0:
            return vector([])
        hashable = self._hashable()
        explored = set() if hashable else list()
        pushfunc = explored.add if hashable else explored.append
        unique_elements = list()
        for x in self:
            if x not in explored:
                unique_elements.append(x)
                pushfunc(x)
        return vector(unique_elements)

    def count(self, *args):
        if len(args) == 0:
            return len(self)
        return super().count(args[0])

    # @overload
    # def index(self, element: int):
    #     return super().index(element)

    # @overload
    def index(self, element):
        if isinstance(element, int):
            return super().index(element)
        elif callable(element):
            for index in range(len(self)):
                if element(self[index]):
                    return index
            return -1
        else:
            raise RuntimeError("error input for index")

    def all(self, func=lambda x: x):
        for t in self:
            if not func(t):
                return False
        return True

    def any(self, func=lambda x: x):
        for t in self:
            if func(t):
                return True
        return False

    def max(self, key=None, with_index=False):
        if len(self) == 0:
            return None
        m_index = 0
        m_key = self._transform(self[0], key)
        for index in range(1, len(self)):
            i_key = self._transform(self[index], key)
            if i_key > m_key:
                m_key = i_key
                m_index = index
        if with_index:
            return self[m_index], m_index
        return self[m_index]


    def min(self, key=None, with_index=False):
        if len(self) == 0:
            return None
        m_index = 0
        m_key = self._transform(self[0], key)
        for index in range(1, len(self)):
            i_key = self._transform(self[index], key)
            if i_key < m_key:
                m_key = i_key
                m_index = index
        if with_index:
            return self[m_index], m_index
        return self[m_index]

    def sum(self):
        return self.reduce(lambda x, y: x + y)

    def group_by(self, key=lambda x: x[0]):
        result = _Vector_Dict()
        for x in self:
            k_x = key(x)
            if k_x not in result:
                result[k_x] = vector([x])
            else:
                result[k_x].append(x)
        return result

    def reduce(self, func):
        if len(self) == 0:
            return None
        temp = self[0]
        for x in self[1:]:
            temp = func(temp, x)
        return temp

    def flatten(self):
        return self.reduce(lambda x, y: x + y)

    def generator(self):
        return ctgenerator(self)

def generator_wrapper(*args, **kwargs):
    if len(args) == 1 and callable(raw_function(args[0])):
        func = raw_function(args[0])
        @wraps(func)
        def wrapper(*args, **kwargs):
            return ctgenerator(func(*args, **kwargs))
        return wrapper
    else:
        raise TypeError("function is not callable")

class ctgenerator:

    @staticmethod
    def _generate(iterable):
        for x in iterable:
            yield x

    @override
    def __init__(self, generator):
        if "__iter__" in generator.__dir__():
            self.generator = ctgenerator._generate(generator)

    @__init__
    def _(self, generator: "ctgenerator"):
        self.generator = generator

    @__init__
    def _(self, generator: GeneratorType):
        self.generator = generator

    @generator_wrapper
    def map(self, func) -> "ctgenerator":
        for x in self.generator:
            yield func(x)

    @generator_wrapper
    def filter(self, func=None) -> "ctgenerator":
        for x in self.generator:
            if func(x):
                yield x

    def reduce(self, func, initial_value=None):
        if not initial_value:
            initial_value = next(self.generator)
        result = initial_value
        for x in self.generator:
            result = func(initial_value, x)
        return result

    def apply(self, func) -> None:
        for x in self.generator:
            func(x)

    def __iter__(self):
        for x in self.generator:
            yield x

    def __next__(self):
        return next(self.generator)

    def vector(self):
        return vector(self)

class path(str):

    sep = os.path.sep #/
    extsep = os.path.extsep #.
    pathsep = os.path.pathsep #:
    namesep = '_'
    File = b'\x04'
    Folder = b'\x07'
    homedir = os.path.expanduser("~")

    class pathList(vector):

        def __new__(cls, lst, main_folder = os.curdir):
            self = super().__new__(cls)
            for e in lst:
                if e not in self: self.append(e)
            self.main_folder = main_folder
            return self

        def __init__(self, *args, **kwargs): pass

        def __or__(self, k): return self[[x|k for x in self]]
        def __sub__(self, y): return path.pathList([x - y for x in self])
        def __neg__(self): return self - self.main_folder
        def __matmul__(self, k): return path.pathList([x @ k for x in self])
        def __mod__(self, k): return path.pathList([x % k for x in self])
        def __getitem__(self, i):
            if callable(i): return self[[i(x) for x in self]]
            if isinstance(i, list) and len(i) == len(self): return path.pathList([x for x, b in zip(self, i) if b])
            return super().__getitem__(i)

    @generator_wrapper
    @staticmethod
    def rlistdir(folder, tofolder=False, relative=False, ext='', filter=lambda x: True):
        folder = path(folder)
        # file_list = []
        for f in os.listdir(str(folder)):
            if f == '.DS_Store': continue
            p = folder / f
            if p.isdir():
                # file_list.extend(path.rlistdir(p, tofolder))
                for cp in path.rlistdir(p, tofolder, relative=relative, ext=ext, filter=filter):
                    if filter(cp) and (cp | ext):
                        yield cp
            if p.isfile() and not tofolder and filter(p) and (p | ext):
                yield p
        if tofolder and not file_list and filter(folder) and (folder | ext):
            # file_list.append(folder)
            yield folder
        # file_list = path.pathList(file_list, main_folder=folder)
        # if relative: file_list = -file_list
        # if ext: file_list = file_list[file_list|ext]
        # return file_list[filter]

    def __new__(cls, *init_texts):
        if len(init_texts) <= 0 or len(init_texts[0]) <= 0:
            self = super().__new__(cls, "")
        elif len(init_texts) == 1 and init_texts[0] == "~":
            self = super().__new__(cls, path.homedir)
        else:
            self = super().__new__(cls, os.path.join(*[str(x).replace('$', '') for x in init_texts]).strip())
        self.init()
        return self

    def init(self): pass
    def __and__(x, y): return path(path.pathsep.join((str(x).rstrip(path.pathsep), str(y).lstrip(path.pathsep))))
    def __mul__(x, y): return path(x).mkdir(y)
    def __mod__(x, y): return path(str(x) % totuple(y))
    def __sub__(x, y): return path(os.path.relpath(str(x), str(y)))
    def __add__(x, y):
        y = str(y)
        if x.isfilepath():
            file_name = x@path.File
            folder_name = x@path.Folder
            parts = file_name.split(path.extsep)
            if parts[-1].lower() in ('zip', 'gz', 'rar') and len(parts) > 2: brk = -2
            else: brk = -1
            ext = path.extsep.join(parts[brk:])
            name = path.extsep.join(parts[:brk])
            return folder_name/(name + y + path.extsep + ext)
        else: return path(str(x) + y)
    def __xor__(x, y):
        y = str(y)
        if x.isfilepath():
            file_name = x@path.File
            folder_name = x@path.Folder
            parts = file_name.split(path.extsep)
            if parts[-1].lower() in ('zip', 'gz', 'rar') and len(parts) > 2: brk = -2
            else: brk = -1
            ext = path.extsep.join(parts[brk:])
            name = path.extsep.join(parts[:brk])
            return folder_name/(name.rstrip(path.namesep) + path.namesep + y.lstrip(path.namesep) + path.extsep + ext)
        else: return path(path.namesep.join((str(x).rstrip(path.namesep), y.lstrip(path.namesep))))
    def __pow__(x, y):
        output = rootdir
        for p, q in zip((~path(x)).split(), (~path(y)).split()):
            if p == q: output /= p
            else: break
        return output - curdir
    def __floordiv__(x, y): return path(path.extsep.join((str(x).rstrip(path.extsep), str(y).lstrip(path.extsep))))
    def __invert__(self): return path(os.path.abspath(str(self)))
    def __abs__(self): return path(os.path.abspath(str(self)))
    def __truediv__(x, y): return path(os.path.join(str(x), str(y)))
    def __or__(x, y):
        if y == "": return True
        if y == path.File: return x.isfile()
        if y == path.Folder: return x.isdir()
        if isinstance(y, int): return len(x) == y
        if '.' not in y: y = '.*\\' + x.extsep + y
        return re.fullmatch(y.lower(), x[-1].lower()) is not None
        # return x.lower().endswith(x.extsep + y.lower())
    def __eq__(x, y): return str(x) == str(y)
    def __matmul__(self, k):
        if k == path.Folder: return path(self[:-1])
        elif k == path.File: return path(self[-1:])
        return
    def __lshift__(self, k): return path.rlistdir(self, k == path.Folder, ext=k if k not in (path.File, path.Folder) else '')
    def __setitem__(self, i, v):
        lst = self.split()
        lst[i] = v
        return path(lst)
    def __getitem__(self, i):
        res = self.split()[i]
        return res if isinstance(res, str) else path(path.sep.join(res))
    def __len__(self): return len(self.split())

    @generator_wrapper
    def __iter__(self):
        for p in self<<path.File:
            yield p
    def __contains__(self, x): return x in str(self)

    @property
    def ext(self):
        file_name = self@path.File
        parts = file_name.split(path.extsep)
        if parts[-1].lower() in ('zip', 'gz', 'rar') and len(parts) > 2: brk = -2
        elif len(parts) > 1: brk = -1
        else: brk = 1
        return path.extsep.join(parts[brk:])
    @property
    def name(self):
        file_name = self@path.File
        parts = file_name.split(path.extsep)
        if parts[-1].lower() in ('zip', 'gz', 'rar') and len(parts) > 2: brk = -2
        elif len(parts) > 1: brk = -1
        else: brk = 1
        return path.extsep.join(parts[:brk])
    def split(self, *args):
        if len(args) == 0: return [path(x) if x else path("$") for x in str(self).split(path.sep)]
        else: return str(self).split(*args)
    def abs(self): return path(os.path.abspath(self))
    def listdir(self, recursively=False):
        return self << path.File if recursively else path.pathList([self / x for x in os.listdir(str(self))])
    # changed by zhangyiteng
    def ls(self, func=None):
        return self.listdir().filter(func)
    def cd(self, folder_name):
        folder_name = path(folder_name)
        if folder_name.isabs():
            return folder_name
        new_folder = self / folder_name
        if new_folder.isdir():
            if self.isabs():
                return new_folder.abs()
            return new_folder
        elif (new_folder @ path.Folder).isdir():
            raise NotADirectoryError("%s doesn't exist, all available folder is: %s" % (new_folder, (new_folder @ path.Folder).ls().filter(lambda x: x.isdir()).map(lambda x: x.name)))
        else:
            raise NotADirectoryError("%s doesn't exist" % new_folder)
    def cmd(self, command):
        try:
            if self.isdir():
                os.system("cd %s; %s" % (self, command))
            elif self.isfile():
                if "{}" in command:
                    self.parent.cmd(command.format(self))
                else:
                    self.parent.cmd(command + " " + self)
        except Exception as e:
            print("cmd error:", e)
    def open(self):
        if self.isdir():
            self.cmd("open .")
        elif self.isfile():
            self.parent.cmd("open %s" % self)
    @property
    def parent(self):
        return self @ path.Folder
    # end changed by zhangyiteng
    def isabs(self): return os.path.isabs(self)
    def exists(self): return os.path.exists(self)
    def isfile(self): return os.path.isfile(self)
    def isdir(self): return os.path.isdir(self)
    def isfilepath(self): return True if os.path.isfile(self) else 0 < len(self.ext) < 7
    def isdirpath(self): return True if os.path.isdir(self) else (len(self.ext) == 0 or len(self.ext) >= 7)
    def mkdir(self, to: str='Auto'):
        cumpath = path(os.path.curdir)
        if self.isabs(): cumpath = cumpath.abs()
        fp = self - cumpath
        if to == path.Folder: fp = fp@path.Folder
        elif to == path.File: pass
        elif self.isfilepath(): fp = fp@path.Folder
        for p in fp.split():
            cumpath /= p
            if not cumpath.exists(): os.mkdir(cumpath)
        return self

if __name__ == "__main__": main()
