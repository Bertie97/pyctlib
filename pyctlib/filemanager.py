#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package <main>
##############################

__all__ = """
    path
    pathList
    file
    pwd
    ls
    cp
    get_search_blacklist
    set_search_blacklist
    get_relative_path
""".split()

import os, re, struct, shutil
from .touch import touch, check
from pyoverload import *
from .basicwrapper import raw_function
from functools import wraps, reduce, partial
import typing
from typing import TextIO, Optional
from .vector import NoDefault, UnDefined, OutBoundary, vector, generator_wrapper, ctgenerator, IndexMapping, EmptyClass
from rapidfuzz import fuzz

"""
from pyinout import *
"""

Search_BlackList = vector([".DS_Store", ".git"])

def get_search_blacklist():
    global Search_BlackList
    return Search_BlackList

def set_search_blacklist(blacklist):
    global Search_BlackList
    Search_BlackList = blacklist

def append_search_blacklist(item):
    global Search_BlackList
    Search_BlackList.append(item)

def totuple(num):
    if isinstance(num, str): return (num,)
    try: return tuple(num)
    except: return (num,)

def pwd():
    return path(".").abs()

def ls(folder=None):
    if folder is None:
        return pwd().ls()
    else:
        return folder.ls()

def cp(src, dst):
    assert isinstance(src, path)
    assert isinstance(dst, path)
    assert dst.is_dir()
    shutil.copy2(src, dst)

if os.name == 'nt':
    import win32api, win32con
def is_hidden(p):
    if os.name== 'nt':
        attribute = win32api.GetFileAttributes(p)
        return attribute & (win32con.FILE_ATTRIBUTE_HIDDEN | win32con.FILE_ATTRIBUTE_SYSTEM)
    else:
        return p.startswith('.') #linux-osx

def filepath_generator_wrapper(*args, **kwargs):
    if len(args) == 1 and callable(raw_function(args[0])):
        func = raw_function(args[0])
        @wraps(func)
        def wrapper(*args, **kwargs):
            ret = func(*args, **kwargs)
            main_folder = None
            if len(args) > 0 and isinstance(args[0], filepath_generator):
                main_folder = args[0]._main_folder
            if len(args) > 0 and isinstance(args[0], pathList):
                main_folder = args[0]._main_folder
            return filepath_generator(ret, main_folder=main_folder)
        return wrapper
    else:
        raise TypeError("function is not callable")

def get_relative_path(p):
    if p._main_folder is None:
        return p
    return str(p - p._main_folder)

def display_relative_path(p):
    if not p.main_folder:
        return str(p)
    else:
        return str(p.main_folder).rstrip(path.sep) + "/<{}>".format(get_relative_path(p))

def get_main_folder(p, query, selected):
    return "main folder: " + str(p.main_folder)

class pathList(vector):

    def __init__(self, *args, index_mapping=IndexMapping(), main_folder = None):
        super().__init__(*args, index_mapping=index_mapping, content_type=path)
        self.main_folder = main_folder

    def __or__(self, k): return self[[x|k for x in self]]
    def __sub__(self, y): return pathList([x - y for x in self])
    def __neg__(self): return self - self.main_folder
    def __matmul__(self, k): return pathList([x @ k for x in self])
    def __mod__(self, k): return pathList([x % k for x in self])
    def __getitem__(self, i):
        ret = super().__getitem__(i)
        if isinstance(ret, vector):
            ret = pathList(ret, main_folder=self._main_folder)
        elif isinstance(ret, path):
            ret.main_folder = self._main_folder
        else:
            raise RuntimeError("wired item in pathList: {}".format(ret))
        return ret
    def append(self, element):
        if element.main_folder == self.main_folder:
            element.main_folder = None
        super().append(element)
        return self

    @property
    def main_folder(self):
        if self._main_folder is None:
            return None
        else:
            return self._main_folder.abs()

    @main_folder.setter
    def main_folder(self, mf):
        self._main_folder = mf

    def map(self, *args, **kwargs):
        ret = super().map(*args, **kwargs)
        if ret.check_type(path):
            return pathList(ret, main_folder=self._main_folder)
        else:
            return ret

    def assign_mainfolder(self, main_folder=UnDefined):
        if isinstance(main_folder, EmptyClass):
            main_folder = self._main_folder
        ret = self.map(lambda x: x.assign_mainfolder(main_folder))
        return ret

    def assign_mainfolder_(self, main_folder=UnDefined):
        if isinstance(main_folder, EmptyClass):
            main_folder = self._main_folder
        self.map_(lambda x: x.assign_mainfolder(main_folder))

    def regex_search(self, query="", max_k=NoDefault, str_func=get_relative_path, str_display=get_relative_path, display_info=get_main_folder):

        def regex_function(candidate, query):
            if len(query) == 0:
                return candidate
            regex = re.compile(query)
            selected = candidate.filter(lambda x: regex.search(x), ignore_error=False).sort(len)
            return selected

        return self.function_search(regex_function, query=query, max_k=max_k, str_func=str_func, str_display=str_display, display_info=display_info)

    def fuzzy_search(self, query="", max_k=NoDefault, str_func=get_relative_path, str_display=get_relative_path, display_info=get_main_folder):

        def fuzzy_function(candidate, query):
            if len(query) == 0:
                return candidate
            partial_ratio = candidate.map(lambda x: (fuzz.partial_ratio(x.lower(), query.lower()), x))
            selected = partial_ratio.filter(lambda x: x[0] > 50)
            score = selected.map(lambda x: x[0] * min(1, len(x[1]) / len(query)) * min(1, len(query) / len(x[1])) ** 0.3, lambda x: round(x * 10) / 10).sort(lambda x: -x)
            return score

        return self.function_search(fuzzy_function, query=query, max_k=max_k, str_func=str_func, display_info=display_info, str_display=str_display)

    def filter(self, func=None, ignore_error=True) -> "pathList":
        if func is None:
            return self
        if isinstance(func, str):
            func = lambda x: x | func
        if isinstance(func, bytes):
            func = lambda x: x | func
        return pathList(super().filter(func, ignore_error=ignore_error), main_folder=self._main_folder)

class path(str):

    sep = os.path.sep #/
    extsep = os.path.extsep #.
    pathsep = os.path.pathsep #:
    namesep = '_'
    File = b'\x04'
    Folder = b'\x07'
    homedir = os.path.expanduser("~")

    # @filepath_generator_wrapper
    # @staticmethod
    # def rlistdir(folder, tofolder=False, relative=False, ext='', filter=lambda x: True):
    #     folder = path(folder)
    #     file_list = []
    #     for f in os.listdir(str(folder)):
    #         if f == '.DS_Store': continue
    #         p = folder / f
    #         if p.is_dir():
    #             file_list.extend(path.rlistdir(p, tofolder))
    #             for cp in path.rlistdir(p, tofolder, relative=relative, ext=ext, filter=filter):
    #                 if filter(cp) and (cp | ext):
    #                     yield cp
    #         if p.is_file() and not tofolder and filter(p) and (p | ext):
    #             yield p
    #     if tofolder and not file_list and filter(folder) and (folder | ext):
    #         file_list.append(folder)
    #         yield folder
    #     file_list = pathList(file_list, main_folder=folder)
    #     if relative: file_list = -file_list
    #     if ext: file_list = file_list[file_list|ext]
    #     return file_list[filter]

    @filepath_generator_wrapper
    def recursively_listdir(self, all_files=False, depth=None):
        """
        parameters:
            all_files: whether to search hidden files or not
            depth: [-1] means folders with no subfolders
                    [0] means all files in the directory
                    [d] means paths with relative depth d (d > 0)
                 [None] means all related recursive paths with any depth
            listing with depth = 1 is equivalent to os.listdir. 
        """
        recursively_searched = False
        for f in os.listdir(self):
            if f in get_search_blacklist():
                continue
            if not all_files and f.is_hidden():
                continue
            p = self / f
            if depth is None:
                yield p
                if p.is_dir():
                    for cp in p.recursively_listdir(all_files=all_files, depth=depth):
                        yield cp
            else:
                assert isinstance(depth, int)
                if p.is_file() and depth >= 0:
                    yield p
                elif p.is_dir():
                    if depth != 1:
                        for cp in p.recursively_listdir(all_files=all_files, depth=depth-1 if depth > 0 else depth):
                            yield cp
                        recursively_searched = True
                    else:
                        yield p
        if depth == -1 and not recursively_searched:
            yield self

    def __new__(cls, *init_texts, main_folder=None):
        if len(init_texts) == 1 and isinstance(init_texts[0], (list, tuple)):
            init_texts = init_texts[0]
        if len(init_texts) <= 0 or len(init_texts[0]) <= 0:
            if main_folder:
                main_folder.main_folder = main_folder
                return main_folder
            else:
                self = super().__new__(cls, os.path.abspath(os.path.curdir))
                self.main_folder = None
                self._is_rel = False
                return self
        elif len(init_texts) == 1 and init_texts[0] == "~":
            string = path.homedir
        elif len(init_texts) == 1 and init_texts[0] == os.curdir:
            string = os.path.abspath(os.curdir)
        elif len(init_texts) == 1 and isinstance(init_texts[0], str):
            string = init_texts[0]
        else:
            [check(len(re.findall(r"[:\?$]", x)) == 0, f"Invalid characters in path '{x}'.") for x in init_texts]
            string = os.path.join(*[str(x) for x in init_texts]).strip()
        while '..' in string[2:]:
            string = re.sub(rf"{path.sep}[^{path.sep}]+{path.sep}\.\.", '', string)
            string = re.sub(rf"[^{path.sep}]+{path.sep}\.\.{path.sep}", '', string)
        _is_rel = not (os.path.isabs(string) if os.name == "nt" else (string.startswith('/') and not './' in string))
        if not main_folder: main_folder = path()
        self = super().__new__(cls, string)
        self._is_rel = _is_rel
        self.main_folder = main_folder
        return self

    def init(self): pass

    @property
    def main_folder(self):
        if self._main_folder is None:
            return None
        else:
            return self._main_folder

    @main_folder.setter
    def main_folder(self, mf):
        if mf is None:
            self._main_folder = None
        elif isinstance(mf, str) and not isinstance(mf, path):
            self._main_folder = path(mf).abs()
        elif isinstance(mf, str):
            self._main_folder = mf.abs()
        else:
            raise TypeError("main_folder should be of type [None|str|path]")

    def __and__(x, y): return path(path.pathsep.join((str(x).rstrip(path.pathsep), str(y).lstrip(path.pathsep))))
    def __mod__(x, y): return path(super().__mod__(totuple(y)), main_folder=x.main_folder)
    def __sub__(self, y):
        if y is not None: self.main_folder = path(y).dirname
        self._is_rel = True
        return self


    def __add__(x, y):
        y = str(y)
        if x.is_filepath():
            return x.dirname/(x.name + y + path.extsep + x.ext)
        else: return path(super(path, x).__add__(y), main_folder=x.main_folder)
    def __xor__(x, y):
        y = str(y).lstrip(path.namesep)
        if x.is_filepath():
            return x.dirname/(x.name.rstrip(path.namesep) + path.namesep + y + path.extsep + x.ext)
        else: return path(super(path, x.rstrip(path.namesep)).__add__(y), main_folder=x.main_folder)
    def __pow__(x, y):
        output = rootdir
        for p, q in zip((~path(x)).split(), (~path(y)).split()):
            if p == q: output /= p
            else: break
        return output - curdir
    def __floordiv__(x, y): return path(path.extsep.join((str(x).rstrip(path.extsep), str(y).lstrip(path.extsep))), main_folder=x.main_folder)
    def __invert__(self): return abs(self)
    def __abs__(self): return self.main_folder/self if self.is_rel() else self
    def abs(self): return abs(self)
    def __truediv__(x, y): return path(os.path.join(str(x), str(y)), main_folder=x.main_folder)
    def __or__(x, y):
        if y == "": return True
        if y == "FILE": return x.is_file()
        if y == "FOLDER": return x.is_dir()
        if isinstance(y, int): return len(x) == y
        if '.' not in y: y = '.*\\' + x.extsep + y
        return re.fullmatch(y.lower(), x[-1].lower()) is not None
        # return x.lower().endswith(x.extsep + y.lower())
    def __eq__(x, y): return str(x.abs()) == str(y.abs())
    def __setitem__(self, i, v):
        lst = self.split()
        lst[i] = v
        return path(lst, main_folder=self.main_folder)
    def __getitem__(self, i):
        res = self.split()[i]
        return res if isinstance(res, str) else path(path.sep.join(res), main_folder=self.main_folder)
    def __len__(self): return len(self.split())
    def __hash__(self): return super().__hash__()

    @filepath_generator_wrapper
    def __iter__(self):
        for p in self.recursively_listdir(depth=0):
            yield p

    def __contains__(self, x):
        for p in self:
            if str(p) == str(x): return True
        return False

    def str_contains(self, x): return x in str(self)

    @property
    def ext(self):
        if touch(lambda: self._ext, None):
            return self._ext
        if self.is_dir():
            self._ext = ""
            return ""
        file_name = self.fullname
        parts = file_name.split(path.extsep)
        if parts[-1].lower() in ('zip', 'gz', 'rar') and len(parts) > 2: brk = -2
        elif len(parts) > 1: brk = -1
        else: brk = 1
        self._ext = path.extsep.join(parts[brk:])
        return self._ext

    @property
    def name(self):
        if touch(lambda: self._name, None):
            return self._name
        file_name = self.fullname
        if self.is_dir():
            self._name = file_name
            return file_name
        parts = file_name.split(path.extsep)
        if parts[-1].lower() in ('zip', 'gz', 'rar') and len(parts) > 2: brk = -2
        elif len(parts) > 1: brk = -1
        else: brk = 1
        self._name = path.extsep.join(parts[:brk])
        return self._name

    def with_name(self, name):
        self = self.abs()
        return self.dirname / path.extsep.join([name, self.ext])

    def with_ext(self, ext: str):
        self = self.abs()
        return self.dirname / path.extsep.join([self.name, ext])

    @property
    def filename(self):
        if not hasattr(self, '_filename'):
            self._filename = self[-1]
        return self._filename
    
    fullname = filename
    
    @property
    def dirname(self):
        if not hasattr(self, '_dirname'):
            self._dirname = self[:-1]
        return self._dirname
    
    def is_rel(self):
        return self._is_rel

    def is_hidden(self):
        return is_hidden(self)

    def split(self, *args, **kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            return [x for x in str(self.abs()).split(path.sep)]
        return str(self).split(*args, **kwargs)

    def listdir(self, recursive=False, all_files=False, depth=None):
        if recursive:
            ret = self.recursively_listdir(all_files=all_files, depth=depth)
            ret.main_folder = self.abs()
            return ret
        else:
            if all_files:
                return pathList([self / x for x in os.listdir(str(self))], main_folder=self.abs())
            else:
                return pathList([self / x for x in os.listdir(str(self)) if not x.startswith(".")], main_folder=self.abs())

    # changed by zhangyiteng
    def ls(self, recursive=False, all_files=False, depth=None, func=None):
        return self.listdir(recursive=recursive, all_files=all_files, depth=depth).filter(func)

    def assign_mainfolder(self,  main_folder):
        self.main_folder = path(main_folder).abs()
        return self

    def cd(self, folder_name=None):
        if folder_name:
            if folder_name == os.path.pardir:
                new_folder = self.parent

            folder_name = path(folder_name)
            if folder_name.is_abs():
                return folder_name
            
            new_folder = self / folder_name
            if new_folder.is_dir():
                if self.is_abs():
                    return new_folder.abs()
                new_folder.main_folder = self.main_folder
                return new_folder
            elif new_folder.dirname.is_dir():
                raise NotADirectoryError("%s doesn't exist, all available folder is: %s" % (new_folder, new_folder.parent.ls().filter(lambda x: x.is_dir()).map(lambda x: x.name)))
            else:
                raise NotADirectoryError("%s doesn't exist" % new_folder)
        else:
            candidate = self.ls().filter(lambda x: x.is_dir()).map(lambda x: x.abs()).append(self.parent.abs())
            ret = candidate.fuzzy_search(str_func=lambda x: x.fullname)
            if ret:
                return ret[-1]

    def rm(self, remind=True):
        if self.is_dir():
            if remind and self.ls():
                print("You want to delete directory: {}".format(self))
                print("with following files inside it:")
                print(self.ls(recursive=True).vector())
                choice = input("Do you want to continue delete? [Y/n]: ")
                if choice.lower() != "y":
                    return
            self.parent.cmd(f"rm -r {self[-1]}")
        elif self.is_file():
            self.parent.cmd(f"rm {self[-1]}")

    def cmd(self, command):
        try:
            if self.is_dir():
                os.system("cd %s; %s" % (self, command))
            elif self.is_file():
                if "{}" in command:
                    self.parent.cmd(command.format(self))
                else:
                    self.parent.cmd(command + " " + self)
        except Exception as e:
            print("cmd error:", e)
    def open(self):
        if self.is_dir():
            self.cmd("open .")
        elif self.is_file():
            self.parent.cmd("open %s" % self)
    @property
    def parent(self):
        if path.sep not in self:
            return path()
        return self.dirname

    @property
    def children(self):
        return self.ls()
    # end changed by zhangyiteng
    def is_abs(self): return not self._is_rel
    def exists(self): return os.path.exists(self)
    def is_file(self): return os.path.isfile(self)
    def is_dir(self): return os.path.isdir(self)
    def is_folder(self): return self.is_dir()
    def is_filepath(self): return True if os.path.isfile(self) else 0 < len(self.ext) < 7
    def is_dirpath(self): return True if os.path.isdir(self) else (len(self.ext) == 0 or len(self.ext) >= 7)
    def mkdir(self, return_new: bool=False) -> Optional["path"]:
        """
        make directory

        for example, p = "/Users/username/code/dataset"
        p.mkdir()

        will recursive check if "/Users", "/Users/username", "/Users/username/code", "/Users/username/code", "/Users/username/code/dataset"

        is exists or not and make the corresponding directory.

        Paramemters:
        -------------------
        return_new: bool
            if return_new is True, the first made directory will be returned. If no dir is made, None will be returned
            if return_new is False, self will be returned
        """
        p = self.abs()
        if return_new:
            ret = None
        if self.main_folder:
            cumpath = path(self.main_folder)
            fp = p - cumpath
        else:
            cumpath = rootdir
            fp = p
        for p in fp.split():
            cumpath /= p
            if not cumpath.exists():
                if return_new and ret is None:
                    ret = cumpath
                os.mkdir(cumpath)
        if return_new:
            return ret
        return self

    def search(self, query="", filter=None, method="fuzzy"):
        """
        search all files in the directory

        Paramemters:
        ------------
        method: str
            which kind of method to search files, it can be:
                "fuzzy": fuzzy search which means it can tolerate minor input error.
                "regex": search files with regex repression.
        """
        if method == "fuzzy":
            return self.ls(True).filter(filter).fuzzy_search(query)
        elif method == "regex":
            return self.ls(True).filter(filter).regex_search(query)
        else:
            raise TypeError("usage: search(['fuzzy'|'regex'])")

    def copyfrom(self, src):
        if not isinstance(src, path):
            src = path(src)
        assert isinstance(src, path)
        if src.is_file():
            if self.is_file():
                shutil.copy2(src, self)
            else:
                shutil.copy2(src, self.name)

    @property
    def relative_path(self):
        return self - self.main_folder

    def get_relative_path(self, main_folder=None):
        if not main_folder:
            main_folder = self._main_folder
        return self - main_folder

    def readlines(self):
        assert self.is_file()
        return file(self).readlines()

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            raise AttributeError("{} is not a method/attribute of path, the most similar name is {}".format(name, vector(dir(self)).fuzzy_search(name, 3)))

    def file_size(self):
        def convert_bytes(num):
            """
            this function will convert bytes to MB.... GB... etc
            """
            for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
                if num < 1000.0:
                    return "%3.1f %s" % (num, x)
                num /= 1000.0
        return convert_bytes(os.stat(str(self)).st_size)

class file(path):

    endl = "\n"
    Integer = b'\x01'
    Float = b'\x02'
    Str = b'\x03'
    List = b'\x04'
    Numpy = b'\x05'
    Tuple = b'\x06'
    Dict = b'\x07'
    Set = b'\x08'
    torch_Tensor = b'\x09'
    Tensor_plus = b'\x0A'
    Vector = b'\x0B'
    torch_Module = b'\x0C'

    class streamstring:

        def __init__(self, s):
            self.s = s

        def read(self, length=-1):
            if length == -1:
                length = len(self.s)
            result = self.s[:length]
            self.s = self.s[length:]
            return result

        def __bool__(self):
            return len(self.s) > 0

    def __new__(cls, *init_texts):
        self = super().__new__(cls, *init_texts)
        self.fp = None
        return self

    @generator_wrapper
    def readlines(self):
        with open(self) as _input:
            while True:
                line = _input.readline()
                if not line:
                    break
                yield line.rstrip(self.endl)


    def writelines(self, content):
        with open(self, "w") as _output:
            for line in content:
                _output.writelines(str(line) + self.endl)

    def __iter__(self):
        with open(self) as _input:
            # while (line := _input.readline()):
            while True:
                line = _input.readline()
                if not line:
                    break
                yield line.rstrip(self.endl)

    @staticmethod
    def pack(data):
        if isinstance(data, int):
            return struct.pack("q", data)
        elif isinstance(data, float):
            return struct.pack("d", data)
        elif isinstance(data, str):
            return bytes(data, encoding="utf-8")
        else:
            raise TypeError("unknown type for pack")

    @staticmethod
    def pack_tag(tag, tag_type="B"):
        return struct.pack(tag_type, tag)

    @overload
    @staticmethod
    def _to_byte__default__(data):
        try:
            import torch
            import pyctlib.torchplus as torchplus
        except ImportError: pass
        if touch(lambda: isinstance(data, torchplus.Tensor)):
            np_array = data.cpu().detach().numpy()
            np_array_content, np_array_content_len = file._to_byte(np_array)
            assert len(np_array_content) == np_array_content_len
            return file.Tensor_plus + np_array_content, np_array_content_len + 1
        if touch(lambda: isinstance(data, torch.Tensor)):
            np_array = data.cpu().detach().numpy()
            np_array_content, np_array_content_len = file._to_byte(np_array)
            assert len(np_array_content) == np_array_content_len
            return file.torch_Tensor + np_array_content, np_array_content_len + 1
        if touch(lambda: isinstance(data, torch.nn.Module)):
            module_state_content, module_state_content_len = file._to_byte(data.state_dict())
            return file.torch_Module + module_state_content, module_state_content_len + 1

    @overload
    @staticmethod
    def _to_byte(data: int):
        result = b""
        result += file.Integer
        result += file.pack(data)
        return result, 9

    @overload
    @staticmethod
    def _to_byte(data: str):
        length = len(data)
        start_off = 0
        total_length = 2
        result = b""
        result += file.Str
        while length > 0:
            consume_length = min(255, length)
            result += file.pack_tag(consume_length)
            result += file.pack(data[start_off:start_off + consume_length])
            length -= consume_length
            start_off += consume_length
            total_length += 1 + consume_length
        result += file.pack_tag(0)
        return result, total_length

    @overload
    @staticmethod
    def _to_byte(data: float):
        result = b""
        result += file.Float
        result += file.pack(data)
        return result, 9

    @overload
    @staticmethod
    def _to_byte(data: vector):
        return file._to_byte_iterable(data, file.Vector)

    @overload
    @staticmethod
    def _to_byte(data: list):
        return file._to_byte_iterable(data, file.List)

    @overload
    @staticmethod
    def _to_byte(data: tuple):
        return file._to_byte_iterable(data, file.Tuple)

    @overload
    @staticmethod
    def _to_byte(data: set):
        return file._to_byte_iterable(data, file.Set)

    @overload
    @staticmethod
    def _to_byte(data: dict):
        items = vector(data.items())
        content = file.Dict + file._to_byte(items.map(lambda x: x[0]))[0] + file._to_byte(items.map(lambda x: x[1]))[0]
        return content, len(content)

    @overload
    @staticmethod
    def _to_byte(data: 'numpy.ndarray'):
        content = data.tobytes()
        shape = data.shape
        dtype = str(data.dtype)
        result = file.Numpy
        shape_bytes, _ = file._to_byte(list(shape))
        dtype_bytes, _ = file._to_byte(dtype)
        total_length = len(shape_bytes) + len(dtype_bytes) + len(content)
        result += file.pack_tag(total_length, "I")
        result += shape_bytes
        result += dtype_bytes
        result += content
        return result, len(result)

    @staticmethod
    def _to_byte_iterable(data, type_tag):
        list_content = b""
        list_content_length = 0
        for t in data:
            r, l = file._to_byte(t)
            list_content += r
            list_content_length += l
        assert list_content_length == len(list_content)
        result = type_tag + file.pack_tag(list_content_length, "I") + list_content
        return result, list_content_length + 5

    def __lshift__(self, data):
        if self.fp is None:
            with open(self, "ab") as _output:
                _output.write(file._to_byte(data)[0])
        else:
            self.fp.write(file._to_byte(data)[0])
        return self

    def __rshift__(self, data):
        try:
            import torch
            if isinstance(data, torch.nn.Module):
                module_data = self.get()
                data.load_state_dict(module_data)
        except ImportError:
            pass
        return self

    # @__rshift__
    # def _(self, data: nn.Module)

    @staticmethod
    def _read(fp: TextIO):
        data_type = fp.read(1)
        if data_type == file.Integer:
            data = struct.unpack("q", fp.read(8))[0]
        elif data_type == file.Float:
            data = struct.unpack("d", fp.read(8))[0]
        elif data_type == file.Str:
            data = file._read_seperated_data(fp).decode('utf-8')
        elif data_type == file.List or data_type == file.Tuple or data_type == file.Set or data_type == file.Vector:
            list_content_length = file.pointer_length(fp)
            ss = file.streamstring(fp.read(list_content_length))
            data = []
            while ss:
                data.append(file._read(ss))
            if data_type == file.Tuple:
                data = tuple(data)
            if data_type == file.Set:
                data = set(data)
            if data_type == file.Vector:
                data = vector(data)
        elif data_type == file.Dict:
            keys = file._read(fp)
            values = file._read(fp)
            data = {x: y for x, y in zip(keys, values)}
        elif data_type == file.Numpy:
            data = file._read_numpy(fp)
        elif data_type == file.torch_Tensor:
            try:
                import torch
                data = file._read(fp)
                data = torch.Tensor(data.copy())
            except: return None
        elif data_type == file.torch_Module:
            try:
                import torch
                state_dict = file._read(fp)
                return state_dict
            except:
                return None
        elif data_type == file.Tensor_plus:
            try:
                import pyctlib.torchplus as torchplus
                data = file._read(fp)
                data = torchplus.Tensor(data.copy())
            except: return None
        elif len(data_type) == 0:
            return None
        return data

    @staticmethod
    def _read_numpy(fp: TextIO):
        try:
            import numpy as np
            total_length = file.pointer_length(fp)
            ss = file.streamstring(fp.read(total_length))
            shape = file._read(ss)
            dtype = file._read(ss)
            content = ss.read()
            return np.frombuffer(content, dtype=np.dtype(dtype)).reshape(shape)
        except ImportError: return None

    @staticmethod
    def pointer_length(fp: TextIO):
        return struct.unpack("I", fp.read(4))[0]

    @generator_wrapper
    def get_all(self):
        with open(self, "rb") as fp:
            # while (a := file._read(fp)) is not None:
            while True:
                a = file._read(fp)
                if a is None:
                    break
                yield a

    def get(self, number=1):
        if self.fp is None:
            self.open("rb")
            # raise ValueError("read of closed file")
        index = 0
        result = vector()
        # while (index < number or number == -1) and (a := file._read(self.fp)) is not None:
        while (index < number or number == -1):
            a = file._read(self.fp)
            if a is None:
                break
            result.append(a)
            index += 1
        if number == -1:
            self.close()
        if not result:
            return None
        if number == 1:
            return result[0]
        else:
            return result

    @staticmethod
    def _read_seperated_data(fp: TextIO):
        result = b""
        while True:
            data_length = struct.unpack("B", fp.read(1))[0]
            if data_length == 0:
                break
            result += fp.read(data_length)
            # print(data_length, result)
        return result

    def open(self, tag):
        try:
            self.close()
            self.fp = open(self, tag)
            return self
        except Exception as e:
            print("can not open file %s" % self, e)

    def close(self):
        try:
            if self.fp: self.fp.close()
        except Exception as e:
            print("can not cloase file %s" % self, e)
        self.fp = None

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        self.close()

    def clear(self):
        with open(self, "w"):
            pass
        self.fp = None

    def read(self, length=-1, tag="b"):
        if self.fp is not None:
            return self.fp(length)
        with open(self, "r" + tag) as _input:
            return _input.read(length)

    def __len__(self):
        with open(self, "rb") as _input:
            return len(_input.read())

class filepath_generator(ctgenerator):

    def __init__(self, generator, main_folder=None):
        ctgenerator.__init__(self, generator)
        self.main_folder = main_folder

    @filepath_generator_wrapper
    def filter(self, func=None) -> "filepath_generator":
        if func is None:
            for x in self.generator:
                yield x
        if isinstance(func, str):
            for x in self.generator:
                if x | func:
                    yield x
        if isinstance(func, bytes):
            for x in self.generator:
                if x | func:
                    yield x
        for x in self.generator:
            if func(x):
                yield x

    @filepath_generator_wrapper
    def map(self, *args, **kwargs):
        return super().map(*arags, **kwargs)

    @property
    def main_folder(self):
        if self._main_folder is None:
            return None
        else:
            return self._main_folder.abs()

    @main_folder.setter
    def main_folder(self, mf):
        self._main_folder = mf

    def vector(self):
        return pathList(self, main_folder=self.main_folder)

    def fuzzy_search(self, query="", max_k=NoDefault, str_func=get_relative_path, str_display=get_relative_path, display_info=get_main_folder):
        return self.vector().fuzzy_search(query=query, max_k=max_k, str_func=str_func, str_display=str_display, display_info=display_info)

    def regex_search(self, query="", max_k=NoDefault, str_func=get_relative_path, str_display=get_relative_path, display_info=get_main_folder):
        return self.vector().regex_search(query=query, max_k=max_k, str_func=str_func, str_display=str_display, display_info=display_info)

curdir = path()
rootdir = curdir.abs()[0] + path.sep
pardir = path(os.path.pardir)
codedir = path(os.getcwd())
codefolder = path(os.getcwd())
File = b'\x04'
Folder = b'\x07'

if __name__ == '__main__': pass
