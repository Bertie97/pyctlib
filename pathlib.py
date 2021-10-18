import os, re, struct, shutil
from .touch import touch, check
from .basicwrapper import raw_function, register_property
from functools import wraps, reduce, partial
import typing
from .vector import NoDefault, UnDefined, OutBoundary, vector, generator_wrapper, ctgenerator, IndexMapping, EmptyClass
from rapidfuzz import fuzz

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
    return Path(".").abs()

class pathList(vector):
    def __init__(self, *args, index_mapping=indexmapping(), main_folder = none):
        super().__init__(*args, index_mapping=index_mapping, content_type=path)
        self.main_folder = main_folder
    def __getitem__(self, index):
        ret = super().__getitem__(index)
        if isinstance(ret, Path):
            ret = pathList(ret, main_folder=self.main_folder)
        elif isinstance(ret, Path):
            ret.main_folder = self.main_folder
        else:
            raise RuntimeError("strange item in pathList: {}".format(str(ret)))
        return ret

    def map(self, *args, **kwargs):
        ret = super().map(*args, **kwargs)
        if ret.check_type(Path):
            return pathList(ret, main_folder=self.main_folder)
        else:
            return ret

    def assign_mainfolder(self, main_folder=UnDefined):
        if isinstance(main_folder, EmptyClass):
            main_folder = self.main_folder
        ret = self.map(lambda x: x.assign_mainfolder(main_folder))
        return ret

    def assign_mainfolder_(self, main_folder=UnDefined):
        if isinstance(main_folder, EmptyClass):
            main_folder = self._main_folder
        self.map_(lambda x: x.assign_mainfolder(main_folder))

    @property
    def main_folder(self):
        if self.__main_folder is None:
            return None
        else:
            return self.__main_folder.abs()

    @main_folder.setter
    def main_folder(self, mf):
        self.__main_folder = Path(mf)

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

class Path(str):

    sep = os.path.sep
    extsep = os.path.extsep
    pathsep = os.path.pathsep
    namesep = "_"
    homedir = os.path.expanduser("~")
    curdir = os.curdir

    def __new__(cls, string="", main_folder=None):
        assert isinstance(string, str)
        if isinstance(string, Path) and main_folder is None:
            self = super().__new__(cls, str(string))
            self.main_folder = main_folder
            return self
        string = str(string)
        if string == "":
            string = ""
        elif string == "~":
            string = Path.homedir
        elif string == Path.curdir:
            string = os.path.abspath(Path.curdir)
        if main_folder is None:
            main_folder = None
        elif isinstance(main_folder, str):
            main_folder = str(main_folder)
            main_folder = Path(main_folder)
        self = super().__new__(cls, string)
        self.main_folder = main_folder
        return self

    def abs(self):
        if self.main_folder is None:
            return Path(os.path.abspath(str(self)))
        else:
            return self.main_folder / self

    def __sub__(self, y):
        assert isinstance(y, str)
        x = self.abs()
        y = Path(os.path.abspath(str(y)))
        assert x.startswith(y)
        x_split = x.split()
        y_split = y.split()
        return Path(Path.sep.join(x_split[len(y_split):]), main_folder=y)

    def __truediv__(x, y):
        return Path(Path.sep.join((str(x).rstrip(Path.extsep), str(y).lstrip(Path.extsep))), main_folder=x.main_folder)

    def __eq__(x, y):
        if isinstance(y, Path):
            return str(x.abs()) == str(y.abs())
        if isinstance(y, str):
            return x == Path(y)
        return False

    def split(self, *args):
        if len(args) == 0:
            return [x for x in str(self.abs()).split(Path.sep)]
        return str(self).split(*args)

    @property
    @register_property
    def ext(self):
        if self.isdir():
            return ""
        file_name = self.fullname
        parts = file_name.split(Path.extsep)
        if parts[-1].lower() in ('zip', 'gz', 'rar') and len(parts) > 2: brk = -2
        elif len(parts) > 1: brk = -1
        else: brk = 1
        return Path.extsep.join(parts[brk:])

    @property
    @register_property
    def name(self):
        file_name = self.fullname
        if self.isdir():
            return file_name
        parts = file_name.split(Path.extsep)
        if parts[-1].lower() in ('zip', 'gz', 'rar') and len(parts) > 2: brk = -2
        elif len(parts) > 1: brk = -1
        else: brk = 1
        return Path.extsep.join(parts[:brk])

    @property
    @register_property
    def fullname(self):
        return self.split()[-1]

    @property
    @register_property
    def dirname(self):
        return Path(Path.sep.join(self.split()[:-1]))

    @property
    @register_property
    def parent(self):
        if Path.sep not in str(self):
            self = self.abs()
            if Path.sep not in str(self):
                return Path()
            else:
                return self.dirname
        else:
            return self.dirname

    def with_name(self, name):
        self = self.abs()
        return self.dirname / Path.extsep.join([name, self.ext])

    def with_ext(self, ext: str):
        self = self.abs()
        return self.dirname / Path.extsep.join([self.name, ext])

    def check_name(self, name):
        assert isinstance(name, str)
        return self.name == name

    def check_ext(self, ext):
        assert isinstance(ext, str)
        return self.ext == ext

    def exists(self):
        return os.path.exists(str(self))

    def isfile(self):
        return os.path.isfile(str(self))

    def isdir(self):
        return os.path.isdir(str(self))

    def isfolder(self):
        return os.path.isfolder(str(self))

    def mkdir(self, return_new: bool=False) -> Optional["Path"]:
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
            cumpath = Path(self.main_folder)
            fp = p - cumpath
        else:
            cumpath = Path()
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

    def assign_mainfolder(self, main_folder):
        self.main_folder = Path(main_folder).abs()
        return self

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
