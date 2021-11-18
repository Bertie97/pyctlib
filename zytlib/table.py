from typing import overload
from .touch import touch, crash
from .vector import vector
import pickle
import argparse
import operator

class table(dict):

    def __init__(self, *args, **kwargs):
        object.__setattr__(self, "__key_locked", kwargs.pop("__key_locked", False))
        if len(args) == 1 and isinstance(args[0], argparse.Namespace):
            super().__init__(vars(args[0]))
        elif len(args) == 2 and isinstance(args[0], list) and isinstance(args[1], list) and len(args[0]) == len(args[1]) and len(kwargs) == 0:
            d = dict()
            for key, value in zip(args[0], args[1]):
                d[key] = value
            super().__init__(d)
        else:
            super().__init__(*args, **kwargs)

    def __add__(x, y):
        y = table(y)
        if set(x.keys()) & set(y.keys()):
            raise ValueError("table+table requires keys in two table are different")
        ret = table()
        ret.update(x)
        ret.update(y)
        return ret

    def update_exist(self, *args, **kwargs):

        for arg in args:
            assert isinstance(arg, dict)
            for key in arg.keys():
                if key in self:
                    self[key] = arg[key]

        for key in kwargs.keys():
            if key in self:
                self[key] = kwargs[key]

        return self

    def key_not_here(self, d):
        ret = vector()
        for key in d.keys():
            if key not in self:
                ret.append(key)
        return ret

    def __setattr__(self, name, value):
        if hasattr(self.__class__, name):
            raise AttributeError("'table' object attribute "
                                 "'{0}' is read-only".format(name))
        else:
            self[name] = value

    def __getattr__(self, item):
        return self.__getitem__(item)

    def __missing__(self, name):
        return KeyError(name)

    def lock_key(self):
        object.__setattr__(self, "__key_locked", True)

    def unlock_key(self):
        object.__setattr__(self, "__key_locked", False)

    @property
    def locked(self):
        return super().__getattribute__("__key_locked")

    def save(self, filepath):
        try:
            import pickle
        except:
            print("Please install pickle package")
            return
        with open(str(filepath), "wb") as output:
            pickle.dump(dict(self), output)

    @staticmethod
    def load(filepath) -> "table":
        try:
            import pickle
        except:
            print("Please install pickle package")
            return
        with open(filepath, "rb") as input:
            content = pickle.load(input)
            ret = table(content)
        return ret

    def __setitem__(self, key, value):
        if key not in self and self.locked:
            raise RuntimeError("dict key is locked")
        super().__setitem__(key, value)

    def filter(self, key=None, value=None):
        if key is None and value is None:
            return self
        if key is None:
            key = lambda x: True
        if value is None:
            value = lambda x: True
        ret = dict()
        for x, y in self.items():
            if key(x) and value(y):
                ret[x] = y
        return table(ret)

    def map(self, key=None, value=None) -> "table":
        if key is None and value is None:
            return self
        if key is None:
            key = lambda x: x
        if value is None:
            value = lambda x: x
        ret = dict()
        for x, y in self.items():
            ret[key(x)] = value(y)
        return table(ret)

    def values(self) -> vector:
        return vector(super().values())

    def keys(self) -> vector:
        return vector(super().keys())

    def items(self) -> vector:
        return vector(super().items())
