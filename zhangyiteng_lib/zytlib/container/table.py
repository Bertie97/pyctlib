from typing import overload, Callable
from zytlib.utils.touch import touch, crash
from .vector import vector
from zytlib.utils.utils import totuple
import pickle
import argparse
import operator
import copy


class table(dict):

    def __init__(self, *args, **kwargs):
        kwargs_has_locked = "__key_locked" in kwargs
        key_locked = kwargs.pop("__key_locked", False)
        if len(args) == 1 and isinstance(args[0], argparse.Namespace):
            for key, value in vars(args[0]).items():
                self[key] = self._hook(value)
        elif len(args) == 1 and isinstance(args[0], table) and len(kwargs) == 0:
            super().__init__(args[0])
            if not kwargs_has_locked:
                key_locked = object.__getattribute__(args[0], "__key_locked")
        elif len(args) == 2 and isinstance(args[0], list) and isinstance(args[1], list) and len(args[0]) == len(args[1]) and len(kwargs) == 0:
            d = dict()
            for key, value in zip(args[0], args[1]):
                d[key] = self._hook(value)
            super().__init__(d)
        else:
            for arg in args:
                if not arg:
                    continue
                elif isinstance(arg, dict):
                    for key, value in arg.items():
                        self[key] = self._hook(value)
                elif isinstance(arg, tuple) and len(arg) == 2 and not isinstance(arg[0], tuple):
                    self[arg[0]] = self._hook(arg[1])
                else:
                    for key, value in iter(arg):
                        self[key] = self._hook(value)
            for key, value in kwargs.items():
                self[key] = value
        object.__setattr__(self, "__key_locked", key_locked)

    def hieratical(self, delimiter=".") -> "table":
        ret = table()
        for key, value in self.items():
            ret.pset(key.split(delimiter), value=value)
        return ret

    def flatten(self, delimiter=".") -> "table":
        ret = table()
        for key, value in self.items():
            if not isinstance(value, table):
                ret[key] = value
            else:
                for _k, _v in value.flatten().items():
                    ret[delimiter.join([key, _k])] = _v
        return ret

    @classmethod
    def _hook(cls, item):
        if isinstance(item, dict):
            return cls(item)
        elif isinstance(item, (list, tuple)):
            return type(item)(cls._hook(x) for x in item)
        return item

    def __add__(x, y):
        if isinstance(y, dict):
            y = table(y)
            if set(x.keys()) & set(y.keys()):
                raise ValueError("table+table requires keys in two table are different")
            ret = table()
            ret.update(x)
            ret.update(y)
            return ret

    @overload
    def merge(self, new_dict: dict, reduce_func: Callable) -> "table": ...

    @overload
    def merge(self, new_dict: dict, reduce_func: Callable, default=None) -> "table": ...

    def merge(self, new_dict: dict, reduce_func: Callable, **kwargs):
        if (has_default:= "default" in kwargs):
            default = kwargs["default"]

        for key, value in new_dict.items():
            if key not in self:
                if has_default:
                    self[key] = reduce_func(default, value)
                else:
                    self[key] = value
            else:
                self[key] = reduce_func(self[key], value)
        return self

    def update_exist(self, *args, **kwargs) -> "table":
        for arg in args:
            assert isinstance(arg, dict)
            for key in arg.keys():
                if key in self:
                    self[key] = arg[key]

        for key in kwargs.keys():
            if key in self:
                self[key] = kwargs[key]

        return self

    def update_where(self, target, condition) -> "table":
        for key, value in super().items():
            if condition(value):
                self[key] = target.get(key, self[key])
        return self

    def update_notexist(self, target) -> "table":
        for key, value in target.items():
            if key in self:
                continue
            self[key] = value
        return self

    def key_not_here(self, d) -> vector:
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

    def __getattr__(self, item: str):
        if item.startswith("__"):
            return object.__getattribute__(self, item)
        return self.__getitem__(item)

    def __missing__(self, name):
        raise KeyError(name)

    def __getstate__(self):
        return self.dict(), object.__getattribute__(self, "__key_locked")

    def __setstate__(self, state):
        content, key_locked = state
        object.__setattr__(self, "__key_locked", False)
        self.update(content)
        object.__setattr__(self, "__key_locked", key_locked)

    def lock_key(self):
        object.__setattr__(self, "__key_locked", True)

    def unlock_key(self):
        object.__setattr__(self, "__key_locked", False)

    def save(self, filepath):
        try:
            import pickle
        except:
            print("Please install pickle package")
            return
        with open(str(filepath), "wb") as output:
            pickle.dump(self, output)

    @staticmethod
    def load(filepath) -> "table":
        try:
            import pickle
        except:
            print("Please install pickle package")
            return
        with open(filepath, "rb") as input:
            ret = pickle.load(input)
            def _dict_to_table(t):
                for key, value in t.items():
                    if isinstance(value, dict) and not isinstance(value, table):
                        t[key] = _dict_to_table(value)
                if isinstance(t, dict) and not isinstance(t, table):
                    return table(t)
                return t
        return _dict_to_table(ret)

    def __setitem__(self, key, value):
        try:
            if object.__getattribute__(self, "__key_locked") and key not in self:
                raise RuntimeError("dict key is locked")
        except RuntimeError:
            raise RuntimeError("dict key is locked")
        except:
            pass

        super().__setitem__(key, value)

    def pset(self, *keys, value) -> None:
        keys = totuple(keys)
        parent = self
        for key in keys[:-1]:
            if key not in parent:
                parent[key] = table()
            parent = parent[key]
        parent[keys[-1]] = value

    def filter(self, key=None, value=None) -> "table":
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
            def key(x): return x
        if value is None:
            def value(x): return x
        ret = dict()
        for x, y in super().items():
            ret[key(x)] = value(y)
        return table(ret)

    def rmap(self, func) -> "table":
        if func is None:
            return self
        ret = table()
        for x, y in super().items():
            if not isinstance(y, table):
                ret[x] = func(y)
            else:
                ret[x] = y.rmap(func)
        return ret

    def values(self) -> vector:
        return vector(super().values())

    def rvalues(self) -> vector:
        ret = vector()
        for v in super().values():
            if isinstance(v, table):
                ret.extend(v.rvalues())
            else:
                ret.append(v)
        return ret

    def keys(self) -> vector:
        return vector(super().keys())

    def items(self) -> vector:
        return vector(super().items())

    def dict(self) -> dict:
        base = {}
        for key, value in self.items():
            if isinstance(value, type(self)):
                base[key] = value.dict()
            elif isinstance(value, (list, tuple, vector)):
                base[key] = type(value)(
                    item.dict() if isinstance(item, type(self)) else item for item in value)
            else:
                base[key] = value
        return base

    def pretty_print(self) -> None:
        for key, value in self.items():
            if isinstance(value, str):
                print(f"{key}: \"{value}\"")
            else:
                print(f"{key}: {value}")

    def copy(self) -> "table":
        ret = table()
        for key, value in super().items():
            if isinstance(value, table):
                ret[key] = value.copy()
            else:
                ret[key] = copy.deepcopy(value)
        return ret

    def __dir__(self):
        return ["keys", "items", "copy", "dict", "values", "rvalues", "map", "rmap", "hieratical", "flatten", "merge", "update_exist", "update_where", "update_notexist", "key_not_here", "lock_key", "unlock_key", "load", "pset", "filter", "pretty_print"] + self.keys()
