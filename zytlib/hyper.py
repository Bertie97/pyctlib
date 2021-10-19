from typing import overload
from .touch import touch, crash

class Dict(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        super().__setattr__("__key_locked", False)

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

    def lock_key(self):
        super().__setattr__("__key_locked", True)

    def unlock_key(self):
        super().__setattr__("__key_locked", False)

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
    def load(filepath) -> "Dict":
        try:
            import pickle
        except:
            print("Please install pickle package")
            return
        with open(filepath, "rb") as input:
            content = pickle.load(input)
            ret = Dict(content)
        return ret

    def __getattr__(self, key):
        if key in super().keys():
            return super().__getitem__(key)
        raise ValueError()

    def __setattr__(self, key, value):
        if key not in self and self.locked:
            raise RuntimeError("dict key is locked")
        super().__setitem__(key, value)

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
        return Dict(ret)

    def map(self, func):
        ret = dict()
        for x, y in self.items():
            ret[x] = func(y)
        return Dict(ret)
