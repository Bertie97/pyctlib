from .utils import iterable
from typing import overload, Tuple, Any

class sequence(tuple):

    def __new__(cls, *args):
        if len(args) == 0:
            return tuple.__new__(cls, tuple())
        elif len(args) == 1:
            x = args[0]
            if iterable(x):
                return tuple.__new__(cls, x)
            else:
                return tuple.__new__(cls, (x, ))
        else:
            return tuple.__new__(cls, args)

    def __add__(self, x) -> "sequence":
        if iterable(x):
            return sequence(super().__add__(x))
        raise TypeError()

    def __radd__(self, x):
        return sequence(x).__add__(self)

    def __mul__(self, t) -> "sequence":
        return sequence(super().__mul__(t))

    def __rmul__(self, t) -> "sequence":
        return self.__mul__(t)

    def __getitem__(self, index):
        if isinstance(index, int):
            return super().__getitem__(index)
        elif isinstance(index, slice):
            return sequence(super().__getitem__(index))
        else:
            raise TypeError()

    @property
    def length(self) -> int:
        return len(self)

    def count(self, x):
        if callable(x):
            return sum([1 for t in self if x])
        else:
            return super().count(x)

    def map(self, func) -> "sequence":
        return sequence(func(x) for x in self)

    def map_where(self, *args):
        assert len(args) % 2 == 1
        args = tuple([sequence.__hook_function(_) for _ in args])
        def _f(x):
            for index in range(0, len(args) - 1, 2):
                if args[index](x):
                    return args[index + 1](x)
            return args[-1](x)
        return self.map(_f)

    def filter(self, func) -> "sequence":
        return sequence(x for x in self if func(x))

    @staticmethod
    def merge(self, *args) -> "sequence":
        for x in args:
            assert iterable(x)
        ret = sequence()
        for x in args:
            ret += x
        return ret

    def add(self, *args) -> "sequence":
        """
        sequence(1, 2, 3).add(4, 5)
        will get:

        (1, 2, 3, 4, 5)
        """
        return self + args

    def tuple(self) -> tuple:
        return tuple(self)

    def pop(self) -> Tuple[Any, "sequence"]:
        return self.head, self.tail

    @staticmethod
    def __hook_function(func):
        if callable(func):
            return func
        if func is None:
            return lambda x: x
        else:
            return lambda x: func

    @overload
    @staticmethod
    def range(stop) -> "sequence": ...

    @overload
    @staticmethod
    def range(start, stop) -> "sequence": ...

    @overload
    @staticmethod
    def range(start, stop, step) -> "sequence": ...

    @staticmethod
    def range(*args):
        """range.

        Parameters
        ----------
        args :
            args
        """
        return sequence(range(*args))

    @property
    def head(self):
        return self[0]

    @property
    def tail(self) -> "sequence":
        return self[1:]

    def get(self, index: int, default: Any=None) -> Any:
        if 0 <= index < self.length:
            return super().__getitem__(index)
        return default
