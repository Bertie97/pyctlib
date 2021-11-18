from .utils import iterable

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
