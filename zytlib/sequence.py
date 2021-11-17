from .utils import iterable

class sequence:

    def __init__(self, *args):
        if len(args) == 0:
            self.__content = tuple()
        elif len(args) == 1:
            x = args[0]
            if iterable(x):
                self.__content = tuple(x)
            else:
                self.__content = (x, )
        else:
            self.__content = args

    def __iter__(self):
        return self.__content.__iter__()

    def __getitem__(self, index):
        return self.__content[index]

    def __add__(self, x) -> sequence:
        if iterable(x):
            self.__content = self.__content + tuple(x)
            return self
        raise TypeError()

    def __radd__(self, x):
        return sequence(x).__add__(self.__content)

    def __contains__(self, x):
        return x in self.__content

    def __mul__(self, t) -> sequence:
        self.__content = self.__content * t
        return self

    def __rmul__(self, t) -> sequence:
        return self.__mul__(t)

    def __len__(self) -> int:
        return len(self.__content)

    def __str__(self) -> str:
        return str(self.__content)

    def __repr__(self) -> str:
        return str(self.__content)

    @property
    def length(self) -> int:
        return len(self)

    def __hash__(self):
        return hash(self.__content)

    def __eq__(self, other) -> bool:
        if isinstance(other, (tuple, sequence)):
            return self.__content == other
        return False

    def count(self, x):
        if callable(x):
            return sum([1 for t in self if x])
        else:
            return self.__content.count(x)

    def map(self, func) -> sequence:
        return sequence(func(x) for x in self)

    def filter(self, func) -> sequence:
        return sequence(x for x in self if func(x))

    @staticmethod
    def merge(self, *args) -> sequence:
        for x in args:
            assert iterable(x)
        ret = sequence()
        for x in args:
            ret += x
        return ret

    def add(self, *args) -> sequence:
        """
        sequence(1, 2, 3).add(4, 5)
        will get:

        (1, 2, 3, 4, 5)
        """
        return self + args

    def tuple(self) -> tuple:
        return tuple(self)
