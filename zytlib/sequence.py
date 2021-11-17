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

    def __add__(self, x):
        if iterable(x):
            self.__content = self.__content + tuple(x)
            return self
        raise TypeError()

    def __radd__(self, x):
        return sequence(x).__add__(self.__content)

    def __contains__(self, x):
        return x in self.__content

    def __mul__(self, t):
        self.__content = self.__content * t
        return self

    def __rmul__(self, t):
        return self.__mul__(t)

    def __len__(self):
        return len(self.__content)

    def __str__(self):
        return str(self.__content)

    def __repr__(self):
        return str(self.__content)

    @property
    def length(self):
        return len(self)

    def __hash__(self):
        return hash(self.__content)

    def count(self, x):
        if callable(x):
            return sum([1 for t in self if x])
        else:
            return self.__content.count(x)

    def map(self, func):
        return sequence(func(x) for x in self)

    def filter(self, func):
        return sequence(x for x in self if func(x))

    @staticmethod
    def merge(self, *args):
        for x in args:
            assert iterable(x)
        ret = sequence()
        for x in args:
            ret += x
        return ret

    def add(self, *args):
        """
        sequence(1, 2, 3).add(4, 5)
        will get:

        (1, 2, 3, 4, 5)
        """
        return self + args

    def tuple(self):
        return tuple(self)
