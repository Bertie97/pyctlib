from pyctlib import vector, totuple
from typing import Union

class MimcMatrix:

    def __init__(self, *args, symbol="x"):
        if len(args) == 1 and isinstance(args[0], vector):
            self.symbol = args[0].flatten()[0].split("[")[0]
            self.shape = args[0].shape
            self.content = args[0].copy()
        else:
            args = totuple(args)
            self.symbol = symbol
            self.shape = args
            self.content = vector.meshgrid(vector(args).map(lambda n: vector.range(1, n + 1))).map(lambda loc: "{}[{}]".format(symbol, ",".join(str(l) for l in loc))).reshape(args)

    def __str__(self):
        return str(self.content)

    def __repr__(self):
        return repr(self.content)

    @property
    def T(self):
        return MimcMatrix(self.content.T)

    def __add__(self, other: Union[MimcMatrix, str]):
        if isinstance(other, MimcMatrix):
            return 
