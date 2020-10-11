import torch
import torch.nn as nn
from pyctlib.basicstype import vector
from pyctlib.override import override

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Tensor(torch.Tensor):

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls, *args, **kwargs).to(device)
        self._batch_dimension = kwargs.get("with_batch", None)
        return self

    def __init__(self, *args, **kwargs):
        pass

    @property
    def batch_dimension(self):
        return self._batch_dimension

    @property
    def batch_size(self):
        if self.batch_dimension is None:
            raise ValueError("there is no dimension provided for this tensor")
