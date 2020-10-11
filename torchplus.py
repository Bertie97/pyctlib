try: import torch
except ImportError: raise ImportError("'pyctlib.torchplus' cannot be used without dependency 'torch'.")
import torch.nn as nn
from pyctlib.basictype import vector
from pyctlib.override import override

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Tensor(torch.Tensor):

    def __new__(cls, data, dtype=None, device=None, requires_grad=False, batch_dimension=None, auto_device=True):
        self = super().__new__(cls, data, dtype=dtype, requires_grad=requires_grad)
        if auto_device:
            self.to(device)
        self._auto_device=auto_device
        self._batch_dimension = batch_dimension
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
        return self.shape[self.batch_dimension]

    def numpy(self: 'Tensor'):
        return self.cpu().detach().numpy()
