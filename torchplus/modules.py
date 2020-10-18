import torch
import torch.nn as nn
from pyctlib.basics.basictype import *
from pyctlib.torchplus.tensor import *
import pyctlib.torchplus as torchplus

class Module(nn.Module):

    def __init__(self):
        super(Module, self).__init__()

    def __dir__(self):
        return vector(super().__dir__())

class Linear(Module):

    def __init__(self, in_features: int, out_features: int, bias: bool = True):

        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_feature = out_features
        self.weight = torchplus.zeros(in_features, out_features).requires_grad_()
        if bias:
            self.bias = torchplus.zeros(out_features).requires_grad_()
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            return x @ self.weight + self.bias.unsqueeze(0)
        else:
            return x @ self.weight

    def __str__(self):
        return "Linear(in_features={}, out_features={}, bias={})".format(self.in_features, self.out_feature, self.bias is not None)

    def __repr__(self):
        return str(self)
