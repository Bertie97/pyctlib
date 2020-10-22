#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package torchplus.nn.modules
##############################

from . import Module
from ..parameter import Parameter
from ...tensor import Tensor
import torchplus
from .. import functional as F
import torch.nn.init as init

class Linear(Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, activation=None) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torchplus.zeros(out_features, in_features))
        if bias:
            self.bias = Parameter(torchplus.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self) -> None:
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='relu')
        # if self.bias is not None:
        #     fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        #     bound = 1 / math.sqrt(fan_in)
        #     init.uniform_(self.bias, -bound, bound)
        # TODO: different initialization method based on different activation function

    def forward(self, input: Tensor) -> Tensor:
        if self.activation is None:
            return F.linear(input, self.weight, self.bias)
        else:
            return self.activation(F.linear(input, self.weight, self.bias))

    def extra_repr(self) -> str:
        if self.activation is None:
            return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)
        else:
            return 'in_features={}, out_features={}, bias={}, activation={}'.format(self.in_features, self.out_features, self.bias is not None, self.activation)
