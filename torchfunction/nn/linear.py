from torch import nn
from torch.nn.parameter import Parameter
import torch
from torch import Tensor
import torch.nn.init as init
import torch.nn.functional as F

class Linear(nn.Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, activation="ReLU") -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.zeros(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        # print(self.weight)
        if activation == "ReLU":
            self.activation = nn.ReLU()
        else:
            self.activation = activation
        self.reset_parameters()
        # print(self.weight)

    def reset_parameters(self) -> None:
        if isinstance(self.activation, torch.nn.ReLU):
            init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='relu')
        else:
            init.xavier_normal_(self.weight)

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
