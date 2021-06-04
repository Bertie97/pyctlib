from torch import nn
from torch.nn.parameter import Parameter
import torch
from torch import Tensor
import torch.nn.init as init
import torch.nn.functional as F
from pyctlib import vector

class Linear(nn.Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, activation="ReLU", hidden_dim=None) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        if hidden_dim is None:
            self.weight = Parameter(torch.zeros(out_features, in_features))
            if bias:
                self.bias = Parameter(torch.zeros(out_features))
            else:
                self.register_parameter('bias', None)
        else:
            temp_dim = vector(in_features, *vector(hidden_dim), out_features)
            self.weight = nn.ParameterList(temp_dim.map_k(lambda in_dim, out_dim: Parameter(torch.zeros(out_dim, in_dim)), 2))
            if bias:
                self.bias = nn.ParameterList(temp_dim.map_k(lambda in_dim, out_dim: Parameter(torch.zeros(out_dim)), 2))
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
        if self.hidden_dim is None:
            if isinstance(self.activation, torch.nn.ReLU):
                init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='relu')
            else:
                init.xavier_normal_(self.weight)
        else:
            if isinstance(self.activation, torch.nn.ReLU):
                for w in self.weight:
                    init.kaiming_normal_(w, a=0, mode='fan_in', nonlinearity='relu')
            else:
                for w in self.weight:
                    init.xavier_normal_(w)

    def forward(self, input: Tensor) -> Tensor:
        if self.hidden_dim is None:
            if self.activation is None:
                return F.linear(input, self.weight, self.bias)
            else:
                return self.activation(F.linear(input, self.weight, self.bias))
        else:
            h = input
            if self.bias is None:
                for w in self.weight:
                    h = self.activation(F.linear(h, w, None))
            else:
                for w, b in zip(self.weight, self.bias):
                    h = self.activation(F.linear(h, w, b))
            return h

    def extra_repr(self) -> str:
        if self.activation is None:
            return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)
        else:
            return 'in_features={}, out_features={}, bias={}, activation={}'.format(self.in_features, self.out_features, self.bias is not None, self.activation)
