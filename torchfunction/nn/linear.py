from torch import nn
from torch.nn.parameter import Parameter
import torch
from torch import Tensor
import torch.nn.init as init
import torch.nn.functional as F
from pyctlib import vector, touch

def identity(x):
    return x

def get_activation_layer(activation):
    if isinstance(activation, nn.Module):
        return activation
    if callable(activation):
        return activation
    if activation == "ReLU":
        return torch.relu
    if activation == "sigmoid":
        return F.sigmoid
    if activation == "tanh":
        return F.tanh
    if activation is None or activation == "none":
        return identity
    raise ValueError

class Linear(nn.Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, activation="ReLU", hidden_dim=None, hidden_activation="ReLU") -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation
        if hidden_dim is None:
            self.dims = vector(in_features, out_features)
            self.weight = Parameter(torch.zeros(out_features, in_features))
            if bias:
                self.bias = Parameter(torch.zeros(out_features))
            else:
                self.register_parameter('bias', None)
            self.activation = get_activation_layer(activation)
        else:
            self.dims = vector(in_features, *vector(hidden_dim), out_features)
            self.weight = nn.ParameterList(self.dims.map_k(lambda in_dim, out_dim: Parameter(torch.zeros(out_dim, in_dim)), 2))
            if bias:
                self.bias = nn.ParameterList(self.dims.map_k(lambda in_dim, out_dim: Parameter(torch.zeros(out_dim)), 2))
            else:
                self.register_parameter('bias', None)
            self.activation = vector(get_activation_layer(hidden_activation) for _ in range(len(hidden_dim)))
            self.activation.append(get_activation_layer(activation))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.hidden_dim is None:
            if isinstance(self.activation, torch.nn.ReLU) or self.activation == torch.relu:
                init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='relu')
            else:
                init.xavier_normal_(self.weight)
        else:
            for a, w in zip(self.activation, self.weight):
                if isinstance(a, torch.nn.ReLU) or a == torch.relu:
                    init.kaiming_normal_(w, a=0, mode='fan_in', nonlinearity='relu')
                else:
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
                for w, a in zip(self.weight, self.activation):
                    h = a(F.linear(h, w, None))
            else:
                for w, b, a in zip(self.weight, self.bias, self.activation):
                    h = a(F.linear(h, w, b))
            return h

    def extra_repr(self) -> str:
        if self.activation is None:
            return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)
        elif isinstance(self.activation, vector):
            ret = 'in_features={}, out_features={}, bias={}, activation={}\n'.format(self.in_features, self.out_features, self.bias is not None, self.activation.map(lambda x: touch(lambda: x.__name__, str(x))))
            ret += "{}".format(self.in_features)
            for d, a in zip(self.dims[1:], self.activation):
                ret += '->{}->{}'.format(d, touch(lambda: a.__name__, str(a)))
            return ret
        else:
            ret = 'in_features={}, out_features={}, bias={}, activation={}'.format(self.in_features, self.out_features, self.bias is not None, touch(lambda: self.activation.__name__, str(self.activation)))
            return ret

    def regulization_loss(self, p=2):
        if self.hidden_dim is None:
            if p == 2:
                return self.weight.square().sum()
            if p == 1:
                return self.weight.abs().sum()
            return (self.weight.abs() ** p).sum()
        else:
            reg = []
            for w in self.weight:
                reg.append((w.weight.abs() ** p).sum())
            return sum(reg)

class test(nn.Module):

    def __init__(self):
        super(test, self).__init__()
        self.l = Linear(4, 5)

    def forward(self, x):
        return self.l(x)
