"""
Activation functions

Mish: A Self Regularized Non-Monotonic Neural Activation Function
Diganta Misra [2019]
See https://arxiv.org/abs/1908.08681

mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))


Swish: a Self-Gated Activation Function (Searching for Activation Functions)
Ramachandran, Zoph and Le [2017]
See https://arxiv.org/abs/1710.05941

swish(x) = x * sigmoid(beta * x)

TanhExp: A Smooth Activation Function with High Convergence Speed for Lightweight Neural Networks
Liu and Di [2020]
See https://arxiv.org/abs/2003.09855

tanhexp(x) = x * tanh(exp(x))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    """
    Mish: A Self Regularized Non-Monotonic Neural Activation Function
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    """
    def __init__(self, inplace=False):
        super(Mish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return mish(x, inplace=self.inplace)


def mish(x: torch.Tensor, inplace=False) -> torch.Tensor:
    """
    Mish: A Self Regularized Non-Monotonic Neural Activation Function
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    """
    if inplace:
        return x.mul_(F.softplus(x).tanh())
    else:
        return x.mul(F.softplus(x).tanh())


class Swish(nn.Module):
    """
    Swish: a Self-Gated Activation Function
    swish(x) = x * sigmoid(beta * x)


    """
    def __init__(self, inplace=False, trainable=False):
        super(Swish, self).__init__()
        self.inplace = inplace
        if trainable:
            self.beta = nn.Parameter(torch.ones(1, dtype=torch.float))  # scalar

    def forward(self, x):
        if hasattr(self, 'beta'):
            return swish(x, beta=self.beta, inplace=self.inplace)
        else:
            return swish(x, beta=1.0, inplace=self.inplace)


def swish(x: torch.Tensor, beta=1.0, inplace=False) -> torch.Tensor:
    """
    Swish: a Self-Gated Activation Function
    swish(x) = x * sigmoid(beta * x)
    """
    if inplace:
        return x.mul_(x.mul(beta).sigmoid())
    else:
        return x.mul(x.mul(beta).sigmoid())


class TanhExp(nn.Module):
    """
    TanhExp: A Smooth Activation Function with High Convergence Speed for Lightweight Neural Networks
    tanhexp(x) = x * tanh(exp(x))
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return tanhExp(x)


def tanhExp(x: torch.Tensor) -> torch.Tensor:
    return x.mul(x.exp().tanh())
