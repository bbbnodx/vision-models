from torch import Tensor
import torch.nn as nn

from typing import Optional, Type


class MLP(nn.Module):
    """ Two Layers MLP with dropout.
    in - fc - act - dropout - fc - dropout - out
    """

    def __init__(self, in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 activation: Type[nn.Module] = nn.GELU,
                 p_dropout: float = 0.):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features if hidden_features is not None else in_features
        self.out_features = out_features if out_features is not None else in_features

        self.fc1 = nn.Linear(self.in_features, self.hidden_features)
        self.act = activation()
        self.fc2 = nn.Linear(self.hidden_features, self.out_features)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x  # => (N, out_features)


class GluMLP(nn.Module):
    """ MLP w/ GLU style gating
    See https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202

    in - fc - split - act - hadamard - dropout - fc - dropout - out
                └────────┘
    """

    def __init__(self, in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 activation: Type[nn.Module] = nn.Sigmoid,
                 p_dropout: float = 0.):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features if hidden_features is not None else in_features
        self.out_features = out_features if out_features is not None else in_features
        assert self.hidden_features & 1 == 0, f"`hidden_features` should be even, but `{self.hidden_features}`."

        self.fc1 = nn.Linear(self.in_features, self.hidden_features)
        self.act = activation()
        self.fc2 = nn.Linear(self.hidden_features // 2, self.out_features)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)  # => (N, hidden_features)
        x, gates = x.chunk(2, dim=-1)  # => (N, hidden_features // 2), (N, hidden_features // 2)
        x = x * self.act(gates)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x  # => (N, out_features)
