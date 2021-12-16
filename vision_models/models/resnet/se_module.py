"""
Squeeze-and-Excitation Networks
Hu, Shen, Albanie, Sun and Wu, [CVPR2018]
See https://arxiv.org/abs/1709.01507

Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks
Roy, Navab and Wachinger, [MICCAI2018]
See https://arxiv.org/abs/1803.02579
"""

from torch import Tensor
import torch.nn as nn

from .layers import conv1x1


# Original paper proposed that reduction ratio r = 16.
REDUCTION_RATIO = 16


class ChannelSEModule(nn.Module):
    """Squeeze and Excitation Module"""

    def __init__(self, in_channels: int, reduction_ratio: int = REDUCTION_RATIO, activation=nn.ReLU):
        super(ChannelSEModule, self).__init__()
        self.reduction_ratio = reduction_ratio
        hidden = in_channels // reduction_ratio
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden, bias=False),
            activation(inplace=True),
            nn.Linear(hidden, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 4
        N, C, _, _ = x.shape
        w = x.mean(dim=(2, 3))  # => (N, C)
        w = self.fc(w)
        x = x * w.reshape(N, C, 1, 1)

        return x


class SpatialSEModule(nn.Module):
    """Spatial Squeeze and Excitation Module"""

    def __init__(self, in_channels: int):
        super(SpatialSEModule, self).__init__()
        self.conv = conv1x1(in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        w = self.conv(x)  # => (N, 1, H, W)
        w = self.sigmoid(w)
        x = x * w

        return x


class SpatialChannelSEModule(nn.Module):
    """Concurrent Spatial and Channel Squeeze and Excitation Module"""

    def __init__(self, in_channels: int, reduction_ratio: int = REDUCTION_RATIO, activation=nn.ReLU):
        super(SpatialChannelSEModule, self).__init__()
        self.cSE = ChannelSEModule(in_channels, reduction_ratio, activation)
        self.sSE = SpatialSEModule(in_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.cSE(x) + self.sSE(x)

    @property
    def reduction_ratio(self) -> int:
        return self.cSE.reduction_ratio
