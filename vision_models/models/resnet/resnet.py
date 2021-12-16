"""
ResNet

References:

"Deep Residual Learning for Image Recognition"
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun [CVPR2016]
See https://arxiv.org/pdf/1512.03385.pdf

"pre activation ResNet"
Identity Mappings in Deep Residual Networks
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun [ECCV2016]
See https://arxiv.org/pdf/1603.05027v1.pdf

torchvisionのResNet実装
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

"Bag of Tricks for Image Classification with Convolutional Neural Networks"
Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, Mu Li [2018]
See https://arxiv.org/pdf/1812.01187.pdf

1. In bottleneck block, switching the strides size of the first two convolutions.
2. In stem block, replacing the 7x7 convolution with three 3x3 convolutions.
3. In downsample path, adding a 2x2 average pooling layer with a stride of 2
   before the 1x1 convolution, whose stride is changed to 1.
"""

import torch.nn as nn

import warnings
from typing import Type, ClassVar, Union
import functools

from .layers import conv3x3, conv1x1
from .se_module import SpatialChannelSEModule
from ..utils import adaptive_groups, initialize_weights_


class ResidualBlock(nn.Module):
    """Base class of ResNet block

    Parameters
    ----------
    in_planes : int
        入力チャンネル数
    planes : int
        出力チャンネル数
        実際にはexpansionを乗じた値になる
    stride : Union[int, Tuple[int, int]], optional
        downsample convolutionにおけるストライド, by default 1
        空間サイズの縮小倍率となる
    dilation : int, optional
        畳み込みカーネルのdilation, by default 1
    use_se_module : bool, optional
        Squeeze-and-Excitation moduleを組み込むかどうか, by default False
    activation : Type[nn.Module], optional
        活性化関数, by default nn.ReLU
        inplace以外のオプションを指定する場合は事前に部分適用しておく
    """

    expansion: ClassVar[int]

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError


class BasicBlock(ResidualBlock):
    """Basic block"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dilation=1,
                 use_se_module=False, activation: Type[nn.Module] = nn.ReLU):

        super(BasicBlock, self).__init__()
        self.norm1 = nn.GroupNorm(num_groups=adaptive_groups(in_planes), num_channels=in_planes)
        self.act = activation(inplace=True)
        self.conv1 = conv3x3(in_planes, planes, stride=stride, dilation=dilation)
        self.norm2 = nn.GroupNorm(num_groups=adaptive_groups(planes), num_channels=planes)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.use_se_module = use_se_module
        if self.use_se_module:
            self.se = SpatialChannelSEModule(planes)
        self.stride = stride
        ex_planes = planes * self.expansion

        if stride != 1 or in_planes != ex_planes:
            # When the spatial size change or the dimensions increase
            self.downsample = nn.Sequential(
                nn.AvgPool2d(stride, stride, ceil_mode=True, count_include_pad=False),
                conv1x1(in_planes, ex_planes),
                nn.GroupNorm(num_groups=adaptive_groups(ex_planes), num_channels=ex_planes),
            )

    def forward(self, x):
        shortcut = self.downsample(x) if hasattr(self, 'downsample') else x

        out = self.norm1(x)
        out = self.act(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.act(out)
        out = self.conv2(out)

        out = self.se(out) if self.use_se_module else out

        out += shortcut

        return out


class BottleneckBlock(ResidualBlock):
    """Residual block (bottleneck)"""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1, dilation=1,
                 use_se_module=False, activation: Type[nn.Module] = nn.ReLU):
        super(BottleneckBlock, self).__init__()
        self.norm1 = nn.GroupNorm(num_groups=adaptive_groups(in_planes), num_channels=in_planes)
        self.act = activation(inplace=True)
        self.conv1 = conv1x1(in_planes, planes)
        self.norm2 = nn.GroupNorm(num_groups=adaptive_groups(planes), num_channels=planes)
        self.conv2 = conv3x3(planes, planes, dilation=dilation, stride=stride)
        self.norm3 = nn.GroupNorm(num_groups=adaptive_groups(planes), num_channels=planes)
        ex_planes = planes * self.expansion
        self.conv3 = conv1x1(planes, ex_planes)
        self.use_se_module = use_se_module
        if self.use_se_module:
            self.se = SpatialChannelSEModule(ex_planes)
        self.stride = stride

        if stride != 1 or in_planes != ex_planes:
            # When the spatial size change or the dimensions increase
            self.downsample = nn.Sequential(
                nn.AvgPool2d(stride, stride, ceil_mode=True, count_include_pad=False),
                conv1x1(in_planes, ex_planes),
                nn.GroupNorm(num_groups=adaptive_groups(ex_planes), num_channels=ex_planes),
            )

    def forward(self, x):
        shortcut = self.downsample(x) if hasattr(self, 'downsample') else x

        out = self.norm1(x)
        out = self.act(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.act(out)
        out = self.conv2(out)

        out = self.norm3(out)
        out = self.act(out)
        out = self.conv3(out)

        out = self.se(out) if self.use_se_module else out

        out += shortcut

        return out


# Type Alias
BlockIdentifier = Union[str, Type[ResidualBlock]]


class ResNetExtractor(nn.Module):
    """
    各Residual layerのブロック数をリストで受け取り、
    動的にネットワークを生成する

    Parameters
    ----------
    in_channels : int
        入力画像のチャンネル数
    block : str or Type[ResidualBlock]
        Residual blockの指定
        BasicBlock or BottleneckBlock
    layers : Sequence[int]
        各Residual blockのレイヤ数を指定する
    channels : Sequence[int]
        各Residual blockの出力マップ数を指定する
        実際の出力マップ数はこの数値にblock.expansionを乗じた値になる
    strides : Sequence[Union[int, Tuple[int, int]]]
        各Residual blockのストライド(空間サイズの縮小倍率)を指定する
        ストライドは(stride_H, stride_W)のタプルでも指定できる
    dilations : Sequence[int]
        各Residual blockのdilationを指定する
    use_se_module : bool, optional
        Squeeze and Excitation Moduleを組み込むかどうか
    activation : Type[nn.Module], optional
        活性化関数(デフォルトはReLU)
    """

    def __init__(self, in_channels: int, block: BlockIdentifier,
                 layers=(2, 2, 2, 2),
                 channels=(64, 128, 256, 512),
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 use_se_module=False,
                 activation=nn.ReLU):
        super(ResNetExtractor, self).__init__()
        self.previous_dilation = dilations[0]

        if not(len(layers) == len(channels) == len(strides) == len(dilations)):
            warnings.warn(f"Length of `layres` ({len(layers)}),"
                          f" `channels` ({len(channels)}),"
                          f" `strides` ({len(strides)}) and"
                          f" `dilations` ({len(dilations)}) should have the same value.")

        # Assign `BasicBlock` or `BottleneckBlock` to `block`.
        block = _get_block(block)
        self.out_channels = channels[-1] * block.expansion
        self.in_planes = channels[0] // 2

        # stem => (N, 64, H/4, W/4)
        self.stem = nn.Sequential(
            conv3x3(in_channels, self.in_planes, stride=2),
            nn.GroupNorm(num_groups=adaptive_groups(self.in_planes), num_channels=self.in_planes),
            activation(inplace=True),
            conv3x3(self.in_planes, self.in_planes),
            nn.GroupNorm(num_groups=adaptive_groups(self.in_planes), num_channels=self.in_planes),
            activation(inplace=True),
            conv3x3(self.in_planes, channels[0]),
            nn.GroupNorm(num_groups=adaptive_groups(channels[0]), num_channels=channels[0]),
            activation(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.in_planes = channels[0]

        c, l, s, d = channels, layers, strides, dilations
        self.block1 = self._make_layer(block, c[0], l[0], s[0], d[0], use_se_module, activation)
        self.block2 = self._make_layer(block, c[1], l[1], s[1], d[1], use_se_module, activation)
        self.block3 = self._make_layer(block, c[2], l[2], s[2], d[2], use_se_module, activation)
        self.block4 = self._make_layer(block, c[3], l[3], s[3], d[3], use_se_module, activation)

        # initialize parameters
        self.apply(functools.partial(initialize_weights_, mode='fan_out', nonlinearity='relu'))

    def _make_layer(self, block: Type[ResidualBlock], planes: int, nb_layers: int,
                    stride=1, dilation=1, use_se_module=False, activation=nn.ReLU) -> nn.Module:
        """
        同一の出力マップ数のResidual blockを重ねたレイヤーを返す
        最初のブロックのみ引数strideを渡して特徴マップの空間サイズを変更する

        Parameters
        ----------
        block : Type[ResidualBlock]
            Residual blockのクラス
            BasicBlock or BottleneckBlock
        planes : int
            出力特徴マップ数
            実際の特徴マップ数はこの値にblock.expansionを乗じた値になる
        nb_layers : int
            Residual blockの数
        stride : int, optional
            最初のブロックのストライド, by default 1
        dilation : int, optional
            Convolutionのカーネル点の間隔
        use_se_module : bool, optional
            Squeeze and Excitation Moduleを組み込むかどうか
        activation : Type[nn.Module], optional
            活性化関数(デフォルトはReLU)

        Returns
        -------
        nn.Module
            Residual blockを重ねたレイヤー
        """
        layers = []
        stride = stride if dilation == 1 else 1
        # downsample block if stride > 1
        layers.append(block(self.in_planes, planes, stride,
                            dilation=self.previous_dilation,
                            use_se_module=use_se_module,
                            activation=activation))
        self.in_planes = planes * block.expansion
        for _ in range(1, nb_layers):
            layers.append(block(self.in_planes, planes,
                                dilation=dilation,
                                use_se_module=use_se_module,
                                activation=activation))

        self.previous_dilation = dilation

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)  # => (N, 64*expansion, H/2, W/2)
        out = self.maxpool(out)  # => (N, 64*expansion, H/4, W/4)
        out = self.block1(out)  # => (N, 64*expansion, H/4, W/4)
        out = self.block2(out)  # => (N, 128*expansion, H/8, W/8)
        out = self.block3(out)  # => (N, 256*expansion, H/16, W/16)
        out = self.block4(out)  # => (N, 512*expansion, H/32, W/32)

        return out


class ResNet(nn.Module):
    def __init__(self, in_channels: int, nb_classes: int, block: BlockIdentifier,
                 layers=(2, 2, 2, 2),
                 channels=(64, 128, 256, 512),
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 use_se_module=False,
                 activation=nn.ReLU):
        super(ResNet, self).__init__()
        self.resnet = ResNetExtractor(in_channels, block, layers, channels,
                                      strides, dilations, use_se_module, activation)
        self.classifier = nn.Linear(self.resnet.out_channels, nb_classes)

    def forward(self, x):
        out = self.resnet(x)  # => (N, out_channels, H//32, W//32)
        out = out.mean(dim=(2, 3))  # Global Average Pooling
        logits = self.classifier(out)

        return logits


def _get_block(block: BlockIdentifier) -> Type[ResidualBlock]:
    """Return `BasicBlock` or `BottleneckBlock`."""

    def _parse_block(block_str: str) -> Type[ResidualBlock]:
        """Parse string as block"""
        block_str = str(block_str).strip().lower()

        if block_str.startswith('basic'):
            return BasicBlock
        elif block_str.startswith('bottleneck'):
            return BottleneckBlock
        else:
            warnings.warn("Residual block type string should be `basic` or `bottleneck`."
                          " So using `BasicBlock` as default.")
            return BasicBlock

    if isinstance(block, str):
        return _parse_block(block)
    elif isinstance(block, type) and block.__name__ in ('BasicBlock' or 'BottleneckBlock'):
        return block
    else:
        raise TypeError("`block` should be str or type of `BasicBlock` or `BottleneckBlock`.")


def build_resnet(in_channels: int,
                 nb_classes: int,
                 block: BlockIdentifier = 'basic',
                 layers=(2, 2, 2, 2),
                 channels=(64, 128, 256, 512),
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 use_se_module=False,
                 activation=nn.ReLU) -> ResNet:
    """Get residual module instance."""

    return ResNet(in_channels, nb_classes, block, layers, channels,
                  strides, dilations, use_se_module, activation)


def build_resnet18(in_channels: int, nb_classes: int,
                   activation=nn.ReLU, use_se_module=False) -> ResNet:
    """ResNet18"""

    params = {
        'block': 'basic',
        'layers': (2, 2, 2, 2),
        'channels': (64, 128, 256, 512),
        'strides': (1, 2, 2, 2),
        'dilations': (1, 1, 1, 1),
        'use_se_module': use_se_module,
        'activation': activation,
    }
    return ResNet(in_channels, nb_classes, **params)


def build_resnet50(in_channels: int, nb_classes,
                   activation=nn.ReLU, use_se_module=False) -> ResNet:
    """ResNet50"""

    params = {
        'block': 'bottleneck',
        'layers': (3, 4, 6, 3),
        'channels': (64, 128, 256, 512),
        'strides': (1, 2, 2, 2),
        'dilations': (1, 1, 1, 1),
        'use_se_module': use_se_module,
        'activation': activation,
    }
    return ResNet(in_channels, nb_classes, **params)
