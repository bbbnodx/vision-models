import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, padding_mode='zeros'):
    """3x3畳み込み層のヘルパー関数"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation,
                     padding_mode=padding_mode)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1畳み込み層のヘルパー関数(主にチャンネル数拡張に用いる)"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3_norm_elu(in_planes, planes, stride=1, padding_mode='zeros', momentum=0.1):
    """conv3x3 - norm - eluのブロックを作成して返すヘルパー関数"""

    layers = nn.Sequential(
        conv3x3(in_planes, planes, stride=stride, padding_mode=padding_mode),
        nn.BatchNorm2d(planes, momentum=momentum),
        nn.ELU(inplace=True),
    )

    return layers


def conv1x1_norm_elu(in_planes, planes, stride=1, momentum=0.1):
    """conv3x3 - norm - eluのブロックを作成して返すヘルパー関数"""

    layers = nn.Sequential(
        conv1x1(in_planes, planes, stride=stride),
        nn.BatchNorm2d(planes, momentum=momentum),
        nn.ELU(inplace=True),
    )

    return layers
