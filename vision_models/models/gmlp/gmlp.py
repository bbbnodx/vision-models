"""
"Pay Attention to MLPs"
Hanxiao Liu, Zihang Dai, David R. So, Quoc V. Le
see https://arxiv.org/abs/2105.08050
"""

from torch import Tensor
import torch.nn as nn

from typing import Optional, Type
from functools import partial

from .drop_layers import DropPath
from .patch_embed import PatchEmbed


class SpatialGatingUnit(nn.Module):
    """ Spatial Gating Unit (SGU)
    in - split - norm - proj - hadamard - out
           └─────────────┘

    in:  (N, L, dim)
    out: (N, L, dim // 2)
    """
    def __init__(self, dim: int, num_patches: int,
                 norm_layer: Type[nn.Module] = nn.LayerNorm):
        super().__init__()
        assert dim & 1 == 0, f"`dim` should be even, but `{dim}`."
        self.gate_dim = dim // 2
        self.norm = norm_layer(self.gate_dim)
        self.proj = nn.Linear(num_patches, num_patches)

    def forward(self, x: Tensor) -> Tensor:
        u, v = x.chunk(2, dim=-1)  # => (N, num_patches, gate_dim), (N, num_patches, gate_dim)
        v = self.norm(v)
        v = self.proj(v.transpose(-1, -2))  # => (N, gate_dim, num_patches)

        return u * v.transpose(-1, -2)  # => (N, num_patches, gate_dim)


class GatedMLPCore(nn.Module):
    """ gMLP core
    in - proj - gelu - dropout - SGU - proj - dropout - out

    in:  (N, L, in_features)
    out: (N, L, out_features)
    """

    def __init__(self, in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 activation: Type[nn.Module] = nn.GELU,
                 spatial_gating_unit: Optional[Type[nn.Module]] = None,
                 p_dropout: float = 0.):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features if hidden_features is not None else in_features
        self.out_features = out_features if out_features is not None else in_features

        self.fc1 = nn.Linear(self.in_features, self.hidden_features)
        self.act = activation()
        if spatial_gating_unit is not None:
            assert self.hidden_features & 1 == 0, f"`hidden_features` should be even, but `{self.hidden_features}`."
            self.gate = spatial_gating_unit(self.hidden_features)
            gate_features = self.hidden_features // 2
        else:
            self.gate = nn.Identity()
            gate_features = self.hidden_features
        self.fc2 = nn.Linear(gate_features, self.out_features)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.gate(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x  # => (N, out_features)


class GatedMLPBlock(nn.Module):
    """
    gMLP block
    in - norm - gMLP - dropout - add - out
       └───────────────┘

    in:  (N, L, d_model)
    out: (N, L, d_model)
    """

    def __init__(self, d_model: int, num_patches: int,
                 mlp_ratio: int = 4,
                 norm_layer: Type[nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 activation: Type[nn.Module] = nn.GELU,
                 p_dropout: float = 0.,
                 p_droppath: float = 0.):
        super().__init__()
        d_ffn = int(d_model * mlp_ratio)
        self.norm = norm_layer(d_model)
        sgu = partial(SpatialGatingUnit, num_patches=num_patches)
        self.mlp_channels = GatedMLPCore(d_model, hidden_features=d_ffn, activation=activation,
                                         spatial_gating_unit=sgu, p_dropout=p_dropout)
        self.drop_path = DropPath(p_droppath)

    def forward(self, x: Tensor) -> Tensor:
        out = self.norm(x)  # => (N, num_patches, d_model)
        out = self.mlp_channels(out)
        out = self.drop_path(out)
        out = x + out

        return out  # => (N, num_patches, d_model)


class GMLP(nn.Module):
    """
    gMLP
    in - patch_embed - gMLP_block * num_blocks - norm - GAP - head - out

    in:  (N, C, H, W)
    out: (N, num_classes)
    """
    def __init__(self, img_size, in_channels, num_classes, patch_size=16, num_blocks=30,
                 d_model=128, mlp_ratio=6, p_dropout=0., p_droppath=0.):
        super().__init__()
        self.num_classes = num_classes

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.embedding = PatchEmbed(img_size, patch_size, in_channels,
                                    embed_dim=d_model, norm_layer=None)
        self.blocks = nn.Sequential(*[
            GatedMLPBlock(d_model,
                          num_patches=self.embedding.num_patches,
                          mlp_ratio=mlp_ratio,
                          norm_layer=norm_layer,
                          activation=nn.GELU,
                          p_dropout=p_dropout,
                          p_droppath=p_droppath)
            for _ in range(num_blocks)])

        self.norm = norm_layer(d_model)
        self.head = nn.Linear(d_model, self.num_classes)

        self.initialize_weights_()

    def initialize_weights_(self):
        for n, m in self.named_modules():
            _initialize_weights_(m, n)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # => (N, num_patches, d_model)
        x = self.blocks(x)  # => (N, num_patches, d_model)
        x = self.norm(x)
        x = x.mean(dim=1)  # => (N, d_model)
        x = self.head(x)  # => (N, num_classes)

        return x


def _initialize_weights_(module, name: str, head_bias: float = 0.):
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.endswith('proj'):
            nn.init.normal_(module.weight, std=1e-4)
            nn.init.ones_(module.bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias'):
                if 'mlp' in name:
                    nn.init.normal_(module.bias, std=1e-6)
                else:
                    nn.init.zeros_(module.bias)

    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
