from torch import Tensor
import torch.nn as nn

from typing import Optional, Union, Iterable, Type
from functools import reduce
import operator

from .helper import to_2tuple


class PatchEmbed(nn.Module):
    """Separate images to N x N patches, and convert each of them to embedding vectors.

    in:  (N, C, H, W)
    out: (N, num_patches, embed_dim)
         (num_patches <= H // patch_size * W // patch_size)
    """

    def __init__(self, img_size: Union[int, Iterable[int]],
                 patch_size: Union[int, Iterable[int]],
                 in_channels: int = 3,
                 embed_dim: int = 768,
                 norm_layer: Optional[Type[nn.Module]] = None):

        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (i_size // p_size for i_size, p_size in zip(self.img_size, self.patch_size))
        self.num_patches = reduce(operator.mul, self.grid_size)

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        assert self.img_size == (H, W), \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)  # => (N, embed_dim, grid_size[0], grid_size[1])
        x = x.flatten(start_dim=2).transpose(1, 2)  # => (N, num_patches, embed_dim)
        x = self.norm(x)

        return x
