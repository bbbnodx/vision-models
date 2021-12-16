import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self, p_droppath: float = 0.):
        super().__init__()
        self.p_droppath = p_droppath

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.p_droppath, self.training)


def drop_path(x: torch.Tensor, p_droppath: float = 0., training: bool = False) -> torch.Tensor:
    if p_droppath == 0. or not training:
        return x

    # 微分可能な手順でdropoutのmaskを生成
    keep_prob = 1. - p_droppath
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
    mask = (keep_prob + random_tensor).floor_()
    out = x.div(keep_prob) * mask

    return out
