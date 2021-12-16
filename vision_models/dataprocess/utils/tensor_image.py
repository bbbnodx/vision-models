import torch
import numpy as np

from .cv2wrapper import cv2_imshow
from .normalize import denormalize


def convert_tensor_to_npimg(tensor_img: torch.Tensor, mean=0.0, std=1.0) -> np.ndarray:
    """
    Tensor画像をNumPy配列形式に変換するヘルパー関数
    Tensor画像は次式で正規化されており、正規化パラメータを用いて元のピクセル値に復号する
        pix'_{i, j} = (pix_{i, j} - mean) / std


    Parameters
    ----------
    tensor_img : torch.Tensor  (N, C, H, W) or (C, H, W)
        Tensor image
    mean : float or Sequence[float], optional
        Normalization parameter, by default 0.0
    std : float or Sequence[float], optional
        Normalization parameter, by default 1.0

    """

    assert tensor_img.ndim in (3, 4)
    denormalized = denormalize(tensor_img, mean, std)
    npimg: np.ndarray = (denormalized.cpu().numpy() * 255).astype(np.uint8)

    if npimg.ndim == 4:
        return npimg.transpose(0, 2, 3, 1)
    elif npimg.ndim == 3:
        return npimg.transpose(1, 2, 0)
    else:
        raise ValueError


def tensor_imshow(tensor_img: torch.Tensor, mean=0.0, std=1.0) -> None:
    """
    Transformで変換したTensor画像をmatplotlibで描画するヘルパー関数
    Tensor画像は次式で正規化されており、正規化パラメータを用いて元のピクセル値に復号する
        pix'_{i, j} = (pix_{i, j} - mean) / std

    Parameters
    ----------
    tensor_img : torch.Tensor  (C, H, W)
        Tensor image
    mean : float, optional
        Normalized parameter, by default 0.0
    std : float, optional
        Normalized parameter, by default 1.0
    """

    assert tensor_img.ndim == 3
    npimg = convert_tensor_to_npimg(tensor_img, mean, std).squeeze()
    cv2_imshow(npimg)
    del npimg
