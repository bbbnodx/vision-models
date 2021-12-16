"""
Normalize and denormalize image functions

c.f. Mean Subtraction
"""

import torch
from torch import Tensor
import torchvision.transforms.functional as tvf
import numpy as np

from typing import Sequence, Union

NormParam = Union[float, Sequence[float]]


def normalize(image: Union[np.ndarray, Tensor], mean: NormParam, std: NormParam, inplace=False):
    """
    Normalize image
        pix'_{i, j} = (pix_{i, j} - mean) / std

    Parameters
    ----------
    image : np.ndarray | Tensor
        np.ndarray or Tensor image to normalize
    mean : float | Sequence[float]
        mean of pixel values
        scalar, channel-wise, or pixel-wise
    std : float | Sequence[float]
        std of pixel values
        scalar, channel-wise, or pixel-wise
    inplace : bool, optional
        if True, image changes destructively, by default False

    Returns
    -------
    same of image type
        normalized image
    """

    if isinstance(image, np.ndarray):
        return _normalize_numpy_image(image, mean, std, inplace)
    elif isinstance(image, Tensor):
        return tvf.normalize(image, mean, std, inplace=inplace)
    else:
        raise ValueError(f"Image type to normalize should be np.ndarray or Tensor, but `{type(image)}`")


def denormalize(image: Union[np.ndarray, Tensor], mean: NormParam, std: NormParam, inplace=False):
    """
    Denormalize image
        pix_{i, j} = pix'_{i, j} * std + mean

    Parameters
    ----------
    image : np.ndarray | Tensor
        np.ndarray or Tensor image to denormalize
    mean : float | Sequence[float]
        mean of pixel values
        scalar, channel-wise, or pixel-wise
    std : float | Sequence[float]
        std of pixel values
        scalar, channel-wise, or pixel-wise
    inplace : bool, optional
        if True, image changes destructively, by default False

    Returns
    -------
    same of image type
        denormalized image
    """

    if isinstance(image, np.ndarray):
        return _denormalize_numpy_image(image, mean, std, inplace)
    elif isinstance(image, Tensor):
        return _denormalize_tensor_image(image, mean, std, inplace)
    else:
        raise ValueError(f"Image type to denormalize should be np.ndarray or Tensor, but `{type(image)}`")


def float_to_uint8(img):
    if isinstance(img, np.ndarray):
        assert img.dtype == np.float32
        return (img * 255).astype(np.uint8)

    if isinstance(img, Tensor):
        assert img.dtype == torch.float
        return (img * 255).to(dtype=torch.uint8)

    raise ValueError


def uint8_to_float(img):
    if isinstance(img, np.ndarray):
        assert img.dtype == np.uint8
        return img.astype(np.float32) / 255.

    if isinstance(img, Tensor):
        assert img.dtype == torch.uint8
        return img.float() / 255.

    raise ValueError


def _verify_std(std, dtype) -> None:
    if (std == 0).any():
        raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")


def _normalize_numpy_image(npimg: np.ndarray, mean: NormParam, std: NormParam, inplace=False) -> np.ndarray:
    assert isinstance(npimg, np.ndarray)
    assert npimg.ndim >= 3

    img = npimg.copy() if not inplace else npimg
    dtype = npimg.dtype

    mean_np = np.array(mean, dtype=dtype)
    std_np = np.array(std, dtype=dtype)
    _verify_std(std_np, dtype)

    img[...] = (img - mean_np) / std_np

    return img


def _denormalize_numpy_image(npimg: np.ndarray, mean: NormParam, std: NormParam, inplace=False) -> np.ndarray:
    assert isinstance(npimg, np.ndarray)
    assert npimg.ndim >= 3

    img = npimg.copy() if not inplace else npimg
    dtype = npimg.dtype

    mean_np = np.array(mean, dtype=dtype)
    std_np = np.array(std, dtype=dtype)
    _verify_std(std_np, dtype)

    img[...] = img * std_np + mean_np

    return img


def _denormalize_tensor_image(tensor_img: Tensor, mean: NormParam, std: NormParam, inplace=False) -> Tensor:
    assert isinstance(tensor_img, Tensor)
    assert tensor_img.ndim >= 3

    img = tensor_img.clone() if not inplace else tensor_img
    dtype, device = tensor_img.dtype, tensor_img.device

    mean_t = torch.as_tensor(mean, dtype=dtype, device=device)
    std_t = torch.as_tensor(std, dtype=dtype, device=device)
    _verify_std(std_t, dtype)

    if mean_t.ndim == 1:
        mean_t = mean_t.reshape(-1, 1, 1)
    if std_t.ndim == 1:
        std_t = std_t.reshape(-1, 1, 1)

    img.mul_(std_t).add_(mean_t)

    return img
