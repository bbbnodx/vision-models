from torch import Tensor
import numpy as np
import cv2

from typing import Union, Optional
from collections.abc import Sequence
import warnings


Array = Union[Tensor, np.ndarray]


def make_heatmap(weights: Array, size: Optional[Sequence] = None) -> np.ndarray:
    heatmap = weights.cpu().numpy() if isinstance(weights, Tensor) else weights

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.ndim in (1, 2)

    if heatmap.ndim == 1:
        warnings.warn("`weights` is vector. So it`s regarded as vertical weights."
                      "If you regard `weights` as horizontal, you should reshape `weights`.")
        heatmap = heatmap[:, None]

    # Resize
    if size is not None:
        assert isinstance(size, Sequence) and len(size) == 2
        heatmap = cv2.resize(heatmap, size, interpolation=cv2.INTER_LINEAR)

    heatmap = colorize_heatmap(heatmap)

    return heatmap


def colorize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """
    Colorize heatmap

    ColorMaps in OpenCV
    https://docs.opencv.org/2.4/modules/contrib/doc/facerec/colormaps.html

    Parameters
    ----------
    heatmap : np.ndarray  (H, W) or (H, W, 1)
        target weights as heatmap

    Returns
    -------
    np.ndarray  (H, W, 3)
        RGB colorized heatmap, whose values in [0, 255]
    """
    assert heatmap.squeeze().ndim == 2
    colorized = None
    colorized = cv2.normalize(heatmap, colorized, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    colorized = cv2.applyColorMap(colorized, cv2.COLORMAP_JET)

    return cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)


def compose_heatmap(base: np.ndarray, heatmap: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    assert base.shape == heatmap.shape,\
        f"Shape of `base` and `heatmap` must be same, but {base.shape} and {heatmap.shape}"
    assert 0.0 <= alpha <= 1.0

    return cv2.addWeighted(base, alpha, heatmap, beta=1.0 - alpha, gamma=0)


def make_heatmap_image(image: Array, weights: Array, alpha: float = 0.7):
    """
    weightsをheatmapに変換してimageに重ねた画像を作成して返す

    Parameters
    ----------
    image : Array
        a base image
    weights : Array
        weights as heatmap
    alpha : float, optional
        alpha ratio in composition, by default 0.7

    Returns
    -------
    np.ndarray
        a heatmap composed imageg
    """
    assert image.ndim == 3

    base: np.ndarray = image.permute(1, 2, 0).cpu().numpy() if isinstance(image, Tensor) else image
    H, W, _ = base.shape

    heatmap = make_heatmap(weights, (W, H))
    composed = compose_heatmap(base, heatmap, alpha)

    return composed
