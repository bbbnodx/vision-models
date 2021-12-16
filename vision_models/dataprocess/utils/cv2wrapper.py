"""
一般的な画像処理関数群
OpenCVの入出力ヘルパー関数

cv2_imread(img_path, mode='rgb', img_size=None) -> np.ndarray
    CV2による読み込み、カラーをRGBに変換、リサイズを実行してNumPy配列を返すヘルパー関数
cv2_imwrite(img_path, img, mode='rgb') -> None
    NumPy配列画像をBGRにカラー変換して画像ファイルとして保存するヘルパー関数

"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

import warnings

from .misc import Pathlike


def _cv2_converter_selector(mode: str, read=True):
    """
    modeの値に応じたCV2の色空間変換フラグを返す
    modeがconverterのキーに存在しない場合は、Noneを返す
    """

    mode = mode.lower()
    read_converter = {
        'rgb': cv2.COLOR_BGR2RGB,
        'gray': cv2.COLOR_BGR2GRAY,
    }
    write_converter = {
        'rgb': cv2.COLOR_RGB2BGR,
        'gray': cv2.COLOR_GRAY2BGR,
    }
    converter = read_converter if read else write_converter
    if mode not in converter:
        warnings.warn("'mode' should be 'RGB' or 'gray'. Image color won't convert.")

    return converter.get(mode, None)


def cv2_imread(img_path: Pathlike, mode='rgb', img_size=None) -> np.ndarray:
    """CV2による読み込み、カラーをRGBに変換、リサイズを実行してNumPy配列を返すヘルパー関数"""

    color_converter = _cv2_converter_selector(mode, read=True)
    img = cv2.imread(str(img_path))
    assert img is not None, f"`cv2.imread(img_path)` couldn't read an image. Confirm `img_path`: {img_path}"
    img = cv2.cvtColor(img, color_converter) if color_converter is not None else img
    if img_size is not None:
        img = cv2.resize(img, dsize=tuple(img_size), interpolation=cv2.INTER_LANCZOS4)
    return img


def cv2_imwrite(img_path: Pathlike, img, mode='rgb') -> bool:
    """NumPy配列画像をBGRにカラー変換して画像ファイルとして保存するヘルパー関数"""

    assert img.ndim in (2, 3), "'img' might not be an image."
    color_converter = _cv2_converter_selector(mode, read=False)
    img = cv2.cvtColor(img, color_converter) if color_converter is not None else img
    ret = cv2.imwrite(str(img_path), img)
    if not ret:
        warnings.warn(f"`cv2.imwrite(img_path, img)` couldn't save an image. Confirm `img_path`: {img_path}")
    return ret


def cv2_imshow(img: np.ndarray, is_RGB=True) -> None:
    """OpenCVの画像をmatplotlibで正しく表示するためのヘルパー関数"""

    if img.ndim == 3:  # カラー画像
        rgb = img if is_RGB else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb)
    elif img.ndim == 2:  # グレースケール
        plt.imshow(img, cmap=plt.cm.gray)
    plt.show()
