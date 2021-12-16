import numpy as np
import torch
import torchvision as tv

from .tensor_image import convert_tensor_to_npimg


def make_grid_npimage(images, nb_columns=8, insert_border=True, border_color=(0, 0, 0)) -> np.ndarray:
    """
    NumPy配列画像のバッチをグリッド画像に変換して返す関数
    グレースケール画像もRGBに変換する

    Parameters
    ----------
    images : numpy.ndarray (N, H, W, C) or (N, H, W)
        画像のバッチデータ
    nb_columns : int, optional
        1行当たりの画像数
    insert_border : bool, optional
        画像間に境界線を入れるかどうか
    border_color : tuple or int, optional
        境界線の色をRGB値で指定する
        デフォルトは黒

    Returns
    -------
    numpy.ndarray (H_grid, W_grid, 3)
        グリッド画像
        H_grid = H * (ceil(N/nb_columns)) + insert_border * (ceil(N/nb_columns) - 1)
        W_grid = W * nb_columns + insert_border * (nb_columns - 1)
            (ceil関数：ceil(x/y) = x//y if x%y == 0 else x//y + 1)
    """
    assert images.ndim in (3, 4), "'images' might not be batch of images."
    RGB_CHANNELS = 3
    if images.ndim == 3:  # グレースケールの場合
        images = images[:, :, :, None].repeat(RGB_CHANNELS, axis=3)
    N, H, W, _ = images.shape
    # データ数を列数で割り、行数(切り捨て)と端数を取得
    max_rows, mod = divmod(N, nb_columns)
    grid = np.vstack([np.hstack(images[row * nb_columns:(row + 1) * nb_columns])
                      for row in range(max_rows)]) if max_rows > 0 else None
    # 端数処理
    #   0埋めした1行のベース画像を作成し、端数分の画像を左詰めで上書きする
    if mod > 0:
        mod_row = np.zeros((H, W * nb_columns, RGB_CHANNELS), dtype=images.dtype)
        mod_row[:, :W * mod] = np.hstack(images[max_rows * nb_columns:])
        grid = np.vstack([grid, mod_row]) if grid is not None else mod_row
        max_rows += 1  # 境界線の挿入位置に用いる

    # 境界線の挿入
    # Hの倍数の行、Wの倍数の列が境界線上なので、引数のRGB値を挿入する
    if insert_border:
        grid = np.insert(grid, range(H, H * max_rows, H), border_color, axis=0)
        grid = np.insert(grid, range(W, W * nb_columns, W), border_color, axis=1)

    return grid


def make_grid_tensor(tensor_img_batch: torch.Tensor, mean, std, nrow=8, padding=2, pad_value=0):
    """
    アノテーション画像バッチをグリッド画像に変換し、
    NumPy配列のRGB画像として返す

    Parameters
    ----------
    tensor_img_batch : torch.Tensor (N, C, H, W)
        入力画像バッチ
    mean : int or list, optional
        正規化パラメータ, by default 0
    std : int or list, optional
        正規化パラメータ, by default 1
    nrow : int, optional
        グリッド画像の1行あたりの画像数, by default 8
    padding : int, optional
        画像ごとにパディングするピクセル数, by default 2
    pad_value : float, optional
        パディングの色, by default 0

    Returns
    -------
    np.ndarray (H * (N//nrows + 1), W * nrows, 3)
        グリッド画像
    """

    assert tensor_img_batch.ndim == 4
    img_grid = tv.utils.make_grid(tensor_img_batch, nrow=nrow, padding=padding, pad_value=pad_value)
    npgrid = convert_tensor_to_npimg(img_grid, mean, std)

    return npgrid
