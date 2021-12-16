"""Export visualized attention to HTML"""

import numpy as np
import cv2

import base64
from pathlib import Path
from typing import Union, Sequence, Callable

Pathlike = Union[str, Path]


def _cv_to_base64(npimg: np.ndarray) -> str:
    """NumPy配列の画像をBase64エンコードして文字列として返す
    OpenCVの関数を用いる関係上、npimgの色空間はBGRを前提としていることに注意
    """

    _, encoded = cv2.imencode(".jpg", npimg)
    img_b64 = base64.b64encode(encoded).decode("ascii")

    return img_b64


def _id_to_label(idx: int) -> str:
    assert idx in (0, 1), f"An anomaly label is expected to be 0 or 1, but `{idx}`."
    id_to_label = ['正常', '異常']
    return id_to_label[idx]


def export_result_to_html(html_path: Pathlike, results_dict: dict,
                          id_to_label: Callable[[int], str] = _id_to_label) -> None:
    """
    推論結果をHTMLにエクスポートする
    画像をbase64にエンコードしてHTMLに埋め込み、
    HTML単独で画像と推論結果を比較できるようにする

    Parameters
    ----------
    html_path : pathlike
        出力先HTMLファイルパス(上書きされる)
    results_dict : dict
        'image': Sequence[np.ndarray (H, W, C)]
        'attention_image': Sequence[np.ndarray (H, W, C)]
        'pred_anomaly': np.ndarray (N,),
        'true_anomaly': np.ndarray (N,),
    id_to_label : Callable[[int], str]
        ラベルインデックスからラベル文字列に変換する関数
    """
    base_images: Sequence[np.ndarray] = results_dict['image']
    attn_images: Sequence[np.ndarray] = results_dict['attention_image']
    pred_labels: Sequence[int] = results_dict['pred_anomaly']
    true_labels: Sequence[int] = results_dict['true_anomaly']
    assert len(base_images) == len(attn_images) == len(pred_labels) == len(true_labels)

    # データ数の桁数を取得
    nb_data = len(base_images)
    digits = len(str(nb_data))  # indexのゼロ埋めのため桁数を取得

    # html_header and style
    html = """<html lang="ja">
    <head><meta charset="utf-8">
        <style type="text/css">
            table {
                margin: auto;
            }
            table th {
                padding: 10px;
            }
            table td {
                padding: 3px 20px;
            }
            .image {
                padding: 0;
            }
            .filename__body, .index__body, .correct__dody {
                text-align: center;
            }
            .image__body {
                width: 500px;
                text-align: center;
            }
            .guess__body, .target__body {
                text-align: center;
                margin: auto;
            }
        </style>
    </head>\n"""
    # contents-table
    html += '<body><table border="1"><tbody>\n'

    # table header
    html += '<tr><th>Index</th><th>Base Image</th><th>Attention Map</th>'\
            '<th>True Label</th><th>Guess Label</th><th>Correct</th>'
    html += '</tr>\n'

    # table body
    for idx, (img, attn, t_label, p_label)\
            in enumerate(zip(base_images, attn_images, true_labels, pred_labels)):
        # Data index(+1 to convert zero-based into one-based)
        html += f'<tr> <td class="index"> <div class="index__body">{idx+1:0{digits}d} </div> </td>'

        # 画像をBASE64エンコードしてHTMLに埋め込む
        base_img_b64 = _cv_to_base64(img[:, :, ::-1])
        html += f'<td class="image"> <img class="image__body" src="data:image/png;base64,{base_img_b64} "></td>'
        attn_img_b64 = _cv_to_base64(attn[:, :, ::-1])
        html += f'<td class="image"> <img class="image__body" src="data:image/png;base64,{attn_img_b64} "></td>'
        del base_img_b64, attn_img_b64

        # true/pred label and judgement
        html += f'<td class="target"> <div class="target__body""> {id_to_label(t_label)} </div> </td>'
        html += f'<td class="guess"> <div class="guess__body""> {id_to_label(p_label)} </div> </td>'
        correct = '○' if t_label == p_label else '×'
        html += f'<td class="correct"> <div class="correct__dody ""> {correct} </div> </td>'
        # The end of table row
        html += '</tr>\n'

    # Write to html file
    with open(html_path, 'w', encoding="utf8") as f:
        f.write(html)
        print(f"Complete export to HTML: {html_path}")
