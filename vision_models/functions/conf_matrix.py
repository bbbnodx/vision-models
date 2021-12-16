import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def make_confusion_matrix(pred_labels, true_labels, id_to_label=None, add_recall_precision=True):
    if id_to_label is None:
        max_label = max(pred_labels.max(), true_labels.max())
        id_to_label = list(range(max_label + 1))
        labels = id_to_label
    else:
        labels = list(range(len(id_to_label)))
    conf_matrix = confusion_matrix(true_labels, pred_labels, labels=labels)

    # 行と列どちらが正解か推論結果か判別できるようにMultiIndexで可視化する
    idx = pd.MultiIndex.from_product((['正解ラベル'], id_to_label))
    col = pd.MultiIndex.from_product((['推論ラベル'], id_to_label))

    conf_matrix_df = pd.DataFrame(conf_matrix, columns=col, index=idx)

    # recallとprecisionを算出して混同行列に追加
    if add_recall_precision:
        _add_recall_precision(conf_matrix_df)

    return conf_matrix_df


def _add_recall_precision(conf_matrix_df):
    precision = np.zeros(len(conf_matrix_df), dtype=np.float)
    for i, (_, pred_sr) in enumerate(conf_matrix_df.iteritems()):
        precision[i] = pred_sr[i] / pred_sr.sum() if pred_sr.max() > 0 else 0.0

    recall = np.zeros(len(conf_matrix_df), dtype=np.float)
    for i, (_, true_sr) in enumerate(conf_matrix_df.iterrows()):
        recall[i] = true_sr[i] / true_sr.sum() if true_sr.max() > 0 else 0.0

    conf_matrix_df.loc['precision', :] = precision
    conf_matrix_df.loc[:-1, 'recall'] = recall
