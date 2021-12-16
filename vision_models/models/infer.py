import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
import numpy as np
import pandas as pd

from pathlib import Path
import warnings

from vision_models.dataprocess.utils.misc import load_dataset_info
from vision_models.functions.conf_matrix import make_confusion_matrix


def infer(model, data_loader: DataLoader, device=None):
    if isinstance(data_loader.sampler, RandomSampler):
        warnings.warn("Caution: order of data is shuffled.")
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        model.eval().to(device)
        images = []
        pred_labels, true_labels, confs = [], [], []
        for imgs, targets in data_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=-1)  # => (N, nb_classes)
            conf, pred = probs.max(dim=-1)  # => (N,), (N,)

            pred_labels.append(pred.cpu())
            true_labels.append(targets.cpu())
            confs.append(conf.cpu())
            images.append(imgs.cpu())

        pred_labels = torch.cat(pred_labels, dim=0).numpy()
        true_labels = torch.cat(true_labels, dim=0).numpy()
        confs = torch.cat(confs, dim=0).numpy()

        correct = (pred_labels == true_labels).astype(np.uint8)

        images = torch.cat(images, dim=0).permute(0, 2, 3, 1).numpy()  # => (N, H, W, C)

        results = {
            'image': images,
            'pred_labels': pred_labels,
            'true_labels': true_labels,
            'correct': correct,
            'confidence': confs,
        }

        return results


def build_inference(result: dict, args, datatype='test', return_conf_matrix=False):
    assert datatype in ('val', 'test')
    json_path = Path(args.json_path)
    json_dict = load_dataset_info(json_path)
    data_info = json_dict['dataset_information']
    csv_path = Path(data_info['root_path']).resolve() / data_info[datatype]
    id_to_label = data_info['labels']

    result_df: pd.DataFrame = pd.read_csv(csv_path)
    result_df['pred'] = [id_to_label[idx] for idx in result['pred_labels']]
    result_df['correct'] = result['correct']
    result_df['conf_form'] = result['confidence']

    if return_conf_matrix:
        conf_matrix_df = make_confusion_matrix(result['pred_labels'],
                                               result['true_labels'],
                                               id_to_label=id_to_label,
                                               add_recall_precision=True)

        return result_df, conf_matrix_df

    return result_df
