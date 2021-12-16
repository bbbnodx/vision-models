import torch
from torch.utils.data import DataLoader

from pathlib import Path
import warnings
from typing import Tuple

from . import models
from . import dataprocess as dp
from vision_models.args import Args


def recognize(image_paths: list, batch_size: int = 32) -> Tuple[list, list]:
    """
    Recognize
    画像パスのリストを受け取り、推論結果と確信度を返す

    Parameters
    ----------
    image_paths : list [pathlike]
        画像パスのリスト
    batch_size : int, optional
        バッチサイズ, by default 32

    Returns
    -------
    list, list
        (推論結果リスト, 確信度リスト)のタプル
        画像が存在しなかった場合、推論結果には空文字列、確信度は0.0が入る
    """

    if len(image_paths) == 0:
        warnings.warn("`image_paths` is empty. So `crnn_ocr.recognize` returns empty lists.")
        return [], []

    CURDIR = Path(__file__).resolve().parent
    json_path = CURDIR / './data/config/dataset_info_1.json'
    model_dir = CURDIR / './data/models'

    arg_params = {
        'batch_size': batch_size,
        'nb_classes': 17,
        # params of gMLP
        'patch_size': 4,
        'num_blocks': 30,
        'd_model': 128,
        'mlp_ratio': 6,

        'device': 'cuda',
        'json_path': json_path,
        'model_dir': model_dir,
        'model_weights': 'stacking-modify-CvT-210615-epoch=129-valid_acc_epoch=1.00.ckpt',
    }

    args = Args(**arg_params)
    model_path = Path(args.model_dir) / args.model_weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert json_path.is_file()
    assert model_path.is_file()

    # Get dataloader for inference
    json_dict = dp.load_dataset_info(json_path)
    transform = dp.get_transforms('test', json_dict=json_dict)
    pseudo_labels = [0] * len(image_paths)
    dataset = dp.ImageDataset(image_paths, pseudo_labels, transform=transform, device=device)
    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=0,
                             drop_last=False,
                             pin_memory=(device.type != 'cuda'))

    # Initialize the model and load weights
    resnet = models.build_resnet(args)
    cvt = models.build_cvt(args)
    gmlp = models.build_gmlp(args)
    system = models.load_stacking_from_checkpoint(checkpoint_path=str(model_path),
                                    pretrained_models=[resnet, gmlp, cvt],
                                    args=args)

    # Inference
    results = models.infer(system, data_loader, device=device)
    guesses = results['pred_labels']
    confidences = results['confidence']

    # GPUのメモリを開放する
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return guesses.tolist(), confidences.tolist()
