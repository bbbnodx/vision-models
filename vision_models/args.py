"""
Usage:
    param = {'lr': 1e-3, 'batch_size': 8, 'epochs': 200}
    args = Args(**params)

    system = models.build_system(args)
"""

from typing import NamedTuple, Union, Optional


class Args(NamedTuple):
    batch_size: int = 4
    nb_classes: int = 17
    epochs: int = 100
    ckpt_interval: int = 20
    eval_interval: int = 1
    device: str = 'cuda'
    log_dir: str = '../logs'
    model_dir: str = '../models'
    project: str = 'default'
    model: str = 'resnet'
    version: Union[str, int, None] = None
    # ResNet
    resnet_layers: int = 18
    use_se_module: bool = False
    activation: str = 'ReLU'
    # gMLP
    patch_size: int = 16
    num_blocks: int = 30
    d_model: int = 128
    mlp_ratio: int = 6
    # Dataset information
    json_path: str = '../data/dataset_info_1.json'
    # Optimizer
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = 'adaberief'
    momentum: float = 0.9  # Momentum
    final_lr: float = 0.1  # AdaBound
    # Scheduler
    scheduler: Optional[str] = None
    warmup: bool = True
    warmup_epoch: int = 5
    # Inference
    model_weights: str = 'stacking-modify-CvT-210615-epoch=129-valid_acc_epoch=1.00.ckpt'
