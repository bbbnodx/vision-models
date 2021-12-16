import torch
import torch.nn as nn

from pathlib import Path
from functools import lru_cache
import datetime


def initialize_weights_(module, mode='fan_in', nonlinearity='relu'):
    """
    Initialize weights with `kaiming`.
    If you'd like to apply this function for every submodule, as follows:
        model.apply(initialize_weights_)
    """
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=nonlinearity)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, mode=mode, nonlinearity=nonlinearity)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


@lru_cache(maxsize=32)
def adaptive_groups(planes: int, channels_per_group=16):
    """
    Group Normalizationの引数のグループ数を
    出力チャンネル数に応じて適応的に計算して返す
        adaptive_groups = ceil(planes / channels_per_groups)

    Parameters
    ----------
    planes : int
        出力チャンネル数
    channels_per_group : int, optional
        1グループあたりのチャンネル数
        原著論文に倣い、16をデフォルト値とする

    Returns
    -------
    int
        グループ数
    """
    return -(-planes // channels_per_group)


def save_model_weights(model: nn.Module, model_path) -> None:
    """Helper function to save model weights."""

    torch.save(model.state_dict(), model_path, pickle_protocol=4)


def load_model_weights(model: nn.Module, model_path) -> None:
    """Helper function to load model weights."""

    model.load_state_dict(torch.load(model_path))


def build_model_path_automatically(args, epoch=None) -> Path:
    """
    モデルの重みのファイル名を所定のフォーマットで生成する
    "MODEL_DIR/PROJECT_MODELNAME_YYMMDD_HHMM_epochN.pth"
    """
    SEP = '_'
    EXT = '.pth'
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")  # YYMMDD_HHMM
    filename_parts = [args.project, args.model, timestamp]
    if epoch is not None:
        filename_parts.append(f"epoch{epoch}")
    model_path = Path(args.model_dir) / (SEP.join(filename_parts) + EXT)

    return model_path


def save_model_weights_automatically(model: nn.Module, args, epoch=None) -> None:
    """
    モデルの重みを所定のフォーマットのファイル名で保存する
    "MODEL_DIR/PROJECT_MODELNAME_YYMMDD_HHMM_epochN.pth"
    """
    model_path = build_model_path_automatically(args, epoch)
    save_model_weights(model, model_path)
