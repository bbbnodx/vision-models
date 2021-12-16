import torch.nn as nn
import pytorch_lightning as pl

from .base import BaseLightningClassifier, Builder
from .resnet import build_resnet as build_resnet_model
from .cvt import build_cvt_13
from .gmlp import GMLP
from .optim_builder import get_optimizer_builder, get_scheduler_builder
from vision_models.dataprocess.utils import load_dataset_info
from vision_models.functions.metrics import Error


class ImageClassifier(BaseLightningClassifier):
    def __init__(self, model: nn.Module, criterion: nn.Module,
                 optimizer_builder: Builder = None, scheduler_builder: Builder = None):

        super().__init__(criterion, optimizer_builder, scheduler_builder)
        self.model = model

    def forward(self, x):
        return self.model(x)


def _parse_data_info(args):
    json_dict = load_dataset_info(args.json_path)
    data_info = json_dict["dataset_information"]
    color_mode = data_info["color_mode"].lower().strip()

    # input channels
    if color_mode == 'rgb':
        in_channels = 3
    elif color_mode == 'gray':
        in_channels = 1
    else:
        raise ValueError(f"`color_mode` should be 'RGB' or 'gray', but `{color_mode}`. Confirm {args.json_path}.")

    # the number of classes
    nb_classes = len(data_info["labels"])

    params = {
        'in_channels': in_channels,
        'nb_classes': nb_classes,
        'image_size': data_info["img_size"][0],
    }

    return params


def build_resnet(args) -> nn.Module:
    params = _parse_data_info(args)
    act = args.activation.lower().strip()
    if act == 'relu':
        activation = nn.ReLU
    elif act == 'elu':
        activation = nn.ELU
    else:
        raise ValueError

    if args.resnet_layers == 18:
        resnet_params = {
            'block': 'basic',
            'layers': (2, 2, 2, 2),
            'channels': (64, 128, 256, 512),
            # 'channels': (32, 64, 128, 256),
            'strides': (1, 2, 2, 2),
            'dilations': (1, 1, 1, 1),
            'use_se_module': args.use_se_module,
            'activation': activation,
        }
    elif args.resnet_layers == 50:
        resnet_params = {
            'block': 'bottleneck',
            'layers': (3, 4, 6, 3),
            'channels': (64, 128, 256, 512),
            'strides': (1, 2, 2, 2),
            'dilations': (1, 1, 1, 1),
            'use_se_module': args.use_se_module,
            'activation': activation,
        }
    else:
        raise ValueError

    model = build_resnet_model(params['in_channels'], params['nb_classes'], **resnet_params)

    return model


def build_cvt(args) -> nn.Module:
    params = _parse_data_info(args)
    model = build_cvt_13(params['in_channels'], params['nb_classes'])
    return model


def build_gmlp(args) -> nn.Module:
    params = _parse_data_info(args)
    assert params['image_size'] >= args.patch_size
    model = GMLP(params['image_size'], params['in_channels'], params['nb_classes'],
                 patch_size=args.patch_size, num_blocks=args.num_blocks,
                 d_model=args.d_model, mlp_ratio=args.mlp_ratio,
                 p_dropout=0., p_droppath=0.)
    return model


def _prepare_system(args):
    model_selector = args.model.lower().strip()
    if model_selector.startswith('resnet'):
        model = build_resnet(args)
    elif model_selector.startswith('cvt'):
        model = build_cvt(args)
    elif model_selector.startswith('gmlp'):
        model = build_gmlp(args)
    else:
        raise ValueError

    criterion = nn.CrossEntropyLoss()
    optim_builder = get_optimizer_builder(args)
    sched_builder = get_scheduler_builder(args)

    return model, criterion, optim_builder, sched_builder


def build_system(args) -> pl.LightningModule:

    model, criterion, optim_builder, sched_builder = _prepare_system(args)
    system = ImageClassifier(model, criterion, optim_builder, sched_builder)

    return system


def load_model_from_checkpoint(checkpoint_path, args) -> pl.LightningModule:

    model, criterion, optim_builder, sched_builder = _prepare_system(args)
    system = ImageClassifier.load_from_checkpoint(checkpoint_path,
                                                  model=model,
                                                  criterion=criterion,
                                                  optimizer_builder=optim_builder,
                                                  scheduler_builder=sched_builder)

    return system
