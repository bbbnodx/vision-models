import torch
import torch.nn as nn
import pytorch_lightning as pl

from typing import List

from .base import BaseLightningClassifier, Builder
from .optim_builder import get_optimizer_builder, get_scheduler_builder


class StackingSystem(BaseLightningClassifier):
    def __init__(self, pretrained_models: nn.ModuleList, nb_classes: int, criterion: nn.Module,
                 optimizer_builder: Builder = None, scheduler_builder: Builder = None, freeze_params=True):

        super().__init__(criterion, optimizer_builder, scheduler_builder)
        self.pretrained_models = pretrained_models

        # freeze parameters of pretrained models
        if freeze_params:
            for model in self.pretrained_models:
                model.requires_grad_(False)

        self.fc = nn.Linear(nb_classes * len(self.pretrained_models), nb_classes)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        logits = torch.cat([model(x) for model in self.pretrained_models], dim=1)
        logits = self.fc(logits)
        return logits


def build_stacking_system(pretrained_models: List[nn.Module], args) -> pl.LightningModule:
    models = nn.ModuleList(pretrained_models)
    criterion = nn.CrossEntropyLoss()
    optim_builder = get_optimizer_builder(args)
    sched_builder = get_scheduler_builder(args)

    system = StackingSystem(models, args.nb_classes, criterion, optim_builder, sched_builder)

    return system


def load_stacking_from_checkpoint(checkpoint_path, pretrained_models: List[nn.Module], args) -> pl.LightningModule:
    models = nn.ModuleList(pretrained_models)
    criterion = nn.CrossEntropyLoss()
    optim_builder = get_optimizer_builder(args)
    sched_builder = get_scheduler_builder(args)

    system = StackingSystem.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                                 pretrained_models=models,
                                                 nb_classes=args.nb_classes,
                                                 criterion=criterion,
                                                 optimizer_builder=optim_builder,
                                                 scheduler_builder=sched_builder)
    return system
