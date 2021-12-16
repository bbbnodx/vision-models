import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics

from typing import Optional, Callable

from vision_models.functions.metrics import Error

Builder = Optional[Callable]


class BaseLightningClassifier(pl.LightningModule):
    """LightningModuleの学習部分のみを記述する基底クラス"""

    criterion: nn.Module
    train_metrics: torchmetrics.Metric
    valid_metrics: torchmetrics.Metric
    optimizer_builder: Builder
    scheduler_builder: Builder

    def __init__(self, criterion: nn.Module, optimizer_builder: Builder = None, scheduler_builder: Builder = None):
        super().__init__()
        self.criterion = criterion
        self.train_metrics = Error()
        self.valid_metrics = Error()
        self.optimizer_builder = optimizer_builder
        self.scheduler_builder = scheduler_builder

    def forward(self, x):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        images, targets = batch  # => (N, C, H, W), (N,)

        logits = self(images)  # => (N, nb_classes)
        loss = self.criterion(logits, targets)

        self.log_dict({
            'train_loss': loss.detach(),
            'train_err_step': self.train_metrics(logits, targets),
        }, prog_bar=False, logger=True)

        return loss

    def training_epoch_end(self, outputs: dict) -> None:
        losses = [d['loss'] for d in outputs]
        self.log_dict({
            'train_loss_epoch': torch.stack(losses).mean().detach(),
            'train_err_epoch': self.train_metrics.compute(),
        }, logger=True)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        logits = self(images)
        self.log_dict({
            'valid_err_step': self.valid_metrics(logits, targets),
        }, logger=False)

    def validation_epoch_end(self, outputs: dict) -> None:
        self.log_dict({
            'valid_err_epoch': self.valid_metrics.compute(),
        }, logger=True)
        self.valid_metrics.reset()

    def configure_optimizers(self):
        optimizer = self.optimizer_builder(self.parameters()) if self.optimizer_builder is not None\
            else optim.Adam(self.parameters(), lr=1e-3)

        if self.scheduler_builder is not None:
            scheduler = self.scheduler_builder(optimizer)
            return [optimizer], [scheduler]

        return optimizer
