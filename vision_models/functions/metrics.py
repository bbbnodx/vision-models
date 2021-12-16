"""
Metrics inherited torchmetrics.Metric
see https://torchmetrics.readthedocs.io/en/latest/pages/implement.html
"""

import torch
from torchmetrics import Metric

from typing import Optional, Callable, Any


class Accuracy(Metric):
    correct: torch.Tensor
    total: torch.Tensor

    def __init__(self, compute_on_step: bool = True, dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None, dist_sync_fn: Callable = None) -> None:
        super().__init__(compute_on_step, dist_sync_on_step, process_group, dist_sync_fn)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        pred_labels = logits.argmax(dim=2)
        correct = (pred_labels == targets).sum()

        self.correct += correct
        self.total += torch.tensor(len(logits))

    def compute(self) -> torch.Tensor:
        return torch.div(self.correct.float(), self.total.float())


class Error(Accuracy):
    def compute(self) -> torch.Tensor:
        return torch.sub(1., super().compute())
