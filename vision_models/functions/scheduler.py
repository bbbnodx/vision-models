"""
Custom Schedulers for PyTorch
"""

from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau


class WarmupScheduler(_LRScheduler):
    """
    Wrapper scheduler for graduall warm-up learning rate in optimizer.
    In warmup epochs, lr = base_lr * E_c / E_w
        E_c: current epoch
        E_w: warmup epoch
    After warmup, lr decays with the specific scheduler, or non scheduler

    "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
    Goyal et al. [NIPS 2017]
    See https://arxiv.org/pdf/1706.02677.pdf

    Parameters
    ----------
    optimizer : optim.Optimizer
        Wrapped optimizer
    warmup_epoch : int
        The epoch when lr gradually reachs to base lr
    scheduler : _LRScheduler, optional
        After warmup_epoch use this scheduler, by default None
    """

    def __init__(self, optimizer, warmup_epoch=5, scheduler=None):
        assert warmup_epoch > 0, "warmup_epoch should be > 0."
        self.warmup_epoch = warmup_epoch
        self.scheduler = scheduler
        if scheduler is not None:
            assert isinstance(self.scheduler, _LRScheduler)
            self.__class__.__name__ = self.scheduler.__class__.__name__ + 'Wrapped' + self.__class__.__name__
        super(WarmupScheduler, self).__init__(optimizer)

    @property
    def is_warmup(self) -> bool:
        return self.last_epoch <= self.warmup_epoch

    @property
    def warmup_rate(self) -> float:
        return self.last_epoch / self.warmup_epoch if self.is_warmup else 1.0

    def get_lr(self):
        """refferenced in step()"""
        if self.is_warmup:
            return [base_lr * self.warmup_rate for base_lr in self.base_lrs]
        elif self.scheduler is not None:
            return self.scheduler.get_lr()
        else:
            return self.base_lrs

    def step(self, epoch=None, metrics=None):
        if self.is_warmup or self.scheduler is None:
            super(WarmupScheduler, self).step(epoch)
        else:
            epoch = epoch - self.warmup_epoch if epoch is not None else epoch
            if isinstance(self.scheduler, ReduceLROnPlateau):
                assert metrics is not None,\
                    "If scheduler is `ReduceLROnPlateau`, an argument `metrics` is necessary."
                self.scheduler.step(metrics, epoch)
            else:
                self.scheduler.step(epoch)
