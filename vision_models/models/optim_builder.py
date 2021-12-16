"""
Optimizer, Schedulerを生成する関数(builder)を返す関数群
"""

import torch.optim as optim
from adabelief_pytorch import AdaBelief
from adabound import AdaBound

from functools import partial
from typing import Callable, Optional

from vision_models.functions.scheduler import WarmupScheduler


def get_optimizer_builder(args) -> Optional[Callable]:
    """
    model.parameters()を引数としてoptimizerを返す関数を作成して返す
    optimizerの指定がない場合はNoneを返す

    Usage:
        optmi_builder = get_optimizer_builder(args)
        model = Model(args)
        optimizer = optmi_builder(model.parameters())
    """

    if not hasattr(args, 'optimizer') or args.optimizer is None:
        return None
    assert hasattr(args, 'lr'), "`args` has no attribute of `lr`."

    optim_name = args.optimizer.strip().lower()
    # TODO: 対応するoptimizerを増やす
    if optim_name.startswith('adaberief'):
        builder = partial(AdaBelief, lr=args.lr, weight_decay=args.weight_decay, weight_decouple=True, rectify=True)
    elif optim_name.startswith('adabound'):
        builder = partial(AdaBound, lr=args.lr, weight_decay=args.weight_decay, final_lr=args.final_lr)
    elif optim_name == 'adam':
        builder = partial(optim.Adam, lr=args.lr, weight_decay=args.weight_decay)
    elif optim_name in ('sgd', 'momentum'):
        builder = partial(optim.SGD, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise ValueError(f"`{args.optimizer}` doesn't match optimizer "
                         "in ('AdaBerief', 'AdaBound', 'Adam', 'SGD', 'Momentum')."
                         " Confirm `args.optimizer`.")

    return builder


def get_scheduler_builder(args) -> Optional[Callable]:
    """
    optimizerオブジェクトを引数としてschedulerを返す関数を作成して返す

    schedulerはargsの値から選択し、warmup==Trueの場合はWarmupSchedulerでラップして返す

    Usage:
        optim_builder = get_optimizer_builder(args)
        sched_builder = get_scheduler_builder(args)
        model = Model(args)
        optimizer = optim_builder(model.parameters())
        scheduler = sched_builder(optimizer)
    """

    use_warmup = hasattr(args, 'warmup') and args.warmup
    use_scheduler = hasattr(args, 'scheduler') and args.scheduler is not None
    if not (use_scheduler or use_warmup):
        return None

    if use_scheduler:
        sched_name = args.scheduler.strip().lower()
        # TODO: 対応するschedulerを増やす
        if sched_name.startswith('cosine'):
            T_max = args.T_max if hasattr(args, 'T_max') else args.epochs
            sched_builder = partial(optim.lr_scheduler.CosineAnnealingLR, T_max=T_max)
        elif sched_name.startswith('step'):
            assert hasattr(args, 'step_size')
            assert hasattr(args, 'gamma')
            sched_builder = partial(optim.lr_scheduler.StepLR, step_size=args.step_size, gamma=args.gamma)
        else:
            raise ValueError(f"`{args.scheduler}` doesn't match scheduler in ('CosineAnnealingLR', 'StepLR')."
                             " Confirm `args.scheduler`.")
    else:
        # Warmupのみの場合、schedulerとしてデフォルト引数のNoneを指定
        sched_builder = lambda x: None

    def builder(optimizer):
        scheduler = sched_builder(optimizer)
        if use_warmup:
            # schedulerをWarmupSchedulerでラップ
            scheduler = WarmupScheduler(optimizer, warmup_epoch=args.warmup_epoch, scheduler=scheduler)

        return scheduler

    return builder
