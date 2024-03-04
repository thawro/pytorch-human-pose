from typing import Literal, Type

from torch import nn, optim

from src.base.lr_scheduler import LRScheduler

_optimizers = Literal["Adam", "Adamax", "Adadelta", "Adagrad", "AdamW", "SGD", "RMSprop"]


optimizers: dict[str, Type[optim.Optimizer]] = {
    "Adam": optim.Adam,
    "Adadelta": optim.Adadelta,
    "Adagrad": optim.Adagrad,
    "AdamW": optim.AdamW,
    "Adamax": optim.Adamax,
    "SGD": optim.SGD,
    "RMSprop": optim.RMSprop,
}

_lr_schedulers = Literal[
    "CosineAnnealingLR",
    "CosineAnnealingWarmRestarts",
    "MultiStepLR",
    "OneCycleLR",
    "ReduceLROnPlateau",
    "ExponentialLR",
    "PolynomialLR",
]
lr_schedulers: dict[str, Type[optim.lr_scheduler.LRScheduler]] = {
    "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
    "CosineAnnealingWarmRestarts": optim.lr_scheduler.CosineAnnealingWarmRestarts,
    "MultiStepLR": optim.lr_scheduler.MultiStepLR,
    "OneCycleLR": optim.lr_scheduler.OneCycleLR,
    "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau,
    "ExponentialLR": optim.lr_scheduler.ExponentialLR,
    "PolynomialLR": optim.lr_scheduler.PolynomialLR,
}


def create_optimizer(net: nn.Module, name: _optimizers, **params) -> optim.Optimizer:
    OptimizerClass = optimizers[name]
    return OptimizerClass(
        filter(lambda p: p.requires_grad, net.parameters()),
        **params,
    )


def create_lr_scheduler(
    optimizer: optim.Optimizer,
    name: _lr_schedulers,
    interval: Literal["epoch", "step"],
    **params,
) -> LRScheduler:
    LRSchedulerClass = lr_schedulers[name]
    return LRScheduler(LRSchedulerClass(optimizer, **params), interval=interval)
