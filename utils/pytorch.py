import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

OPTIMIZERS = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "adagrad": optim.Adagrad,
    "adadelta": optim.Adadelta,
    "rmsprop": optim.RMSprop,
    "asgd": optim.ASGD,
    "lbfgs": optim.LBFGS,
    "nadam": optim.NAdam,
    "radam": optim.RAdam,
}

SCHEDULERS = {
    "step": lr_scheduler.StepLR,
    "multistep": lr_scheduler.MultiStepLR,
    "exponential": lr_scheduler.ExponentialLR,
    "cosine": lr_scheduler.CosineAnnealingLR,
    "cosine_warm_restarts": lr_scheduler.CosineAnnealingWarmRestarts,
    "plateau": lr_scheduler.ReduceLROnPlateau,
    "cyclic": lr_scheduler.CyclicLR,
    "onecycle": lr_scheduler.OneCycleLR,
    "linear": lr_scheduler.LinearLR,
    "constant": lr_scheduler.ConstantLR,
    "polynomial": lr_scheduler.PolynomialLR,
    "sequential": lr_scheduler.SequentialLR,
    "chained": lr_scheduler.ChainedScheduler,
}


def init_optimizer(name: str, params, **kwargs) -> optim.Optimizer:
    name = name.lower()
    if name not in OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer '{name}'. Available: {list(OPTIMIZERS.keys())}"
        )
    return OPTIMIZERS[name](params, **kwargs)


def init_scheduler(
    name: str, optimizer: optim.Optimizer, **kwargs
) -> lr_scheduler._LRScheduler:
    name = name.lower()
    if name not in SCHEDULERS:
        raise ValueError(
            f"Unknown scheduler '{name}'. Available: {list(SCHEDULERS.keys())}"
        )
    return SCHEDULERS[name](optimizer, **kwargs)
