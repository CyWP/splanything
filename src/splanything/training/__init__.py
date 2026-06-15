"""Training loop orchestration.

Exposes:
- STAGES: List of training lifecycle stages
- Trainer: Main training loop class
- TrainSampler: Patch sampler for training
"""

from .trainer import (
    Trainer,
    STAGES,
    TRAIN_START,
    TRAIN_END,
    EPOCH_START,
    EPOCH_END,
    PRE_STEP,
)
from .train_sampler import TrainSampler

__all__ = [
    "Trainer",
    "TrainSampler",
    "STAGES",
    "TRAIN_START",
    "TRAIN_END",
    "EPOCH_START",
    "EPOCH_END",
    "PRE_STEP",
]
