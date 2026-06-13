"""Training loop orchestration.

Exposes:
- STAGES: List of training lifecycle stages
- Trainer: Main training loop class
- load_train_config: Factory function to load full training config from JSON/YAML
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
from .train_config import load_train_config
