"""Callback saving the primitive state at end of training."""

import logging
from typing import List

import torch

from ..trainer import Trainer
from ..stages import TRAIN_END
from .base import Callback

_logger = logging.getLogger(__name__)


class PrimitiveSave(Callback):
    """Save primitive state at end of training.

    Saves the primitive's state_dict to a file when training completes.

    Stages: TRAIN_END
    """

    _stages: List[str] = [TRAIN_END]

    def __init__(self, path: str):
        """Initialize primitive save callback.

        Args:
            path: File path to save state to.
        """
        super().__init__()
        self.path = path

    def run(self, trainer: Trainer, stage: str):
        """Save primitive state.

        Args:
            trainer: Current trainer instance.
            stage: Current training stage.
        """
        if stage != TRAIN_END:
            return

        try:
            torch.save(trainer.primitive.state_dict(), self.path)
            _logger.info(f"Primitive saved to {self.path}")
        except Exception as e:
            _logger.error(f"Failed to save primitive: {e}")
