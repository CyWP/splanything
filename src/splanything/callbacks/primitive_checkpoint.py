import logging

from typing import List, Optional

from splanything.training import Trainer, EPOCH_END

from .base import Callback

_logger = logging.getLogger(__name__)


class PrimitiveCheckpoint(Callback):
    """Save primitive checkpoint at specified interval.

    Saves the primitive's state_dict to the trainer's run folder
    at regular intervals during training.

    Stages: EPOCH_END
    """

    _stages: List[str] = [EPOCH_END]

    def __init__(self, interval: int = 10):
        """Initialize primitive checkpoint callback.

        Args:
            interval: Save checkpoint every N epochs (default: 10).
        """
        super().__init__()
        self.interval = max(interval, 1)

    def run(self, trainer: Trainer, stage: str):
        """Save checkpoint if epoch matches interval.

        Args:
            trainer: Current trainer instance.
            stage: Current training stage.
        """
        if trainer.epoch % self.interval == 0:
            trainer.save_checkpoint(epoch=trainer.epoch)
