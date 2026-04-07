from typing import List

from trainers import Trainer, EPOCH_END

from .generic import Callback


class LossLogger(Callback):
    """Loss value logger for training metrics.

    Records per-epoch loss values to trainer's log dictionary for
    post-training analysis or real-time monitoring.

    Stages: EPOCH_END
    """

    _stages: List[str] = [EPOCH_END]

    def run(self, trainer: Trainer, stage: str):
        """Log loss values for current epoch.

        Args:
            trainer: Current trainer instance.
            stage: Current training stage.
        """
        losses = {k: v.item() for k, v in trainer.last_losses.items()}
        losses["total"] = trainer.last_loss.item()
        trainer.log_stats("losses", losses)
