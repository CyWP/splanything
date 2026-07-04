from typing import List, Optional

from splanything.training import Trainer, EPOCH_START, EPOCH_END

from .base import Callback


class LoopControl(Callback):
    """Training loop control with epoch counting and progress bar.

    Manages training duration and updates a tqdm progress bar if epochs
    are specified.

    Stages: EPOCH_START, EPOCH_END
    """

    _stages: List[str] = [EPOCH_START, EPOCH_END]

    def __init__(self, epochs: Optional[int] = None):
        """Initialize loop control.

        Args:
            epochs: Optional max epochs. If None, runs until interrupted.
        """
        super().__init__()
        self.epochs = epochs
        self._pbar = None

    def _get_pbar(self):
        """Lazy initialization of tqdm progress bar."""
        if self._pbar is None and self.epochs is not None and self.epochs > 0:
            try:
                from tqdm import tqdm

                self._pbar = tqdm(total=self.epochs, desc="Training")
            except ImportError:
                pass
        return self._pbar

    def run(self, trainer: Trainer, stage: str):
        """Handle epoch start/end.

        Args:
            trainer: Current trainer instance.
            stage: Current training stage.
        """
        if stage == EPOCH_START:
            pass
        elif stage == EPOCH_END:
            pbar = self._get_pbar()
            if pbar is not None:
                pbar.update(1)
                if hasattr(trainer, "last_loss") and trainer.last_loss is not None:
                    pbar.set_postfix({"loss": f"{trainer.last_loss.item():.4f}"})
        if trainer.epoch == self.epochs:
            trainer.stop()
