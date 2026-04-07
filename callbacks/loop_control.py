import torch

from typing import List, Optional, Any

from trainers import Trainer, EPOCH_START, EPOCH_END
from utils.comfy import ComfyUtils

from .generic import Callback


class LoopControl(Callback):
    """Training loop control with epoch counting and interrupt handling.

    Manages training duration and responds to ComfyUI interrupt signals.
    Updates a progress bar if epochs are specified.

    Stages: EPOCH_START, EPOCH_END
    """

    _stages: List[str] = [EPOCH_START, EPOCH_END]

    def __init__(self, epochs: Optional[int] = None, node: Optional[Any] = None):
        """Initialize loop control.

        Args:
            epochs: Optional max epochs. If None, runs until interrupted.
            node: Optional ComfyUI node reference.
        """
        super().__init__(node=node)
        self.epochs = epochs
        if epochs is not None and epochs > 0:
            self.pbar = ComfyUtils.make_progress(epochs)
        else:
            self.pbar = None

    def run(self, trainer: Trainer, stage: str):
        """Handle epoch start/end.

        Args:
            trainer: Current trainer instance.
            stage: Current training stage.
        """
        if stage == EPOCH_START:
            if ComfyUtils.is_interrupted():
                trainer.stop()
        elif stage == EPOCH_END:
            ComfyUtils.update_progress(self.pbar, trainer.epoch)
