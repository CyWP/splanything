import torch

from typing import List, Optional, Any

from trainers import Trainer, EPOCH_START, EPOCH_END
from utils.comfy import ComfyUtils

from .generic import Callback


class LoopControl(Callback):

    _stages: List[str] = [EPOCH_START, EPOCH_END]

    def __init__(self, epochs: Optional[int] = None, node: Optional[Any] = None):
        super().__init__(node=node)
        self.epochs = epochs
        if epochs is not None and epochs > 0:
            self.pbar = ComfyUtils.make_progress(epochs)
        else:
            self.pbar = None

    def run(self, trainer: Trainer, stage: str):
        if stage == EPOCH_START:
            if ComfyUtils.is_interrupted():
                trainer.stop()
        elif stage == EPOCH_END:
            ComfyUtils.update_progress(self.pbar, trainer.epoch)
