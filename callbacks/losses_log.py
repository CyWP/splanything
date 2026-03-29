import torch

from typing import List, Optional, Any

from trainers import Trainer, EPOCH_END
from utils.comfy import ComfyUtils

from .generic import Callback


class LossLogger(Callback):

    _stages: List[str] = [EPOCH_END]

    def run(self, trainer: Trainer, stage: str):
        losses = {k: v.item() for k, v in trainer.last_losses.items()}
        losses["total"] = trainer.last_loss.item()
        trainer.log_stats("losses", losses)
