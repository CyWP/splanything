import torch

from typing import List

from trainers import Trainer, EPOCH_END
from utils.img import ImgUtils

from .generic import Callback


class NodePreview(Callback):
    """Real-time image preview during training.

    Sends generated images to ComfyUI for visual inspection at specified
    epoch intervals.

    Stages: EPOCH_END
    """

    _stages: List[str] = [EPOCH_END]

    def __init__(self, frequency: int = 1):
        """Initialize node preview callback.

        Args:
            frequency: Update preview every N epochs (default: 1).
        """
        super().__init__()
        self.frequency = max(frequency, 1)

    def run(self, trainer: Trainer, stage: str):
        """Send preview image if epoch matches frequency.

        Args:
            trainer: Current trainer instance.
            stage: Current training stage.
        """
        from utils.comfy import ComfyUtils

        if trainer.epoch % self.frequency == 0:
            img = ImgUtils.tensor2img(trainer.last_output.clone().detach())
            ComfyUtils.preview_image(None, img)
