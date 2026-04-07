import torch

from typing import List, Optional, Any

from trainers import Trainer, EPOCH_END
from utils.comfy import ComfyUtils
from utils.img import ImgUtils

from .generic import Callback


class ImgUpdate(Callback):
    """Real-time image preview during training.

    Sends generated images to ComfyUI for visual inspection at specified
    epoch intervals. Uses node's `send_preview` method if available.

    Stages: EPOCH_END
    """

    _stages: List[str] = [EPOCH_END]

    def __init__(self, frequency: int = 1, node: Optional[Any] = None):
        """Initialize image update callback.

        Args:
            frequency: Update preview every N epochs (default: 1).
            node: ComfyUI node for preview output.
        """
        super().__init__(node=node)
        self.frequency = max(frequency, 1)

    def run(self, trainer: Trainer, stage: str):
        """Send preview image if epoch matches frequency.

        Args:
            trainer: Current trainer instance.
            stage: Current training stage.
        """
        if trainer.epoch % self.frequency == 0:
            img = ImgUtils.tensor2img(trainer.last_output.clone().detach())
            ComfyUtils.preview_image(self.node, img)
