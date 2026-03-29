import torch

from typing import List, Optional, Any

from trainers import Trainer, EPOCH_END
from utils.comfy import ComfyUtils
from utils.img import ImgUtils

from .generic import Callback


class ImgUpdate(Callback):

    _stages: List[str] = [EPOCH_END]

    def __init__(self, frequency: int = 1, node: Optional[Any] = None):
        super().__init__(node=node)
        self.frequency = max(frequency, 1)

    def run(self, trainer: Trainer, stage: str):
        if trainer.epoch % self.frequency == 0:
            img = ImgUtils.tensor2img(trainer.last_output.clone().detach())
            ComfyUtils.preview_image(self.node, img)
