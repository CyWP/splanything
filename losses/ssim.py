import torch

from typing import Union, Sequence

from trainers import Trainer

from utils.img import ImgUtils
from .generic import Loss

F = torch.FloatTensor


class SSIMLoss(Loss):
    _name = "SSIM"

    def __init__(
        self,
        weight: float,
        kernel_size: Union[int, Sequence[int]] = 11,
        sigma: Union[float, Sequence[float]] = 6,
    ):
        super().__init__(weight)
        self.kernel = ImgUtils.gaussian_kernel(kernel_size, sigma)

    def compute(self, trainer: Trainer) -> F:
        out = trainer.last_output
        target = trainer.target
        return 1 - (ImgUtils.SSIM(target, out, self.kernel)) / 2
