import torch

from typing import Union, Sequence
from trainers import Trainer
from jaxtyping import Float
from torch import Tensor
from utils.img import ImgUtils
from .generic import Loss


class SSIMLoss(Loss):
    """Structural Similarity Index (SSIM) loss.

    Computes 1 - SSIM/2 as a differentiable loss for image quality.
    SSIM measures perceptual similarity using luminance, contrast, and structure.

    Attributes:
        kernel: Gaussian kernel for SSIM computation.

    Notes:
        - SSIM ranges from -1 to 1, where 1 is perfect similarity.
        - Loss is 1 - SSIM/2, so 0 means identical, 1 means maximally different.
    """

    _name = "SSIM"

    def __init__(
        self,
        weight: float,
        kernel_size: Union[int, Sequence[int]] = 11,
        sigma: Union[float, Sequence[float]] = 6,
    ):
        """Initialize SSIM loss.

        Args:
            weight: Weight multiplier for this loss term.
            kernel_size: Size of Gaussian kernel (int or [H, W]).
            sigma: Standard deviation of Gaussian (float or [H, W]).
        """
        super().__init__(weight)
        self.kernel = ImgUtils.gaussian_kernel(kernel_size, sigma)

    def compute(self, trainer: Trainer) -> Float[Tensor, ""]:
        """Compute SSIM loss.

        Args:
            trainer: Current trainer state.

        Returns:
            1 - SSIM/2 scalar.
        """
        out = trainer.last_output
        target = trainer.target
        return 1 - (ImgUtils.SSIM(target, out, self.kernel)) / 2
