import torch
from jaxtyping import Float
from torch import Tensor

from .base import ImageLoss


class L1ImageLoss(ImageLoss):
    """L1 (Absolute Error) image loss.

    Per-pixel absolute difference between target and output.
    """

    def compute(
        self,
        x: Float[Tensor, "B C H W"],
        target: Float[Tensor, "B C H W"],
    ) -> Float[Tensor, "B C H W"]:
        """Compute L1 loss map between output and target.

        Args:
            x: Model output (B, C, H, W).
            target: Ground truth target (B, C, H, W).

        Returns:
            Per-pixel absolute error (B, C, H, W).
        """
        return torch.abs(target - x)
