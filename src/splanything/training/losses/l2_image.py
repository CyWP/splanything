"""L2 (squared error) image-level loss."""

from jaxtyping import Float
from torch import Tensor

from .base import ImageLoss


class L2ImageLoss(ImageLoss):
    """L2 (Squared Error) image loss.

    Per-pixel squared difference between target and output.
    """

    def compute(
        self,
        x: Float[Tensor, "B C H W"],
        target: Float[Tensor, "B C H W"],
    ) -> Float[Tensor, "B C H W"]:
        """Compute L2 loss map between output and target.

        Args:
            x: Model output (B, C, H, W).
            target: Ground truth target (B, C, H, W).

        Returns:
            Per-pixel squared error (B, C, H, W).
        """
        return (target - x) ** 2
