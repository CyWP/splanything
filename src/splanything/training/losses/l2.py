"""L2 (mean squared error) per-sample loss."""

from jaxtyping import Float
from torch import Tensor

from .base import Loss


class L2Loss(Loss):
    """L2 (Mean Squared Error) loss.

    Computes the mean squared difference between target and output pixels.
    """

    def compute(
        self,
        x: Float[Tensor, "..."],
        target: Float[Tensor, "..."],
    ) -> Float[Tensor, ""]:
        """Compute L2 loss between output and target.

        Args:
            x: Model output.
            target: Ground truth target.

        Returns:
            Mean squared error scalar.
        """
        return ((target - x) ** 2).mean()
