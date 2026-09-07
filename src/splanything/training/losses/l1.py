"""L1 (mean absolute error) per-sample loss."""

import torch
from jaxtyping import Float
from torch import Tensor

from .base import Loss


class L1Loss(Loss):
    """L1 (Mean Absolute Error) loss.

    Computes the mean absolute difference between target and output pixels.
    """

    def compute(
        self,
        x: Float[Tensor, "..."],
        target: Float[Tensor, "..."],
    ) -> Float[Tensor, ""]:
        """Compute L1 loss between output and target.

        Args:
            x: Model output.
            target: Ground truth target.

        Returns:
            Mean absolute error scalar.
        """
        return torch.abs(target - x).mean()
