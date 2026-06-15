import torch
import torch.nn as nn

from splanything.training import Trainer
from jaxtyping import Float
from torch import Tensor


class Loss(nn.Module):
    """Base class for loss functions.

    A Loss computes a scalar value measuring the difference between
    the target image and the primitive's current output.

    Attributes:
        weight: Multiplier for the loss value in combined loss computation.

    Notes:
        - Subclasses must implement `compute(trainer) -> Float[Tensor, ""]`.
        - The `forward` method applies the weight multiplier.
    """

    def __init__(self, weight: float, **kwargs):
        """Initialize loss.

        Args:
            weight: Weight multiplier for this loss term.
        """
        super().__init__()
        self.weight = weight

    def compute(self, trainer: Trainer) -> Float[Tensor, ""]:
        """Compute unweighted loss value.

        Args:
            trainer: Current trainer state.

        Returns:
            Loss scalar tensor.
        """
        raise NotImplementedError()

    def forward(self, trainer: Trainer) -> Float[Tensor, ""]:
        """Compute weighted loss.

        Args:
            trainer: Current trainer state.

        Returns:
            Weighted loss scalar.
        """
        return self.compute(trainer) * self.weight
