import torch

from trainers import Trainer
from jaxtyping import Float
from torch import Tensor
from .generic import Loss


class L2Loss(Loss):
    """L2 (Mean Squared Error) loss.

    Computes the mean squared difference between target and output pixels.
    """

    _name = "L2"

    def compute(self, trainer: Trainer) -> Float[Tensor, ""]:
        """Compute L2 loss.

        Args:
            trainer: Current trainer state.

        Returns:
            Mean squared error scalar.
        """
        return ((trainer.target - trainer.last_output) ** 2).mean()
