import torch

from trainers import Trainer
from jaxtyping import Float
from torch import Tensor
from .generic import Loss


class L1Loss(Loss):
    """L1 (Mean Absolute Error) loss.

    Computes the mean absolute difference between target and output pixels.
    """

    _name = "L1"

    def compute(self, trainer: Trainer) -> Float[Tensor, ""]:
        """Compute L1 loss.

        Args:
            trainer: Current trainer state.

        Returns:
            Mean absolute error scalar.
        """
        return torch.abs(trainer.target - trainer.last_output).mean()
