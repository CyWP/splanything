import torch

from splanything.training import Trainer
from jaxtyping import Float
from torch import Tensor
from .generic import Loss


class L1Loss(Loss):
    """L1 (Mean Absolute Error) loss.

    Computes the mean absolute difference between target and output pixels.
    """

    _name = "L1"

    def compute(self, trainer: Trainer) -> Float[Tensor, ""]:
        """Compute L1 loss between current patch output and target.

        Args:
            trainer: Current trainer state.

        Returns:
            Mean absolute error scalar.

        Notes:
            - Uses trainer.last_output and trainer.last_target, both (S, C).
            - Patch losses are accumulated over the sampler loop.
        """
        return torch.abs(trainer.last_target - trainer.last_output).mean()
