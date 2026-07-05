from jaxtyping import Float
from torch import Tensor

from ..trainer import Trainer
from .base import Loss


class L2Loss(Loss):
    """L2 (Mean Squared Error) loss.

    Computes the mean squared difference between target and output pixels.
    """

    _name = "L2"

    def compute(self, trainer: Trainer) -> Float[Tensor, ""]:
        """Compute L2 loss between current patch output and target.

        Args:
            trainer: Current trainer state.

        Returns:
            Mean squared error scalar.

        Notes:
            - Uses trainer.last_output and trainer.last_target, both (S, C).
            - Patch losses are accumulated over the sampler loop.
        """
        return ((trainer.last_target - trainer.last_output) ** 2).mean()
