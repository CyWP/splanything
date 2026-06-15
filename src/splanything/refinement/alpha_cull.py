from jaxtyping import Bool
from torch import Tensor

from splanything.training import Trainer

from .generic import FilterRule


class AlphaCull(FilterRule):
    """Cull primitives with alpha values below threshold.

    Removes nearly-transparent primitives that contribute little to
    the reconstruction, reducing computational overhead and preventing
    overfitting to noise.

    Attributes:
        threshold: Alpha threshold for culling.
        interval: Apply every N epochs.

    Required Behaviours:
        - HasAlphas: Primitive must have alphas attribute/property.
    """

    def __init__(self, threshold: float = 0.05, interval: int = 10):
        """Initialize AlphaCull rule.

        Args:
            threshold: Alpha threshold below which primitives are culled.
            interval: Apply every N epochs.
        """
        self.threshold = threshold
        self.interval = interval

    def apply(self, trainer: Trainer) -> Bool[Tensor, "N"]:
        """Return which primitives to keep.

        Args:
            trainer: Optional trainer for epoch checking.

        Returns:
            keep: Boolean tensor. True = keep, False = cull.
        """
        if trainer.epoch % self.interval != 0:
            return None
        p = trainer.primitive
        keep: Bool[Tensor, "N"] = p.alphas >= self.threshold
        print("Cull: ", (~keep).sum().item())
        return keep
