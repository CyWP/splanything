from typing import Optional

from jaxtyping import Bool
from torch import Tensor

from primitives import Primitive, HasAlphas
from trainers import Trainer

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

    def __init__(
        self, primitive: Primitive, threshold: float = 0.05, interval: int = 10
    ):
        """Initialize AlphaCull rule.

        Args:
            primitive: The primitive to refine.
            threshold: Alpha threshold below which primitives are culled.
            interval: Apply every N epochs.
        """
        super().__init__(primitive)
        self.threshold = threshold
        self.interval = interval

    @property
    def required_behaviours(self):
        """Requires HasAlphas protocol."""
        return (HasAlphas,)

    def apply(self, trainer: Optional[Trainer] = None) -> Bool[Tensor, "N"]:
        """Return which primitives to keep.

        Args:
            trainer: Optional trainer for epoch checking.

        Returns:
            keep: Boolean tensor. True = keep, False = cull.
        """
        if trainer is not None and trainer.epoch % self.interval != 0:
            return None
        p = self.primitive
        keep: Bool[Tensor, "N"] = p.alphas >= self.threshold
        print("Cull: ", (~keep).sum().item())
        return keep
