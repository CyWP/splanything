from typing import Optional

from jaxtyping import Bool
from torch import Tensor

from ...primitives import Primitive
from .base import FilterRule


class AlphaCull(FilterRule):
    """Cull primitives with alpha values below threshold.

    Removes nearly-transparent primitives that contribute little to
    the reconstruction, reducing computational overhead and preventing
    overfitting to noise.

    Attributes:
        threshold: Alpha threshold for culling.
        interval: Apply every N epochs.
    """

    def __init__(self, threshold: float = 0.05, interval: int = 10):
        """Initialize AlphaCull rule.

        Args:
            threshold: Alpha threshold below which primitives are culled.
            interval: Apply every N epochs.
        """
        super().__init__(interval=interval)
        self.threshold = threshold

    def apply(self, primitive: Primitive, **kwargs) -> Optional[Bool[Tensor, "N"]]:
        """Return which primitives to keep.

        Args:
            primitive: Primitive to evaluate.

        Returns:
            keep: Boolean tensor. True = keep, False = cull.
        """
        keep: Bool[Tensor, "N"] = primitive.alphas >= self.threshold
        print("Cull: ", (~keep).sum().item())
        return keep
