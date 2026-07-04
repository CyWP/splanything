from jaxtyping import Bool
from torch import Tensor

from splanything.primitives import Primitive

from .base import SplitRule


class AreaSplit(SplitRule):
    """Split primitives based on area/scale threshold.

    Identifies primitives whose area exceeds a given threshold and splits
    them into smaller primitives for better coverage of fine details.

    Attributes:
        threshold: Maximum allowed area before splitting.
        interval: Apply every N epochs.
    """

    def __init__(self, threshold: float = 1e-4, interval: int = 10):
        """Initialize AreaSplit rule.

        Args:
            threshold: Maximum area before splitting.
            interval: Apply every N epochs.
        """
        super().__init__(interval=interval)
        self.threshold = threshold

    def apply(self, primitive: Primitive, **kwargs) -> Bool[Tensor, "N"]:
        """Return which primitives to split.

        Args:
            primitive: Primitive to evaluate.

        Returns:
            split: Boolean tensor. True = split, False = ignore.
        """
        areas = primitive.areas
        return areas > self.threshold
