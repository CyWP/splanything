from jaxtyping import Bool, Float
from torch import Tensor

from ....primitives import Primitive
from ..base import RefinementRule, SplitRule


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

    def criterion(self, primitive: Primitive, **kwargs) -> Float[Tensor, "N"]:
        """Compute per-primitive areas.

        Args:
            primitive: Primitive to evaluate.

        Returns:
            Areas tensor (N,).
        """
        return primitive.areas

    def judge(self, criterion: Float[Tensor, "N"]) -> Bool[Tensor, "N"]:
        """Threshold areas into a split mask.

        Args:
            criterion: Per-primitive areas (N,).

        Returns:
            split: Boolean tensor. True = split, False = ignore.
        """
        return criterion > self.threshold

    apply = RefinementRule.apply
