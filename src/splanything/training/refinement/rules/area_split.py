from jaxtyping import Bool, Float
from torch import Tensor

from ....primitives.base import Primitive
from ..base import RefinementRule, SplitRule


class AreaSplit(SplitRule):
    """Split primitives whose area exceeds a threshold.

    Identifies primitives covering too much of the image and splits them
    into smaller primitives for finer detail coverage.

    Attributes:
        threshold: Maximum area before splitting.
        interval: Fire every N invocations.
    """

    def __init__(self, threshold: float = 1e-4, interval: int = 10):
        """
        Args:
            threshold: Maximum allowed area before splitting (default 1e-4).
            interval: Fire every N invocations (default 10).
        """
        super().__init__(interval=interval)
        self.threshold = threshold

    def criterion(self, primitive: Primitive, **kwargs) -> Float[Tensor, "N"]:
        """Read per-primitive areas.

        Args:
            primitive: Primitive to evaluate.

        Returns:
            Areas (N,).
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
