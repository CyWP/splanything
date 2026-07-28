from typing import Optional

from jaxtyping import Bool, Float
from torch import Tensor

from ....primitives.base import Primitive
from ..base import FilterRule, RefinementRule


class AlphaFilter(FilterRule):
    """Cull primitives whose alpha values fall below a threshold.

    Removes nearly-transparent primitives that contribute little to the
    reconstruction.

    Attributes:
        threshold: Alpha threshold for culling.
        interval: Apply every N epochs.
    """

    def __init__(self, threshold: float = 0.05, interval: int = 10):
        """
        Args:
            threshold: Alpha below which primitives are culled (default 0.05).
            interval: Fire every N invocations (default 10).
        """
        super().__init__(interval=interval)
        self.threshold = threshold

    def criterion(self, primitive: Primitive, **kwargs) -> Float[Tensor, "N"]:
        """Read per-primitive alphas.

        Args:
            primitive: Primitive to evaluate.

        Returns:
            Alphas (N,).
        """
        return primitive.alphas

    def judge(self, criterion: Float[Tensor, "N"]) -> Optional[Bool[Tensor, "N"]]:
        """Threshold alphas into a keep mask.

        Args:
            criterion: Per-primitive alphas (N,).

        Returns:
            keep: Boolean tensor. True = keep, False = cull.
        """
        keep: Bool[Tensor, "N"] = criterion >= self.threshold
        return keep

    apply = RefinementRule.apply
