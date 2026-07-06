from typing import Optional

from jaxtyping import Bool, Float
from torch import Tensor

from ....primitives import Primitive
from ..base import FilterRule, RefinementRule


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

    def criterion(self, primitive: Primitive, **kwargs) -> Float[Tensor, "N"]:
        """Compute per-primitive alphas.

        Args:
            primitive: Primitive to evaluate.

        Returns:
            Alphas tensor (N,).
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
        print("Cull: ", (~keep).sum().item())
        return keep

    apply = RefinementRule.apply
