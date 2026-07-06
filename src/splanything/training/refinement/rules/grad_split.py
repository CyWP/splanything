import torch
from jaxtyping import Bool, Float
from torch import Tensor

from ....primitives import Primitive
from ..base import RefinementRule, SplitRule


class GradSplit(SplitRule):
    """Split primitives based on gradient magnitude relative to area.

    Identifies primitives with high gradient-to-area ratios (indicating
    high detail regions) and splits them into smaller primitives for
    better reconstruction fidelity.

    Attributes:
        threshold: Ratio threshold for splitting (grad_mag * alphas).
        interval: Apply every N epochs.
    """

    def __init__(self, threshold: float = 0.05, interval: int = 10):
        """Initialize GradSplit rule.

        Args:
            threshold: Gradient-to-area ratio threshold for splitting.
            interval: Apply every N epochs.
        """
        super().__init__(interval=interval)
        self.threshold = threshold

    def criterion(self, primitive: Primitive, **kwargs) -> Float[Tensor, "N"]:
        """Compute per-primitive gradient-to-area ratios.

        Args:
            primitive: Primitive to evaluate.

        Returns:
            Ratios tensor (N,).
        """
        areas = (
            primitive.areas
            if hasattr(primitive, "areas")
            else torch.ones(
                (len(primitive),), device=primitive.device, dtype=primitive.dtype
            )
        )
        grad_mag = torch.zeros(
            (len(primitive),), device=areas.device, dtype=areas.dtype
        )
        for _, grad in primitive.batched_grads():
            g = grad.abs()
            if len(g.shape) > 1:
                g = g.sum(dim=tuple(range(1, len(g.shape))))
            grad_mag += g
        return grad_mag * primitive.alphas

    def judge(self, criterion: Float[Tensor, "N"]) -> Bool[Tensor, "N"]:
        """Threshold ratios into a split mask.

        Args:
            criterion: Per-primitive ratios (N,).

        Returns:
            split: Boolean tensor. True = split, False = ignore.
        """
        return criterion > self.threshold

    apply = RefinementRule.apply
