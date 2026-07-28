import torch
from jaxtyping import Bool, Float
from torch import Tensor

from ....primitives.base import Primitive
from ..base import RefinementRule, SplitRule


class GradSplit(SplitRule):
    """Split primitives with high gradient magnitude relative to area.

    Aggregates the absolute gradient across all parameter groups,
    weights by alpha, and splits primitives with large
    gradient-to-area ratios (indicating high-detail regions).

    Attributes:
        threshold: Split threshold on ``grad_mag * alphas``.
        interval: Fire every N invocations.
    """

    def __init__(self, threshold: float = 0.05, interval: int = 10):
        """
        Args:
            threshold: Gradient-to-area ratio threshold (default 0.05).
            interval: Fire every N invocations (default 10).
        """
        super().__init__(interval=interval)
        self.threshold = threshold

    def criterion(self, primitive: Primitive, **kwargs) -> Float[Tensor, "N"]:
        """Compute per-primitive gradient-alpha products.

        Args:
            primitive: Primitive to evaluate.

        Returns:
            Gradient-alpha products (N,).
        """
        areas = (
            primitive.areas
            if hasattr(primitive, "areas")
            else torch.ones(
                (len(primitive),), device=primitive.device, dtype=primitive.dtype
            )
        ) ** 0.5
        grad_mag = torch.zeros(
            (len(primitive),), device=areas.device, dtype=areas.dtype
        )
        for named, grad in primitive.batched_grads():
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
