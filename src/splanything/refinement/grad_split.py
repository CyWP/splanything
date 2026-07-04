import torch

from jaxtyping import Bool
from torch import Tensor

from splanything.primitives import Primitive

from .base import SplitRule


class GradSplit(SplitRule):
    """Split primitives based on gradient magnitude relative to area.

    Identifies primitives with high gradient-to-area ratios (indicating
    high detail regions) and splits them into smaller primitives for
    better reconstruction fidelity.

    Attributes:
        threshold: Ratio threshold for splitting (grad_mag / area).
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

    def apply(self, primitive: Primitive, **kwargs) -> Bool[Tensor, "N"]:
        """Return which primitives to split.

        Args:
            primitive: Primitive to evaluate.

        Returns:
            split: Boolean tensor. True = split, False = ignore.
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
        ratios = grad_mag * primitive.alphas
        return ratios > self.threshold
