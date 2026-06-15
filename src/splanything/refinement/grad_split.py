import torch

from jaxtyping import Bool
from torch import Tensor

from splanything.training import Trainer

from .generic import SplitRule


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
        self.threshold = threshold
        self.interval = interval

    def apply(self, trainer: Trainer) -> Bool[Tensor, "N"]:
        """Return which primitives to split.

        Args:
            trainer: Optional trainer for epoch checking.

        Returns:
            split: Boolean tensor. True = split, False = ignore.
        """
        if trainer.epoch % self.interval != 0:
            return None
        p = trainer.primitive
        areas = (
            p.areas
            if hasattr(p, "areas")
            else torch.ones((len(p),), device=p.device, dtype=p.dtype)
        )
        grad_mag = torch.zeros((len(p),), device=areas.device, dtype=areas.dtype)
        for _, grad in p.batched_grads():
            g = grad.abs()
            if len(g.shape) > 1:
                g = g.sum(dim=tuple(range(1, len(g.shape))))
            grad_mag += g
        ratios = grad_mag * p.alphas
        return ratios > self.threshold
