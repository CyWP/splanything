import torch

from typing import Optional
from jaxtyping import Float, Bool
from torch import Tensor

from primitives import Primitive, Splittable, HasAreas, HasAlphas
from trainers import Trainer

from .generic import SplitRule


class GradSplit(SplitRule):
    """Split primitives based on gradient magnitude relative to area.

    Identifies primitives with high gradient-to-area ratios (indicating
    high detail regions) and splits them into smaller primitives for
    better reconstruction fidelity.

    Attributes:
        threshold: Ratio threshold for splitting (grad_mag / area).
        interval: Apply every N epochs.

    Required Behaviours:
        - Splittable: Primitive must implement split() method.
        - HasAreas: Primitive must have areas property.
    """

    def __init__(
        self, primitive: Primitive, threshold: float = 0.05, interval: int = 10
    ):
        """Initialize GradSplit rule.

        Args:
            primitive: The primitive to refine.
            threshold: Gradient-to-area ratio threshold for splitting.
            interval: Apply every N epochs.
        """
        super().__init__(primitive)
        self.threshold = threshold
        self.interval = interval

    @property
    def required_behaviours(self):
        """Requires Splittable, HasAreas, and HasAlphas protocols."""
        return (Splittable, HasAreas, HasAlphas)

    def apply(self, trainer: Optional[Trainer] = None) -> Bool[Tensor, "N"]:
        """Return which primitives to split.

        Args:
            trainer: Optional trainer for epoch checking.

        Returns:
            split: Boolean tensor. True = split, False = ignore.
        """
        if trainer is not None and trainer.epoch % self.interval != 0:
            return None
        p = self.primitive
        areas: Float[Tensor, "N"] = p.areas
        grad_mag = torch.zeros((len(p),), device=areas.device, dtype=areas.dtype)
        for _, grad in p.batched_grads():
            g = grad.abs()
            if len(g.shape) > 1:
                g = g.sum(dim=tuple(range(1, len(g.shape))))
            grad_mag += g
        ratios = grad_mag * p.alphas
        return ratios > self.threshold
