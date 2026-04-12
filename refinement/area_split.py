import torch

from typing import Optional
from jaxtyping import Bool
from torch import Tensor

from primitives import Primitive, Splittable, HasAreas
from trainers import Trainer

from .generic import SplitRule


class AreaSplit(SplitRule):
    """Split primitives based on area/scale threshold.

    Identifies primitives whose area exceeds a given threshold and splits
    them into smaller primitives for better coverage of fine details.

    Attributes:
        threshold: Maximum allowed area before splitting.
        interval: Apply every N epochs.

    Required Behaviours:
        - Splittable: Primitive must implement split() method.
        - HasAreas: Primitive must have areas property.
    """

    def __init__(
        self, primitive: Primitive, threshold: float = 1e-4, interval: int = 10
    ):
        """Initialize AreaSplit rule.

        Args:
            primitive: The primitive to refine.
            threshold: Maximum area before splitting.
            interval: Apply every N epochs.
        """
        super().__init__(primitive)
        self.threshold = threshold
        self.interval = interval

    @property
    def required_behaviours(self):
        """Requires Splittable and HasAreas protocols."""
        return (Splittable, HasAreas)

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
        areas = p.areas
        return areas > self.threshold
