import torch

from typing import Optional
from jaxtyping import Bool
from torch import Tensor

from primitives import Primitive, Splittable, HasScales
from trainers import Trainer

from .generic import SplitRule


class IsoSplit(SplitRule):
    """Split primitives that are too anisotropic.

    Identifies primitives where scales[0] / scales[1]
    deviates significantly from 1, indicating anisotropy. Splits them
    to become more isotropic.

    Attributes:
        threshold: Ratio threshold for splitting (e.g., threshold=2 means
            split if scale[0]/scale[1] > 2 or < 0.5).
        interval: Apply every N epochs.

    Required Behaviours:
        - HasScales: Primitive must have scales property.
        - Splittable: Primitive must implement split() method.
    """

    def __init__(
        self, primitive: Primitive, threshold: float = 5.0, interval: int = 10
    ):
        """Initialize IsoSplit rule.

        Args:
            primitive: The primitive to refine.
            threshold: Anisotropy ratio threshold for splitting.
            interval: Apply every N epochs.
        """
        super().__init__(primitive)
        self.threshold = threshold
        self.interval = interval

    @property
    def required_behaviours(self):
        """Requires HasScales and Splittable protocols."""
        return (HasScales, Splittable)

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

        scale_1, scale_2 = p.scales
        ratio = scale_1 / (scale_2 + 1e-8)
        return (ratio > self.threshold) | (ratio < 1.0 / self.threshold)
