import torch

from jaxtyping import Bool
from torch import Tensor

from splanything.primitives import Primitive

from .base import SplitRule


class IsoSplit(SplitRule):
    """Split primitives that are too anisotropic.

    Identifies primitives where scales[0] / scales[1]
    deviates significantly from 1, indicating anisotropy. Splits them
    to become more isotropic.

    Attributes:
        threshold: Ratio threshold for splitting (e.g., threshold=2 means
            split if scale[0]/scale[1] > 2 or < 0.5).
        interval: Apply every N epochs.
    """

    def __init__(self, threshold: float = 5.0, interval: int = 10):
        """Initialize IsoSplit rule.

        Args:
            threshold: Anisotropy ratio threshold for splitting.
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
        scale_1, scale_2 = primitive.scales
        ratio = scale_1 / (scale_2 + 1e-8)
        return (ratio > self.threshold) | (ratio < 1.0 / self.threshold)
