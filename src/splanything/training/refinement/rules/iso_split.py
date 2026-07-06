from jaxtyping import Bool, Float
from torch import Tensor

from ....primitives import Primitive
from ..base import RefinementRule, SplitRule


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

    def criterion(self, primitive: Primitive, **kwargs) -> Float[Tensor, "N"]:
        """Compute per-primitive anisotropy ratios.

        Args:
            primitive: Primitive to evaluate.

        Returns:
            scale_1 / scale_2 ratios (N,).
        """
        scale_1, scale_2 = primitive.scales
        return scale_1 / (scale_2 + 1e-8)

    def judge(self, criterion: Float[Tensor, "N"]) -> Bool[Tensor, "N"]:
        """Threshold anisotropy ratios into a split mask.

        Args:
            criterion: Per-primitive ratios (N,).

        Returns:
            split: Boolean tensor. True = split, False = ignore.
        """
        return (criterion > self.threshold) | (criterion < 1.0 / self.threshold)

    apply = RefinementRule.apply
