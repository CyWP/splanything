"""Split rule for primitives with anisotropic scale ratios."""

from jaxtyping import Bool, Float
from torch import Tensor

from ....primitives.base import Primitive
from ..base import RefinementRule, SplitRule


class IsoSplit(SplitRule):
    """Split primitives whose aspect ratio deviates from isotropic.

    Splits primitives where ``scales[0] / scales[1]`` is above
    ``threshold`` or below ``1 / threshold``, indicating significant
    anisotropy.

    Attributes:
        threshold: Anisotropy ratio threshold (default 5.0).
        interval: Fire every N invocations.
    """

    def __init__(self, threshold: float = 5.0, interval: int = 10):
        """
        Args:
            threshold: Split if ``s1/s2 > threshold`` or ``s1/s2 < 1/threshold``
                (default 5.0).
            interval: Fire every N invocations (default 10).
        """
        super().__init__(interval=interval)
        self.threshold = threshold

    def criterion(self, primitive: Primitive, **kwargs) -> Float[Tensor, "N"]:
        """Compute per-primitive scale ratios.

        Args:
            primitive: Primitive to evaluate.

        Returns:
            Scale ratios (N,) = ``scales[0] / (scales[1] + 1e-8)``.
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
