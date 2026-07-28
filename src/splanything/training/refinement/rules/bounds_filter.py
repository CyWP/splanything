import torch

from typing import Optional

from jaxtyping import Bool, Float
from torch import Tensor

from ....primitives.base import Primitive
from ..base import FilterRule


class BoundsFilter(FilterRule):
    """Cull primitives whose centroids fall outside a margin from the image border.

    Primitives whose nearest border distance (in UV space [0, 1]) is
    below ``margin`` are culled. When ``use_areas`` is True, the
    primitive's extent (derived from its area) is subtracted from the
    distance, effectively requiring the whole primitive to lie within
    the margin.

    Attributes:
        margin: Minimum distance to the image border (default 0.0).
        use_areas: Account for primitive extent in the border check.
        interval: Fire every N invocations.
    """

    def __init__(
        self, margin: float = 0.0, interval: int = 10, use_areas: bool = False
    ):
        """
        Args:
            margin: Minimum distance from image border (default 0.0).
            interval: Fire every N invocations (default 10).
            use_areas: When True, subtract primitive half-extent from
                the border distance (default False).
        """
        super().__init__(interval=interval)
        self.margin = margin
        self.use_areas = use_areas

    def criterion(self, primitive: Primitive, **kwargs) -> Float[Tensor, "N"]:
        """Compute per-primitive minimum distance to the image border.

        Args:
            primitive: Primitive to evaluate.

        Returns:
            Border distances (N,).
        """
        c_min = primitive.centroids.min(dim=1).values
        c_max = primitive.centroids.max(dim=1).values
        if self.use_areas and hasattr(primitive, "areas"):
            half_extent = primitive.areas[:, None] ** 0.5 * 0.5
            c_min += half_extent
            c_max -= half_extent
        c_max = 1 - c_max
        return torch.max(c_min, c_max)

    def judge(self, criterion: Float[Tensor, "N"]) -> Optional[Bool[Tensor, "N"]]:
        """Threshold border distance into a keep mask.

        Args:
            criterion: Per-primitive border distances (N,).

        Returns:
            keep: Boolean mask (N,). True = keep, False = cull.
        """
        return criterion > self.margin
