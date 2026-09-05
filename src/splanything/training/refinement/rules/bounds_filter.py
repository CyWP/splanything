"""Filter rule culling primitives near the image border."""

from __future__ import annotations

from typing import List, Optional

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from ....primitives.base import Primitive
from ..base import FilterRule
from .threshold_filter import ThresholdFilter


class BoundsFilter(FilterRule):
    """Cull primitives whose coordinates fall within a margin of the image border.

    The margin always defines the **outer** cull zone: primitives whose
    coordinates are within ``margin`` of any border are removed. For a
    margin of 0.1, the outer 10% of the image is the cull zone.

    Operates as a container of ``ThresholdFilter`` instances: for each
    coordinate dimension, an OVER check ensures ``coord > margin`` and an
    UNDER check ensures ``coord < 1 - margin``.

    When ``use_areas`` is True, the primitive's half-extent (derived
    from its area) is added to the effective margin, requiring the whole
    primitive to lie within bounds.

    Attributes:
        margin: Width of the outer cull zone from the image border (default 0.0).
        use_areas: Account for primitive extent in the border check.
        coords_attr: Attribute name for primitive coordinates (default ``"centroids"``).
        interval: Fire every N invocations.
    """

    def __init__(
        self,
        margin: float = 0.0,
        interval: int = 10,
        use_areas: bool = False,
        coords_attr: str = "centroids",
    ):
        """Build the internal border ``ThresholdFilter`` s.

        Args:
            margin: Width of the outer cull zone from the image border.
            interval: Fire every N invocations of ``__call__``.
            use_areas: Account for primitive extent in the border check.
            coords_attr: Attribute name for primitive coordinates.
        """
        super().__init__(interval=interval)
        self.margin = margin
        self.use_areas = use_areas
        self.coords_attr = coords_attr

        self._filters: List[ThresholdFilter] = [
            ThresholdFilter(
                attr_name=coords_attr,
                threshold=margin,
                comparison="OVER",
                method="ALL",
                interval=1,
            ),
            ThresholdFilter(
                attr_name=coords_attr,
                threshold=1.0 - margin,
                comparison="UNDER",
                method="ALL",
                interval=1,
            ),
        ]

    def criterion(self, primitive: Primitive, **kwargs) -> Float[Tensor, "N"]:
        """Compute per-primitive minimum distance to the image border.

        For the simple case (no ``use_areas``), returns a dummy criterion
        --- the actual judgment is delegated to internal ``ThresholdFilter`` s.

        When ``use_areas`` is True, computes the effective border distance
        adjusted by each primitive's half-extent.

        Args:
            primitive: Primitive to evaluate.

        Returns:
            Border distances (N,) when ``use_areas``, otherwise a dummy tensor.
        """
        coords = getattr(primitive, self.coords_attr)
        if self.use_areas and hasattr(primitive, "areas"):
            c_min = coords.min(dim=1).values
            c_max = coords.max(dim=1).values
            half_extent = primitive.areas**0.5 * 0.5
            c_min += half_extent
            c_max -= half_extent
            c_max = 1 - c_max
            return torch.max(c_min, c_max)
        return coords

    def judge(self, criterion: Float[Tensor, "N ..."]) -> Optional[Bool[Tensor, "N"]]:
        """Threshold border distance into a keep mask.

        Without ``use_areas``, delegates to the internal ``ThresholdFilter``
        instances. With ``use_areas``, applies a direct threshold on the
        pre-computed border distance.

        Args:
            criterion: Per-primitive border distances (N,) or coordinates (N, 2).

        Returns:
            keep: Boolean mask (N,). True = keep, False = cull.
        """
        if self.use_areas:
            return criterion > self.margin
        keep = torch.ones(criterion.shape[0], dtype=torch.bool, device=criterion.device)
        for f in self._filters:
            mask = f.judge(criterion)
            if mask is not None:
                keep &= mask
        return keep
