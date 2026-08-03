from __future__ import annotations

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from ....primitives.base import Primitive
from ....utils.img import ImgUtils
from ..base import FilterRule, RefinementRule


class MapFilter(FilterRule):
    """Cull primitives by sampling a spatial probability map at coordinates.

    The map (B, 1, H, W) is bilinearly sampled at each primitive's
    coordinate (default: centroid). Each primitive is independently kept
    or culled via a Bernoulli draw against its sampled probability.
    Useful when culling should vary spatially.

    Attributes:
        map: Probability map (B, 1, H, W) with values in [0, 1].
        coords_attr: Attribute name for sampling coordinates (default ``"centroids"``).
        interval: Fire every N invocations.

    Notes:
        - Only ``map[0]`` (first batch element) is sampled.
        - Coordinates must be in [0, 1] UV space.
    """

    def __init__(
        self,
        map: Float[Tensor, "B 1 H W"],
        coords_attr: str = "adjusted_coords",
        interval: int = 1,
    ):
        """
        Args:
            map: Probability map (B, 1, H, W), values in [0, 1].
            coords_attr: Attribute name for sampling coordinates (default ``"centroids"``).
            interval: Fire every N invocations (default 1).
        """
        super().__init__(interval=interval)
        self.map = map
        self.coords_attr = coords_attr

    def criterion(self, primitive: Primitive, **kwargs) -> Float[Tensor, "N"]:
        """Sample the probability map at each primitive's coordinates.

        Args:
            primitive: Primitive whose coordinates define sample locations.

        Returns:
            Per-primitive probabilities (N,).
        """
        sampled = ImgUtils.uv_sample(self.map, getattr(primitive, self.coords_attr))
        return sampled[0, :, 0]

    def judge(self, criterion: Float[Tensor, "N"]) -> Bool[Tensor, "N"]:
        """Decide which primitives to keep.

        Args:
            criterion: Per-primitive probabilities (N,) in [0, 1].

        Returns:
            keep: Boolean mask (N,). True = KEEP, False = CULL.
        """
        return torch.bernoulli(criterion).bool()

    apply = RefinementRule.apply
