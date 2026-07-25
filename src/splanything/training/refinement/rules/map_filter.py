from __future__ import annotations

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from ....primitives.base import Primitive
from ....utils.img import ImgUtils
from ..base import FilterRule, RefinementRule


class MapFilter(FilterRule):
    """Cull primitives based on a probability map sampled at centroids.

    The map (shape ``(B, 1, H, W)``) is bilinearly sampled at each
    primitive's centroid to produce a per-primitive probability in
    ``[0, 1]``. Each primitive is then KEPT or CULLED by a Bernoulli
    draw against its sampled probability.

    Use this rule when culling intensity should vary spatially — for
    example, to concentrate primitives in detailed regions and cull
    them in flat regions.

    Attributes:
        map: Probability map (B, 1, H, W). Values in [0, 1].
        interval: Fire every N invocations of ``__call__``.

    Notes:
        - Only the first batch element of the map (``map[0]``) is sampled.
          Use a batch dimension of 1 unless you want different per-batch
          behaviour.
        - Centroids are assumed to be in ``[0, 1]`` (matching how
          ``CubicFanPrimitive``, ``RadialFreqPrimitive``, and ``Gaussian``
          initialise them via ``torch.rand``).
    """

    def __init__(
        self,
        map: Float[Tensor, "B 1 H W"],
        interval: int = 1,
    ):
        super().__init__(interval=interval)
        self.map = map

    def criterion(self, primitive: Primitive, **kwargs) -> Float[Tensor, "N"]:
        """Sample the probability map at each primitive's centroid.

        Args:
            primitive: Primitive whose centroids define the sample locations.

        Returns:
            Per-primitive probabilities (N,).
        """
        sampled = ImgUtils.uv_sample(self.map, primitive.centroids)
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
