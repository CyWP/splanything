import torch

from typing import Optional

from jaxtyping import Bool, Float
from torch import Tensor

from ....primitives.base import Primitive
from ..base import FilterRule


class BoundsFilter(FilterRule):
    def __init__(
        self, margin: float = 0.0, interval: int = 10, use_areas: bool = False
    ):
        super().__init__(interval=interval)
        self.margin = margin
        self.use_areas = use_areas

    def criterion(self, primitive: Primitive, **kwargs) -> Float[Tensor, "N"]:
        c_min = primitive.centroids.min(dim=1).values
        c_max = primitive.centroids.max(dim=1).values
        if self.use_areas and hasattr(primitive, "areas"):
            half_extent = primitive.areas[:, None] ** 0.5 * 0.5
            c_min += half_extent
            c_max -= half_extent
        c_max = 1 - c_max
        return torch.max(c_min, c_max)

    def judge(self, criterion: Float[Tensor, "N"]) -> Optional[Bool[Tensor, "N"]]:
        return criterion > self.margin
