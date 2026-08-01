from torch import Tensor
from jaxtyping import Float
from typing import Callable, Optional

from ..base import CriterionProcessor, RefinementRule
from ....primitives.base import Primitive
from ....utils.img import ImgUtils


class MapCriterionProcessor(CriterionProcessor):
    """Modify a refinement criterion by sampling a spatial map.

    The map (B, 1, H, W) is bilinearly sampled at each primitive's
    coordinate (default: centroid), reshaped to match the criterion shape,
    and combined via ``proc_fn`` (default: element-wise multiplication).

    Attributes:
        map: Spatial map (B, 1, H, W).
        coords_attr: Attribute name for sampling coordinates (default ``"centroids"``).
        proc_fn: Callable ``(map_values, criterion) -> modified_criterion``.
            Defaults to element-wise multiplication.

    Notes:
        - Only ``map[0]`` (first batch element) is sampled.
        - Coordinates must be in [0, 1] UV space.
    """

    def __init__(
        self,
        map: Float[Tensor, "B 1 H W"],
        coords_attr: str = "centroids",
        proc_fn: Optional[
            Callable[Float[Tensor, "N ..."], Float[Tensor, "N ..."]]
        ] = None,
    ):
        """
        Args:
            map: Spatial map (B, 1, H, W).
            coords_attr: Attribute name for sampling coordinates (default ``"centroids"``).
            proc_fn: Combination function ``(map, criterion) -> criterion``.
                Defaults to ``map * criterion``.
        """
        self.map = map
        self.coords_attr = coords_attr
        self.proc_fn = lambda x, y: x * y if proc_fn is None else proc_fn

    def apply(
        self,
        primitive: Primitive,
        rule: RefinementRule,
        criterion: Float[Tensor, "N ..."],
        **kwargs,
    ) -> Float[Tensor, "N ..."]:
        """Sample map and combine with the criterion.

        Args:
            primitive: Primitive whose coordinates define sample locations.
            rule: Parent refinement rule (unused by default).
            criterion: Per-primitive criterion (N, ...).

        Returns:
            Modified criterion (N, ...).
        """
        sampled = ImgUtils.uv_sample(
            self.map, getattr(primitive, self.coords_attr)
        ).view(criterion.shape)
        return self.proc_fn(sampled, criterion)
