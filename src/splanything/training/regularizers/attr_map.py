from typing import Optional

import torch
from jaxtyping import Float
from torch import Tensor

from ...utils.img import Splimage
from ...primitives.base import Primitive
from .base import Regularizer


class AttributeMap(Regularizer):
    """Penalise deviation of a primitive attribute from a spatial map.

    Samples the provided ``map`` at the primitive's coordinate attribute
    (default ``"centroids"``) to obtain per-primitive targets, then
    penalises the squared deviation of the named attribute from those
    targets.

    The sampled map is squeezed on its channel dimension when it has
    size 1 so that it broadcasts against scalar (N,) attributes.  For
    multi-channel attributes the map's channel count must match.

    Attributes:
        map: Splimage whose values serve as per-primitive targets.
        attr: Name of the batched primitive attribute to regularize.
        coord_attr: Name of the batched coordinate attribute used to
            sample ``map`` (default ``"centroids"``).

    Notes:
        - Operates directly on a ``Primitive``; the caller is
          responsible for any loss weighting.
    """

    def __init__(
        self,
        map: Splimage,
        attr: str,
        coord_attr: str = "centroids",
        weight_map: Optional[Splimage] = None,
    ):
        """Initialize the regularizer.

        Args:
            map: Splimage whose sampled values are the per-primitive
                targets.  Sampled at the primitive's ``coord_attr``.
            attr: Name of the batched attribute to compare against the
                sampled map.
            coord_attr: Name of the batched coordinate attribute used
                to sample ``map`` (default ``"centroids"``).
            weight_map: Optional spatial map (B, 1, H, W) sampled at
                the primitive's ``coord_attr`` (or at an explicit
                ``co``) to spatially weight the regularization (see
                :class:`Regularizer`).
        """
        super().__init__(weight_map=weight_map, coords_attr=coord_attr)
        self.map = map
        self.attr = attr
        self.coord_attr = coord_attr

    def compute(self, primitive: Primitive) -> Float[Tensor, " N"]:
        """Compute the map-based regularization.

        Args:
            primitive: Primitive whose named attribute is compared
                against the sampled map.

        Returns:
            Per-primitive regularization tensor of shape ``(N,)``;
            ``forward`` reduces it to a scalar by averaging across
            primitives (after the optional ``weight_map`` sampling).
        """
        val = getattr(primitive, self.attr)
        co = getattr(primitive, self.coord_attr)
        target = self.map.mask_sample(co)[0].squeeze(0)  # (N, C) or (N,)
        if target.ndim > 1 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        sqdev = (val - target) ** 2
        if sqdev.ndim > 1:
            sqdev = sqdev.mean(dim=tuple(range(1, sqdev.ndim)))
        return sqdev
