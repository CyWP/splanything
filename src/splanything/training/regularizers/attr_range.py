from typing import Optional

import torch
from jaxtyping import Float
from torch import Tensor

from ...utils.img import Splimage
from ...primitives.base import Primitive
from .base import Regularizer


class AttributeRange(Regularizer):
    """Soft-clip a primitive attribute to a [min, max] band or to a target.

    Returns a scalar penalty composed of:
    - Squared overshoot below ``min`` (only when ``min`` is set).
    - Squared overshoot above ``max`` (only when ``max`` is set).
    - Squared deviation from ``target`` (only when ``target`` is set).

    Each bound may be a scalar or a tensor matching the attribute shape;
    tensors are unsqueezed on dim=0 for broadcasting against the
    ``primitive``'s batched attribute.

    Attributes:
        attr_name: Name of the batched primitive attribute to clamp.
        min: Optional lower bound (scalar or tensor).
        max: Optional upper bound (scalar or tensor).
        target: Optional target value (scalar or tensor).

    Notes:
        - Operates directly on a ``Primitive``; the caller is
          responsible for any loss weighting.
    """

    def __init__(
        self,
        attr_name: str,
        min: Optional[float | Float[Tensor, "..."]] = None,
        max: Optional[float | Float[Tensor, "..."]] = None,
        target: Optional[float | Float[Tensor, "..."]] = None,
        weight_map: Optional[Splimage] = None,
    ):
        """Initialize the regularizer.

        Args:
            attr_name: Name of the batched attribute to regularize.
            min: Optional lower bound.
            max: Optional upper bound.
            target: Optional target value.
            weight_map: Optional spatial map (B, 1, H, W) sampled at
                the primitive's centroids (or at an explicit ``co``)
                to spatially weight the regularization.
        """
        super().__init__(weight_map=weight_map)
        self.attr_name = attr_name
        self.min = min.unsqueeze(0) if isinstance(min, Tensor) else min
        self.max = max.unsqueeze(0) if isinstance(max, Tensor) else max
        self.target = target.unsqueeze(0) if isinstance(target, Tensor) else target

    def compute(self, primitive: Primitive) -> Float[Tensor, " N"]:
        """Compute the range regularization.

        Args:
            primitive: Primitive whose named attribute is regularized.

        Returns:
            Per-primitive regularization tensor of shape ``(N,)``;
            ``forward`` reduces it to a scalar by averaging across
            primitives (after the optional ``weight_map`` sampling).
        """
        val = getattr(primitive, self.attr_name)
        loss = val.new_zeros(val.shape[:1])
        if self.min is not None:
            overshoot = torch.where(val < self.min, (val - self.min) ** 2, 0)
            loss = loss + self._reduce(overshoot)
        if self.max is not None:
            overshoot = torch.where(val > self.max, (val - self.max) ** 2, 0)
            loss = loss + self._reduce(overshoot)
        if self.target is not None:
            loss = loss + self._reduce((val - self.target) ** 2)
        return loss

    @staticmethod
    def _reduce(x: Tensor) -> Tensor:
        if x.ndim > 1:
            return x.mean(dim=tuple(range(1, x.ndim)))
        return x
