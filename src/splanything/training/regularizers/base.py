from typing import Optional

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from ...primitives.base import Primitive
from ...utils.img import Splimage


class Regularizer(nn.Module):
    """Base class for primitive regularizers.

    A Regularizer computes a scalar value penalising some structural
    property of a ``Primitive``'s parameters (e.g., per-attribute
    spread or attribute range).

    Notes:
        - Subclasses must implement
          ``compute(primitive) -> Float[Tensor, ""]``.
        - Optional spatial weighting via ``weight_map``: when set, the
          map is sampled at the primitive's centroids (or at an
          explicit ``co`` if provided) and the sampled values
          multiply the result. Useful when regularization strength
          should vary spatially (e.g., encourage proximity more in
          detailed regions).
        - Regularizer weighting (scalar) is the responsibility of the
          caller; subclasses carry no scalar ``weight`` argument or
          attribute.
    """

    def __init__(
        self,
        weight_map: Optional[Splimage] = None,
        coords_attr: str = "adjusted_coords",
    ):
        """Initialize the regularizer.

        Args:
            weight_map: Optional spatial map (B, 1, H, W) sampled at
                the primitive's centroids (or at the ``co`` passed to
                ``forward``) to spatially weight the regularization.
                Not premultiplied by any scalar weight.
        """
        super().__init__()
        self.weight_map = weight_map
        self._coords_attr = coords_attr

    def compute(self, primitive: Primitive) -> Float[Tensor, ""]:
        """Compute unweighted regularization value.

        Args:
            primitive: Primitive whose parameters are evaluated.

        Returns:
            Regularization scalar tensor.
        """
        raise NotImplementedError()

    def forward(
        self,
        primitive: Primitive,
        co: Optional[Float[Tensor, "N 2"]] = None,
        **kwargs,
    ) -> Float[Tensor, ""]:
        """Compute the regularization, optionally weighted by ``weight_map``.

        Args:
            primitive: Primitive whose parameters are evaluated.
            co: Optional coordinates used to sample ``weight_map``.
                Defaults to ``primitive.centroids`` when ``weight_map``
                is set and ``co`` is not provided.
            **kwargs: Accepted for call-site compatibility; unused by
                the base implementation.

        Returns:
            Regularization value (scalar when ``weight_map`` is
            unused; otherwise broadcasted against the sampled weight
            tensor).
        """
        out = self.compute(primitive)
        if self.weight_map is not None:
            sample_at = getattr(primitive, self._coords_attr) if co is None else co
            return out * self.map.mask_sample(sample_at)[0].squeeze(-1)
        return out
