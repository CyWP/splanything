from typing import List, Literal, Optional

import torch
from jaxtyping import Float
from torch import Tensor

from ...primitives.base import Primitive
from .base import Regularizer


class AttributeProximity(Regularizer):
    """Penalise or reward proximity between primitives on selected attributes.

    Concatenates the named attributes per primitive, computes the
    squared deviation from the per-primitive mean, then returns either
    a PUSH term (``exp(-dist).mean()`` — drives primitives apart) or
    an ATTRACT term (``dist.mean()`` — pulls primitives together).

    Attributes:
        attr_names: Names of batched primitive attributes to compare.
        mode: ``"PUSH"`` (default) or ``"ATTRACT"``.

    Notes:
        - Operates directly on a ``Primitive``; the caller is
          responsible for any loss weighting.
    """

    def __init__(
        self,
        attr_names: List[str],
        mode: Literal["ATTRACT", "PUSH"] = "PUSH",
        weight_map: Optional[Float[Tensor, "B 1 H W"]] = None,
    ):
        """Initialize the regularizer.

        Args:
            attr_names: Names of batched attributes to compare.
            mode: ``"PUSH"`` (default) drives primitives apart;
                ``"ATTRACT"`` pulls them together.
            weight_map: Optional spatial map (B, 1, H, W) sampled at
                the primitive's centroids (or at an explicit ``co``)
                to spatially weight the regularization.
        """
        super().__init__(weight_map=weight_map)
        self.attr_names = attr_names
        self.mode = mode

    def compute(self, primitive: Primitive) -> Float[Tensor, ""]:
        """Compute the proximity regularization.

        Args:
            primitive: Primitive whose named attributes are compared.

        Returns:
            Scalar regularization tensor.
        """
        vals = torch.cat([getattr(primitive, name) for name in self.attr_names], dim=-1)
        dist = (vals - vals.mean(dim=-1, keepdim=True)) ** 2
        if self.mode == "PUSH":
            return torch.exp(-dist).mean()
        if self.mode == "ATTRACT":
            return dist.mean()
        raise AttributeError("Attribute 'mode' must either equal 'PUSH' or 'ATTRACT'.")
