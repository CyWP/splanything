from typing import List, Literal, Optional

import torch
from jaxtyping import Float
from torch import Tensor

from ...primitives.base import Primitive
from .base import Regularizer


class AttributeProximity(Regularizer):
    """Penalise or reward proximity between primitives on selected attributes.

    Modes:
        - ``"PUSH"`` (default): concatenates the named attributes per
          primitive, computes the squared deviation from the
          per-primitive mean and returns ``exp(-dist).mean()`` —
          drives primitives apart on those attributes.
        - ``"ATTRACT"``: same setup but returns ``dist.mean()``,
          pulling primitives together on those attributes.
        - ``"RATIO"``: enforces an exact ratio relationship
          ``first == ratio * second`` between the first two entries
          of ``attr_names``. Returns ``(first - ratio * second)^2.mean()``.
          Extra entries in ``attr_names`` are ignored.

    Attributes:
        attr_names: Names of batched primitive attributes to compare.
        mode: ``"PUSH"`` (default), ``"ATTRACT"``, or ``"RATIO"``.
        ratio: Ratio value used by ``"RATIO"`` mode
            (``first == ratio * second``). Required in ``"RATIO"``
            mode; ignored otherwise.

    Notes:
        - Operates directly on a ``Primitive``; the caller is
          responsible for any loss weighting.
    """

    def __init__(
        self,
        attr_names: List[str],
        mode: Literal["ATTRACT", "PUSH", "RATIO"] = "PUSH",
        ratio: Optional[float] = None,
        weight_map: Optional[Float[Tensor, "B 1 H W"]] = None,
    ):
        """Initialize the regularizer.

        Args:
            attr_names: Names of batched attributes to compare.
                ``"RATIO"`` mode uses ``attr_names[0]`` and
                ``attr_names[1]``; extra names are ignored.
            mode: ``"PUSH"`` (default) drives primitives apart;
                ``"ATTRACT"`` pulls them together; ``"RATIO"``
                enforces ``attr_names[0] == ratio * attr_names[1]``.
            ratio: Ratio value (``first / second``) used by
                ``"RATIO"`` mode. Required in ``"RATIO"`` mode.
            weight_map: Optional spatial map (B, 1, H, W) sampled at
                the primitive's centroids (or at an explicit ``co``)
                to spatially weight the regularization.
        """
        super().__init__(weight_map=weight_map)
        self.attr_names = attr_names
        self.mode = mode
        self.ratio = ratio
        if self.mode == "RATIO":
            if self.ratio is None:
                raise ValueError("`ratio` is required when mode == 'RATIO'.")
            if len(self.attr_names) < 2:
                raise ValueError(
                    "`mode == 'RATIO'` requires at least two attribute names."
                )

    def compute(self, primitive: Primitive) -> Float[Tensor, ""]:
        """Compute the proximity regularization.

        Args:
            primitive: Primitive whose named attributes are compared.

        Returns:
            Scalar regularization tensor.
        """
        if self.mode == "RATIO":
            first = getattr(primitive, self.attr_names[0])
            second = getattr(primitive, self.attr_names[1])
            return ((first - self.ratio * second) ** 2).mean()
        vals = torch.cat(
            [getattr(primitive, name) for name in self.attr_names], dim=-1
        )
        dist = (vals - vals.mean(dim=-1, keepdim=True)) ** 2
        if self.mode == "PUSH":
            return torch.exp(-dist).mean()
        if self.mode == "ATTRACT":
            return dist.mean()
        raise AttributeError(
            "Attribute 'mode' must be one of 'PUSH', 'ATTRACT', 'RATIO'."
        )
