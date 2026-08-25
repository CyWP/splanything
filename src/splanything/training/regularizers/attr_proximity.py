from typing import List, Literal, Optional

import torch
from jaxtyping import Float
from torch import Tensor

from ...utils.img import Splimage
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
        weight_map: Optional[Splimage] = None,
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

    def _ensure_dim(self, data: Float[Tensor, "N ..."]) -> Float[Tensor, "N ..."]:
        if data.ndim == 1:
            return data.unsqueeze(-1)
        return data

    def compute(self, primitive: Primitive) -> Float[Tensor, " N"]:
        """Compute the proximity regularization.

        Args:
            primitive: Primitive whose named attributes are compared.

        Returns:
            Per-primitive regularization tensor of shape ``(N,)``;
            ``forward`` reduces it to a scalar by averaging across
            primitives (after the optional ``weight_map`` sampling).
        """
        if self.mode == "RATIO":
            first = getattr(primitive, self.attr_names[0])
            second = getattr(primitive, self.attr_names[1])
            sqdev = (first - self.ratio * second) ** 2
            return sqdev.mean(dim=tuple(range(1, sqdev.ndim)))
        vals = torch.cat(
            [self._ensure_dim(getattr(primitive, name)) for name in self.attr_names],
            dim=-1,
        )
        dist = (vals - vals.mean(dim=-1, keepdim=True)) ** 2
        if self.mode == "PUSH":
            return torch.exp(-dist).mean(dim=-1)
        if self.mode == "ATTRACT":
            return dist.mean(dim=-1)
        raise AttributeError(
            "Attribute 'mode' must be one of 'PUSH', 'ATTRACT', 'RATIO'."
        )
