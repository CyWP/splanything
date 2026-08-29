from typing import Callable, Dict, Literal, Optional, Union

import torch
from jaxtyping import Float
from torch import Tensor

from ...utils.img import Splimage
from ...primitives.base import Primitive
from .base import Regularizer


class AttributeAttractor(Regularizer):
    """Weight primitive attribute deviations by distance to a set of target points.

    The selected coordinate attribute (default ``"centroids"``) defines one
    position per primitive. The user provides target points ``f`` (M, 2)
    and a mapping ``attrs`` of additional attribute names to target values
    (channel-dim scalar or tensor; tensors are unsqueezed on dim 0 to
    broadcast against the batched primitive parameter).

    For each primitive, the Euclidean distance from its coordinate to ``f``
    is aggregated via ``MIN``/``MIN_K``/``MEAN``/``MAX``/``MAX_K`` and
    mapped through ``compute_force`` (default ``exp(-dist)``) into a
    per-primitive force weight. The weight scales the squared deviation of
    each ``attrs`` entry from its target; ``mode`` controls the
    interpretation:

    - ``ATTRACT`` (default): ``force * (val - target)^2`` -- primitives
      near ``f`` are pulled toward ``target``.
    - ``PUSH``: ``force * exp(-(val - target)^2)`` -- primitives near
      ``f`` are pushed away from ``target``.
    - ``NEITHER``: ``(val - target)^2`` -- force is dropped, plain
      squared deviation.

    Notes:
        - Operates directly on a ``Primitive``; the caller is responsible
          for any loss weighting.
        - ``compute_force`` is a method that subclasses can override.
          Passing ``force_fn`` to the constructor binds that callable as
          a per-instance override of ``compute_force`` for this object
          only.

    Warnings:
        - The coordinate attribute is used only to compute the force
          weight; it is not itself included in ``attrs``. Add a coordinate
          (e.g. ``"centroids"``) entry in ``attrs`` only if you want the
          coordinate to be pulled toward a channel-dim target.
    """

    def __init__(
        self,
        f: Float[Tensor, "M 2"],
        coord_attr: str = "centroids",
        attrs: Optional[Dict[str, Union[float, Float[Tensor, "..."]]]] = None,
        mode: Literal["ATTRACT", "PUSH", "NEITHER"] = "ATTRACT",
        agg: Literal["MIN", "MIN_K", "MEAN", "MAX", "MAX_K"] = "MIN",
        k: int = 1,
        force_fn: Optional[Callable[[Float[Tensor, "N"]], Float[Tensor, "N"]]] = None,
        weight_map: Optional[Splimage] = None,
    ):
        """Initialize the attractor regularizer.

        Args:
            f: Target points (M, 2) in normalised pixel-centre coordinates.
            coord_attr: Name of the batched coordinate attribute (default
                ``"centroids"``).
            attrs: Dict mapping each batched attribute name to its target
                (scalar or tensor on the channel dim; tensors are
                unsqueezed on dim 0 for broadcasting).
            mode: Force interpretation -- ``ATTRACT``/``PUSH``/``NEITHER``.
            agg: Distance aggregation mode over the ``M`` target points.
            k: Number of points used for ``MIN_K``/``MAX_K`` (ignored by
                ``MIN``/``MEAN``/``MAX``).
            force_fn: Optional callable mapping aggregated distance to a
                per-primitive force weight. When provided, overrides
                ``self.compute_force`` on this instance; otherwise the
                class method (default ``exp(-dist)``) is used.
            weight_map: Optional spatial map (B, 1, H, W) sampled at the
                primitive's coordinate attribute (or at an explicit
                ``co``) to spatially weight the regularization (see
                :class:`Regularizer`).
        """
        super().__init__(weight_map=weight_map)
        self.coord_attr = coord_attr
        self.attrs = {}
        if attrs is not None:
            for name, target in attrs.items():
                self.attrs[name] = (
                    target.unsqueeze(0) if isinstance(target, Tensor) else target
                )
        self.mode = mode
        self.agg = agg
        self.k = int(k)

        if f.ndim != 2 or f.shape[1] != 2:
            raise ValueError(f"`f` must have shape (M, 2); got {tuple(f.shape)}.")
        if f.shape[0] < 1:
            raise ValueError("`f` must contain at least one point.")
        self.register_buffer("f", f.detach().clone())

        if force_fn is not None:
            self.compute_force = lambda d, _fn=force_fn: _fn(d)  # noqa: E731

    def compute_force(
        self,
        d: Float[Tensor, " N"],
    ) -> Float[Tensor, " N"]:
        """Map an aggregated per-primitive distance to a force weight.

        Default implementation returns ``exp(-d)``. Subclasses may
        override this method to customise the force kernel; an instance
        attribute set from the ``force_fn`` constructor argument takes
        precedence over a class-level override.

        Args:
            d: Aggregated distance (N,) from each primitive's coordinate
                attribute to the target points ``f``.

        Returns:
            Per-primitive force weight (N,).
        """
        return torch.exp(-d)

    def compute(self, primitive: Primitive) -> Float[Tensor, " N"]:
        """Compute the attractor-based regularization.

        Args:
            primitive: Primitive whose coordinate attribute and named
                attributes are evaluated.

        Returns:
            Per-primitive regularization tensor of shape ``(N,)``;
            ``forward`` reduces it to a scalar by averaging across
            primitives (after the optional ``weight_map`` sampling).
        """
        co = getattr(primitive, self.coord_attr)
        f = self.f.to(dtype=co.dtype, device=co.device)
        if co.shape[0] == 0 or f.shape[0] == 0:
            return co.new_zeros(co.shape[:1])

        dists = torch.cdist(co, f)  # (N, M)
        d = self._aggregate(dists)

        force = self.compute_force(d)  # (N,)

        loss = co.new_zeros(co.shape[:1])
        for name, target in self.attrs.items():
            val = getattr(primitive, name)
            loss = loss + self._attr_penalty(val, target, force)
        return loss

    def _aggregate(self, dists: Float[Tensor, "N M"]) -> Float[Tensor, " N"]:
        """Reduce per-point distances to a single per-primitive distance.

        Args:
            dists: Pairwise distances (N, M) between primitives and
                target points.

        Returns:
            Reduced distance (N,).
        """
        if self.agg == "MIN":
            return dists.min(dim=1).values
        if self.agg == "MIN_K":
            k = max(1, min(self.k, dists.shape[1]))
            return torch.topk(dists, k, largest=False).values.mean(dim=1)
        if self.agg == "MEAN":
            return dists.mean(dim=1)
        if self.agg == "MAX":
            return dists.max(dim=1).values
        if self.agg == "MAX_K":
            k = max(1, min(self.k, dists.shape[1]))
            return torch.topk(dists, k, largest=True).values.mean(dim=1)
        raise ValueError(
            f"Unknown agg '{self.agg}'; expected one of "
            f"'MIN', 'MIN_K', 'MEAN', 'MAX', 'MAX_K'."
        )

    def _attr_penalty(
        self,
        val: Float[Tensor, "N ..."],
        target: Float[Tensor, "1 ..."] | float,
        force: Float[Tensor, " N"],
    ) -> Float[Tensor, " N"]:
        """Compute a single attribute's contribution to the loss.

        Args:
            val: Primitive attribute values (N, ...).
            target: Target (scalar or tensor broadcastable against
                ``val`` via a leading singleton).
            force: Per-primitive force weight (N,).

        Returns:
            Per-primitive penalty tensor of shape ``(N,)``.
        """
        sqdev = (val - target) ** 2  # (N, ...)
        if sqdev.ndim > 1:
            sqdev = sqdev.mean(dim=tuple(range(1, sqdev.ndim)))
        if self.mode == "ATTRACT":
            return force * sqdev  # (N,)
        if self.mode == "PUSH":
            return force * torch.exp(-sqdev)  # (N,)
        if self.mode == "NEITHER":
            return sqdev  # (N,)
        raise ValueError(
            f"Unknown mode '{self.mode}'; expected one of 'ATTRACT', 'PUSH', 'NEITHER'."
        )
