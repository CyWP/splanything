"""Regular polygon primitive with axis-projection Gaussian falloff."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from jaxtyping import Bool, Float, Integer
from torch import Tensor

from .base import Primitive, cached_property, ParamDef


class PolygonPrimitive(Primitive):
    """Regular polygon primitive: Gaussian falloff on the nearest-axis projection.

    Each instance derives ``n_sides`` unit axes from ``thetas`` by rotating a
    reference axis ``(cos(thetas), sin(thetas))`` in regular ``2*pi/n_sides``
    steps around the circle. The weight at a coordinate is a Gaussian falloff
    of the centroid-to-coordinate offset projected onto the axis it most
    closely aligns with — the maximum dot product over the axes, which is
    always the positive-aligned projection. Level sets of that projection are
    regular polygons (intersections of half-planes), so the falloff forms a
    smooth regular polygon.

    Attributes:
        centroids: Center positions (N, 2).
        thetas: Rotation of the reference axis in radians (N,).
        sigma: Falloff scale; inradius of the ``exp(-0.5)`` weight level set (N,).
        color: Per-primitive color (N, 3).
        alphas: Peak opacity, attained at the centroid (N,).
        n_sides: Number of sides (and axes) of each polygon.

    Construction:
        PolygonPrimitive(size, n_sides=6, **kwargs): ``n_sides`` sets the polygon
        order (>= 3); other kwargs are forwarded to ``Primitive.__init__``.

    Notes:
        - ``sigma`` is shared by all axes (regular, equiangular polygons).
        - Level set at weight ``w`` is a regular ``n_sides``-gon with
          inradius ``sigma * sqrt(2 * ln(1 / w))`` and circumradius
          ``inradius / cos(pi / n_sides)``; vertices point along the
          bisectors between adjacent axes.
    """

    _sigma_cutoff = 2.5

    def __init__(
        self,
        size: int = 1,
        n_sides: int = 6,
        **kwargs,
    ):
        """Initialize the primitive.

        Args:
            size: Number of primitives.
            n_sides: Number of sides (and axes); must be >= 3.
            **kwargs: Forwarded to ``Primitive.__init__``.

        Raises:
            ValueError: If ``n_sides`` < 3.
        """
        if n_sides < 3:
            raise ValueError(f"n_sides must be >= 3 for a polygon, got {n_sides}.")
        self._n_sides = n_sides
        super().__init__(size=size, **kwargs)

    @property
    def default_params(self) -> Dict[str, ParamDef]:
        """Parameter definitions for this primitive."""
        return dict(
            centroids=ParamDef(True, True, (2,), 0.5),
            thetas=ParamDef(True, True, None),
            sigma=ParamDef(True, True, None, scalable=True),
            color=ParamDef(True, True, (3,)),
            alphas=ParamDef(True, True, None),
        )

    @cached_property
    def axes(self) -> Float[Tensor, "N A 2"]:
        """Unit axes of each polygon.

        Returns:
            out: Axes (N, n_sides, 2): the reference axis
            ``(cos(thetas), sin(thetas))`` rotated in regular
            ``2*pi/n_sides`` steps around the circle.
        """
        steps = torch.arange(
            self._n_sides, device=self.device, dtype=self.thetas.dtype
        )  # (A,)
        angles = self.thetas[:, None] + steps[None, :] * (
            2 * math.pi / self._n_sides
        )  # (N, A)
        return torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # (N, A, 2)

    @cached_property
    def circumradii(self) -> Float[Tensor, "N"]:
        """Circumradius of the ``exp(-0.5)`` weight level set of each polygon.

        Returns:
            out: Circumradii (N,) = ``sigma / cos(pi / n_sides)``.
        """
        return self.sigma / math.cos(math.pi / self._n_sides)

    @cached_property
    def scales(self) -> Tuple[Float[Tensor, "N"], Float[Tensor, "N"]]:
        """Scale parameters used by refinement/splitting.

        Returns:
            out: Tuple of (sigma * _sigma_cutoff, sigma * _sigma_cutoff),
            each (N,); the polygon is isotropic so both entries match.
        """
        s = self.sigma * self._sigma_cutoff
        return (s, s)

    @cached_property
    def areas(self) -> Float[Tensor, "N"]:
        """Approximate area of each primitive.

        Returns:
            out: Tensor of shape (N,) with the area of the cutoff level
            set: ``n_sides * (sigma * _sigma_cutoff)**2 * tan(pi / n_sides)``
            (regular polygon area from its inradius).
        """
        r = self._sigma_cutoff * self.sigma
        return self._n_sides * r**2 * math.tan(math.pi / self._n_sides)

    @torch.no_grad()
    def _raw_patch_mask(
        self,
        centers: Float[Tensor, "P 2"],
        patch_sizes: Integer[Tensor, "P"],
        H: Integer[Tensor, "P"],
        W: Integer[Tensor, "P"],
    ) -> Bool[Tensor, "P N"]:
        """Compute mask for valid patches at given centers.

        Args:
            centers: Patch center coordinates (P, 2).
            patch_sizes: Size of patches (P,).
            H: Image heights (P,).
            W: Image widths (P,).

        Returns:
            out: Bool tensor (P, N) indicating which primitives are valid
            for a given patch, using the cutoff-level circumradius as extent.
        """
        circum = (
            self._sigma_cutoff * self.sigma / math.cos(math.pi / self._n_sides)
        )  # (N,)
        unit_patches = patch_sizes / torch.minimum(H, W)  # (P,)
        dists = (centers[:, None, :] - self.centroids[None, :, :]).norm(dim=2)  # (P, N)
        return dists - unit_patches[:, None] < circum[None, :]

    def sample_rgb(
        self,
        co: Float[Tensor, "Nc 2"],
        **kwargs,
    ) -> Float[Tensor, "Nc Np 3"]:
        """Sample per-primitive colors at coordinates.

        Args:
            co: Coordinates to sample at (Nc, 2).

        Returns:
            out: RGB tensor (Nc, Np, 3): each primitive's constant color.
        """
        return self.color[None, :, :].expand(co.shape[0], -1, -1)

    def sample_weights(
        self,
        co: Float[Tensor, "Nc 2"],
        **kwargs,
    ) -> Float[Tensor, "Nc N"]:
        """Sample per-primitive weights at coordinates.

        Gaussian falloff of the offset's projection onto the most closely
        aligned axis (maximum dot product over the ``n_sides`` axes, always
        the positive-aligned projection), scaled by alpha.

        Args:
            co: Coordinates to sample at (Nc, 2).

        Returns:
            out: Weights tensor (Nc, Np).
        """
        centroids = self.centroids
        sigma = self.sigma
        alpha = self.alphas
        deltas = co[:, None, :] - centroids[None, :, :]  # (Nc, N, 2)
        dots = torch.einsum("nid,iad->nia", deltas, self.axes)  # (Nc, N, A)
        proj = dots.amax(dim=-1)  # (Nc, N) positive-aligned projection
        weights = torch.exp(-(proj**2) / (2 * sigma**2 + 1e-8))
        return weights * alpha[None, :]
