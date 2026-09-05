"""Star-shaped primitive with angular ray falloff."""

from __future__ import annotations
from typing import Tuple, Dict

import math
import torch
from jaxtyping import Bool, Float, Integer
from torch import Tensor

from .base import Primitive, cached_property, ParamDef


class StarPrimitive(Primitive):
    """Star-shaped primitive with exponential radial falloff.

    Each instance is an oriented star: ``thetas`` rotates a pair of
    perpendicular axes, and the radial falloff range varies with the
    angle to the nearest star axis, producing ``n_axes`` pointed rays.

    Attributes:
        centroids: Center positions (N, 2).
        thetas: Rotation of each star in radians (N,).
        range_1: Extent along the first axis (N,).
        range_2: Extent along the second axis (N,).
        color: Per-primitive color (N, 3).
        alphas: Opacity (N,).
        n_axes: Number of star points.

    Construction:
        StarPrimitive(size, n_axes=2, **kwargs): ``n_axes`` sets the
        number of star points; other kwargs are forwarded to
        ``Primitive.__init__``.
    """

    def __init__(
        self,
        size: int = 1,
        n_axes: int = 2,
        **kwargs,
    ):
        """Initialize the primitive.

        Args:
            size: Number of primitives.
            n_axes: Number of star points.
            **kwargs: Forwarded to ``Primitive.__init__``.
        """
        self._n_axes = n_axes
        super().__init__(size=size, **kwargs)

    @property
    def default_params(self) -> Dict[str, ParamDef]:
        """Parameter definitions for this primitive."""
        return dict(
            centroids=ParamDef(True, True, (2,), 0.5),
            thetas=ParamDef(True, True, None),
            range_1=ParamDef(True, True, None, scalable=True),
            range_2=ParamDef(True, True, None, scalable=True),
            color=ParamDef(True, True, (3,)),
            alphas=ParamDef(True, True, None),
        )

    @cached_property
    def ranges(self) -> Tuple[Float[Tensor, "N"], Float[Tensor, "N"]]:
        """Sorted absolute extents per primitive.

        Returns:
            Tuple of (rng_min, rng_max), each (N,).
        """
        rng_1, rng_2 = self.range_1.abs(), self.range_2.abs()
        rng_mask = rng_1 < rng_2
        rng_min = torch.where(rng_mask, rng_1, rng_2)
        rng_max = torch.where(rng_mask, rng_2, rng_1)
        return rng_min, rng_max

    @cached_property
    def rng_1(self) -> Tuple[Float[Tensor, "N"]]:
        """Minimum of the two absolute ranges (N,)."""
        return self.ranges[0]

    @cached_property
    def rng_2(self) -> Tuple[Float[Tensor, "N"]]:
        """Maximum of the two absolute ranges (N,)."""
        return self.ranges[1]

    @cached_property
    def axes(self) -> Tuple[Float[Tensor, "N 2"], Float[Tensor, "N 2"]]:
        """Perpendicular axes from the star rotation.

        Returns:
            Tuple of (ax_1, ax_2), each (N, 2); ax_2 is ax_1 rotated
            90 degrees counterclockwise.
        """
        c = torch.cos(self.thetas)
        s = torch.sin(self.thetas)
        ax_1 = torch.stack([c, s], dim=-1)
        ax_2 = torch.stack([-s, c], dim=-1)
        return ax_1, ax_2

    @cached_property
    def axis_angles(self) -> Float[Tensor, "N A"]:
        """Angles of the star's axis directions.

        Returns:
            Angles (N, n_axes), evenly spaced over 2*pi starting at
            ``thetas``.
        """
        n = self._n_axes
        return (
            self.thetas[:, None]
            + torch.linspace(
                0, 2 * math.pi, n + 1, device=self.device, dtype=self.thetas.dtype
            )[:-1][None, :]
        )

    @cached_property
    def areas(self) -> Float[Tensor, "N"]:
        """Approximate area of each primitive (2 * |range_1 * range_2|)."""
        return (2 * self.range_1 * self.range_2).abs()

    @cached_property
    def scales(self) -> Tuple[Float[Tensor, "N"], Float[Tensor, "N"]]:
        """Scale parameters used by refinement/splitting.

        Returns:
            Tuple of (range_1, range_2), each (N,).
        """
        return (self.range_1, self.range_2)

    @torch.no_grad()
    def _raw_patch_mask(
        self,
        centers: Float[Tensor, "P 2"],
        patch_sizes: Integer[Tensor, "P"],
        H: Integer[Tensor, "P"],
        W: Integer[Tensor, "P"],
    ) -> Bool[Tensor, "P N"]:
        rng = torch.maximum(self.range_1, self.range_2)
        unit_patches = patch_sizes / torch.minimum(H, W)
        dists = (centers[:, None, :] - self.centroids[None, :, :]).norm(dim=2)
        return dists - unit_patches[:, None] < rng[None, :]

    def sample_rgb(
        self,
        co: Float[Tensor, "Nc 2"],
        **kwargs,
    ) -> Float[Tensor, "Nc N 3"]:
        """Sample per-primitive colors at coordinates.

        Args:
            co: Coordinates to sample at (Nc, 2).

        Returns:
            RGB tensor (Nc, Np, 3): each primitive's constant color.
        """
        return self.color[None, :, :].expand(co.shape[0], -1, -1)

    def sample_weights(
        self,
        co: Float[Tensor, "Nc 2"],
        **kwargs,
    ) -> Float[Tensor, "Nc N"]:
        """Sample per-primitive weights at coordinates.

        Exponential falloff of the radial distance over an angle-dependent
        range alternating between the two ranges across ``n_axes`` rays,
        scaled by alpha.

        Args:
            co: Coordinates to sample at (Nc, 2).

        Returns:
            Weights tensor (Nc, Np).
        """
        centroids = self.centroids
        alpha = self.alphas
        rng_min, rng_max = self.ranges
        deltas = co[:, None, :] - centroids[None, :, :]
        dist = deltas.norm(dim=-1)

        angles = (
            torch.atan2(deltas[..., 1], deltas[..., 0]) - self.thetas[None, :]
        ) % (torch.pi / self._n_axes) - (torch.pi / self._n_axes / 2)
        range_weights = 1 - 2 * torch.sin(angles**2) / torch.pi
        angular_ranges = range_weights * rng_min + (1 - range_weights) * rng_max
        weights = torch.exp(-(dist) / angular_ranges.clamp(min=1e-6))
        return weights * alpha[None, :]
