"""Radial-frequency splat primitive with angular ray modulation."""

import math

import torch

from typing import Tuple, Dict

from jaxtyping import Bool, Float, Integer
from torch import Tensor

from .base import Primitive, cached_property, ParamDef

from .initializers.base import Initializer


class RadialFreqInitializer(Initializer):
    """Initializer sampling ``freq`` and ``floor`` for radial frequency splats."""

    def init_param(
        self, name: str, param_shape: Tuple[int], batched: bool
    ) -> Float[Tensor, "N ..."]:
        """Initialize a parameter tensor.

        Args:
            name: Parameter name.
            param_shape: Shape of the parameter tensor.
            batched: Whether the parameter has a batch dimension.

        Returns:
            Initialized tensor; ``freq`` is scaled by pi, ``floor`` is
            standard normal; other names fall back to ``Initializer``.
        """
        if name == "freq":
            return (torch.randn(param_shape)) * torch.pi
        if name == "floor":
            return torch.randn(param_shape)
        return super().init_param(name, param_shape, batched)


class RadialFreqPrimitive(Primitive):
    """Radial-frequency splat: a sun-like primitive with angular ray modulation.

    The weight falloff is an isotropic Gaussian on the radial distance from
    the centroid (using a single ``sigma`` scale), modulated by a sinusoid of
    the angle ``theta(centroid, co)`` from the centroid to the coordinate.
    The number of rays is controlled by ``freq`` and their phase offset by
    ``thetas``.

    Attributes:
        thetas: Per-primitive ray phase offset in [0, 1] (scaled by 2*pi).
        centroids: Center positions (N, 2).
        sigma: Single radial falloff scale (N,).
        freq: Per-primitive angular frequency of the ray modulation (N,).
        color: Per-primitive color (N, 3).
        alphas: Per-primitive opacity (N,).
    """

    _sigma_cutoff = 2.5

    @property
    def default_params(self) -> Dict[str, ParamDef]:
        """Parameter definitions for this primitive."""
        return dict(
            thetas=ParamDef(True, True, None),
            centroids=ParamDef(True, True, (2,), 0.5),
            sigma=ParamDef(True, True, None, scalable=True),
            floor=ParamDef(True, True, None, scalable=True),
            freq=ParamDef(True, True, None),
            color_1=ParamDef(True, True, (3,)),
            color_2=ParamDef(True, True, (3,)),
            alphas=ParamDef(True, True, None),
        )

    @property
    def default_initializers(self) -> Dict[str, Initializer] | Initializer:
        """Default initializer: ``RadialFreqInitializer`` for all parameters."""
        return RadialFreqInitializer()

    @cached_property
    def scales(self) -> Tuple[Float[Tensor, "N"], Float[Tensor, "N"]]:
        """Scale parameters used by refinement/splitting.

        Returns:
            Tuple of (sigma * _sigma_cutoff, sigma * _sigma_cutoff), each (N,).
        """
        s = self.sigma * self._sigma_cutoff
        return (s, s)

    @cached_property
    def areas(self) -> Float[Tensor, "N"]:
        """Approximate area of each primitive.

        Returns:
            Tensor of shape (N,) with area values
            (sigma * _sigma_cutoff)**2 * pi.
        """
        return (self.sigma * self._sigma_cutoff) ** 2 * math.pi

    @cached_property
    def orientations(self) -> Float[Tensor, "N 2"]:
        """Unit direction vectors along each primitive's ray phase offset.

        Returns:
            Direction vectors (N, 2) computed from thetas as
            (cos(2*pi*thetas), sin(2*pi*thetas)).
        """
        ang = 2 * math.pi * self.thetas
        return torch.stack([torch.cos(ang), torch.sin(ang)], dim=1)

    @torch.no_grad()
    def _raw_patch_mask(
        self,
        centers: Float[Tensor, "P 2"],
        patch_sizes: Integer[Tensor, "P"],
        H: Integer[Tensor, "P"],
        W: Integer[Tensor, "P"],
    ) -> Bool[Tensor, "P N"]:
        unit_patches = patch_sizes / torch.minimum(H, W)
        dists = (centers[:, None, :] - self.centroids[None, :, :]).norm(dim=2)
        return dists - unit_patches[:, None] < self._sigma_cutoff * self.sigma[None, :]

    def sample_rgb(
        self,
        co: Float[Tensor, "Nc 2"],
        **kwargs,
    ) -> Float[Tensor, "Nc Np 3"]:
        """Sample per-primitive colors at coordinates.

        Color interpolates between ``color_1`` and ``color_2`` with the
        ray phase of the coordinate relative to its centroid.

        Args:
            co: Coordinates to sample at (Nc, 2).

        Returns:
            RGB tensor (Nc, Np, 3).
        """
        centroids = self.centroids
        deltas = co[:, None, :] - centroids[None, :, :]
        angles = torch.atan2(deltas[..., 1], deltas[..., 0]) / torch.pi
        phase = (self.thetas[None, :] + angles)[:, :, None] % 1.0
        return self.color_1[None, :, :] * phase + self.color_2[None, :, :] * (1 - phase)

    def sample_weights(
        self,
        co: Float[Tensor, "Nc 2"],
        **kwargs,
    ) -> Float[Tensor, "Nc N"]:
        """Sample modulated weights at coordinates.

        Computes an isotropic Gaussian falloff on radial distance and
        modulates it by ``sin(2*pi*thetas + freq * theta(centroid, co))``,
        then scales by alpha.

        Args:
            co: Coordinates to sample at (Nc, 2).

        Returns:
            Weights tensor (Nc, N).
        """
        centroids = self.centroids
        sigma = self.sigma
        alpha = self.alphas
        freq = self.freq
        floor = torch.sigmoid(self.floor)
        deltas = co[:, None, :] - centroids[None, :, :]
        dists = deltas.norm(dim=-1)
        angles = torch.atan2(deltas[..., 1], deltas[..., 0])
        phase = (self.thetas)[None, :] + freq[None, :] * angles
        modulation = (0.5 * torch.sin(phase) + 0.5) * (1 - floor) + floor
        gauss = torch.exp(-(dists**2) / (2 * sigma**2 + 1e-8))
        return gauss * modulation * alpha[None, :]
