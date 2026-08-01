import math

import torch

from typing import Tuple, Dict

from jaxtyping import Bool, Float, Integer
from torch import Tensor

from .base import Primitive, cached_property, ParamDef

from .initializers.base import Initializer


class RadialFreqInitializer(Initializer):
    def init_param(
        self, name: str, param_shape: Tuple[int], batched: bool
    ) -> Float[Tensor, "N ..."]:
        if name == "freq":
            return (1.0 + torch.rand(param_shape) * 3) * torch.pi
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
        return dict(
            thetas=ParamDef(True, True, None),
            centroids=ParamDef(True, True, (2,), 0.5),
            sigma=ParamDef(True, True, None, scalable=True),
            freq=ParamDef(True, True, None),
            color_1=ParamDef(True, True, (3,)),
            color_2=ParamDef(True, True, (3,)),
            alphas=ParamDef(True, True, None),
        )

    @property
    def default_initializers(self) -> Dict[str, Initializer] | Initializer:
        return RadialFreqInitializer()

    @cached_property
    def scales(self) -> Tuple[Float[Tensor, "N"], Float[Tensor, "N"]]:
        s = self.sigma * self._sigma_cutoff
        return (s, s)

    @cached_property
    def areas(self) -> Float[Tensor, "N"]:
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
    def patch_mask(
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
        deltas = co[:, None, :] - centroids[None, :, :]
        dists = deltas.norm(dim=-1)
        angles = torch.atan2(deltas[..., 1], deltas[..., 0])
        phase = (2 * math.pi * self.thetas)[None, :] + freq[None, :] * angles
        modulation = torch.sin(phase)
        gauss = torch.exp(-(dists**2) / (2 * sigma**2 + 1e-8))
        return gauss * modulation * alpha[None, :]
