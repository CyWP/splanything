import math

import torch

from typing import Tuple, Optional

from jaxtyping import Bool, Float, Integer
from torch import Tensor

from ..utils.pytorch import TensorIndex1D
from .base import Primitive, cached_property


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

    def __init__(self, size: int = 1, scale_factor: float = 1.0):
        """Initialize RadialFreqPrimitive parameters.

        Args:
            size: Number of radial-frequency splats to create.
        """
        super().__init__()
        area_factor = scale_factor / size**0.5
        self.add_parameter("thetas", torch.rand((size,)), batched=True, trainable=True)
        self.add_parameter(
            "centroids", torch.rand((size, 2)), batched=True, trainable=True
        )
        self.add_parameter(
            "sigma",
            (1 + torch.randn((size,)) * 0.2) * area_factor + 1e-3,
            batched=True,
            trainable=True,
        )
        self.add_parameter(
            "freq",
            1.0 + torch.rand((size,)) * 6.0,
            batched=True,
            trainable=True,
        )
        self.add_parameter(
            "color_1", torch.rand((size, 3)), batched=True, trainable=True
        )
        self.add_parameter(
            "color_2", torch.rand((size, 3)), batched=True, trainable=True
        )
        self.add_parameter("alphas", torch.rand((size,)), batched=True, trainable=True)

    @cached_property
    def scales(self) -> Tuple[Float[Tensor, "N"], Float[Tensor, "N"]]:
        return (self.sigma, self.sigma)

    @cached_property
    def areas(self) -> Float[Tensor, "N"]:
        return (self.sigma * 2.5) ** 2 * math.pi

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
        return dists - unit_patches[:, None] < 2.5 * self.sigma[None, :]

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

    @torch.no_grad()
    def split_params(self, mask: TensorIndex1D):
        """Compute new parameters for splitting primitives at given indices.

        Duplicates primitives at mask positions, halving ``sigma`` and
        offsetting centroids symmetrically along the ray phase offset
        direction. Other parameters (including ``freq`` and ``thetas``) are
        copied unchanged to the new primitives.

        Args:
            mask: Boolean mask or integer indices selecting primitives to split.

        Returns:
            Dict mapping parameter names to new tensors.
        """
        sigma_mask = self.sigma[mask]
        dirs = self.orientations[mask]
        disp = 0.25 * sigma_mask[:, None] * dirs

        centroids_copy = self.centroids
        centroids_copy[mask] -= disp
        new_centroids = torch.cat([centroids_copy, self.centroids[mask] + disp], dim=0)

        sigma_split = sigma_mask / (2**0.5)
        sigma_new = self.sigma.clone()
        sigma_new[mask] = sigma_split
        new_sigma = torch.cat([sigma_new, sigma_split], dim=0)

        new_thetas = torch.cat([self.thetas, self.thetas[mask]], dim=0)
        new_freq = torch.cat([self.freq, self.freq[mask]], dim=0)
        new_color = torch.cat([self.color, self.color[mask]], dim=0)
        new_alphas = torch.cat([self.alphas, self.alphas[mask]], dim=0)

        return {
            "thetas": new_thetas,
            "centroids": new_centroids,
            "sigma": new_sigma,
            "freq": new_freq,
            "color": new_color,
            "alphas": new_alphas,
        }
