from __future__ import annotations
from typing import Tuple, Dict

import math
import torch
from jaxtyping import Bool, Float, Integer
from torch import Tensor

from .base import Primitive, cached_property, ParamDef


class AnisotropicFanPrimitive(Primitive):
    def __init__(
        self,
        size: int = 1,
        n_axes: int = 3,
        **kwargs,
    ):
        self._n_axes = n_axes
        super().__init__(size=size, **kwargs)

    @property
    def default_params(self) -> Dict[str, ParamDef]:
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
        rng_1, rng_2 = self.range_1, self.range_2
        rng_mask = rng_1 < rng_2
        rng_min = torch.where(rng_mask, rng_1, rng_2)
        rng_max = torch.where(rng_mask, rng_2, rng_1)
        return rng_min, rng_max

    @cached_property
    def axes(self) -> Tuple[Float[Tensor, "N 2"], Float[Tensor, "N 2"]]:
        c = torch.cos(self.thetas)
        s = torch.sin(self.thetas)
        ax_1 = torch.stack([c, s], dim=-1)
        ax_2 = torch.stack([-s, c], dim=-1)
        return ax_1, ax_2

    @cached_property
    def axis_angles(self) -> Float[Tensor, "N A"]:
        n = self._n_axes
        return (
            self.thetas[:, None]
            + torch.linspace(
                0, 2 * math.pi, n + 1, device=self.device, dtype=self.thetas.dtype
            )[:-1][None, :]
        )

    @cached_property
    def areas(self) -> Float[Tensor, "N"]:
        return math.pi * (self.range_1 * self.range_2).abs()

    @cached_property
    def scales(self) -> Tuple[Float[Tensor, "N"], Float[Tensor, "N"]]:
        return (self.range_1, self.range_2)

    @torch.no_grad()
    def patch_mask(
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
        return self.color[None, :, :].expand(co.shape[0], -1, -1)

    def sample_weights(
        self,
        co: Float[Tensor, "Nc 2"],
        **kwargs,
    ) -> Float[Tensor, "Nc N"]:
        centroids = self.centroids
        alpha = self.alphas
        rng_min, rng_max = self.ranges
        deltas = co[:, None, :] - centroids[None, :, :]
        dist = deltas.norm(dim=-1)

        angles = (
            torch.atan2(deltas[..., 1], deltas[..., 0]) - self.thetas[None, :]
        ) % (torch.pi / self._n_axes)
        range_weights = (angles / (torch.pi / self._n_axes)) ** 4
        angular_ranges = range_weights * rng_min + (1 - range_weights) * rng_max
        weights = 1 - (dist / angular_ranges).clamp(0, 1)
        return weights
