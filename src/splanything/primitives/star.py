from __future__ import annotations
from typing import Tuple, Dict

import math
import torch
from jaxtyping import Bool, Float, Integer
from torch import Tensor

from .base import Primitive, cached_property, ParamDef


class StarPrimitive(Primitive):
    def __init__(
        self,
        size: int = 1,
        n_axes: int = 5,
        **kwargs,
    ):
        self._n_axes = n_axes
        super().__init__(size=size, **kwargs)

    @property
    def default_params(self) -> Dict[str, ParamDef]:
        return dict(
            centroids=ParamDef(True, True, (2,), 0.5),
            thetas=ParamDef(True, True, None),
            range=ParamDef(True, True, None, scalable=True),
            axis_weight=ParamDef(True, True, None),
            color=ParamDef(True, True, (3,)),
            alphas=ParamDef(True, True, None),
        )

    @cached_property
    def areas(self) -> Float[Tensor, "N"]:
        return math.pi * self.range**2

    @cached_property
    def scales(self) -> Tuple[Float[Tensor, "N"], Float[Tensor, "N"]]:
        return (self.range, self.range)

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
        return dists - unit_patches[:, None] < self.range[None, :]

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
        rng = self.range
        alpha = self.alphas
        aw = self.axis_weight
        n_axes = self._n_axes

        deltas = co[:, None, :] - centroids[None, :, :]
        dists = deltas.norm(dim=-1)
        radial = torch.exp(-(dists**2) / (2 * rng[None, :] ** 2 + 1e-8))

        angles = torch.atan2(deltas[..., 1], deltas[..., 0])
        rel_angles = angles - self.thetas[None, :]
        angular = 0.5 + 0.5 * torch.cos(n_axes * rel_angles)
        angular = angular**2
        modulated = aw[None, :] * angular + (1 - aw[None, :])

        return radial * modulated * alpha[None, :]
