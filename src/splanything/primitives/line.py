from __future__ import annotations
from typing import Tuple, Dict

import math
import torch
from jaxtyping import Bool, Float, Integer
from torch import Tensor

from .base import Primitive, cached_property, ParamDef


class LinePrimitive(Primitive):
    @property
    def default_params(self) -> Dict[str, ParamDef]:
        return dict(
            centroids=ParamDef(True, True, (2,), 0.5),
            thetas=ParamDef(True, True, None),
            length=ParamDef(True, True, None),
            width=ParamDef(True, True, None),
            color_1=ParamDef(True, True, (3,)),
            color_2=ParamDef(True, True, (3,)),
            alphas=ParamDef(True, True, None),
        )

    @cached_property
    def direction(self) -> Float[Tensor, "N 2"]:
        ang = 2 * math.pi * self.thetas
        return torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)

    @cached_property
    def normal(self) -> Float[Tensor, "N 2"]:
        d = self.direction
        return torch.stack([-d[:, 1], d[:, 0]], dim=-1)

    @cached_property
    def endpoint_a(self) -> Float[Tensor, "N 2"]:
        return self.centroids - self.length[:, None] * self.direction

    @cached_property
    def endpoint_b(self) -> Float[Tensor, "N 2"]:
        return self.centroids + self.length[:, None] * self.direction

    @cached_property
    def areas(self) -> Float[Tensor, "N"]:
        return 4 * self.length * self.width + math.pi * self.width**2

    @cached_property
    def scales(self) -> Tuple[Float[Tensor, "N"], Float[Tensor, "N"]]:
        return (self.length, self.width)

    @torch.no_grad()
    def patch_mask(
        self,
        centers: Float[Tensor, "P 2"],
        patch_sizes: Integer[Tensor, "P"],
        H: Integer[Tensor, "P"],
        W: Integer[Tensor, "P"],
    ) -> Bool[Tensor, "P N"]:
        radius = self.length + self.width
        unit_patches = patch_sizes / torch.minimum(H, W)
        dists = (centers[:, None, :] - self.centroids[None, :, :]).norm(dim=2)
        return dists - unit_patches[:, None] < radius[None, :]

    def sample_rgb(
        self,
        co: Float[Tensor, "Nc 2"],
        **kwargs,
    ) -> Float[Tensor, "Nc N 3"]:
        a = self.endpoint_a
        deltas = co[:, None, :] - a[None, :, :]
        seg = (self.endpoint_b - a)[None, :, :]
        seg_len_sq = (seg**2).sum(dim=-1).clamp(min=1e-8)
        t = (deltas * seg).sum(dim=-1) / seg_len_sq
        t_clamped = t.clamp(0, 1)
        closest = a[None, :, :] + t_clamped[:, :, None] * seg
        diff = co[:, None, :] - closest
        normal_dist = (diff * self.normal[None, :, :]).sum(dim=-1)
        side = (normal_dist > 0).float()[:, :, None]
        return side * self.color_1[None, :, :] + (1 - side) * self.color_2[None, :, :]

    def sample_weights(
        self,
        co: Float[Tensor, "Nc 2"],
        **kwargs,
    ) -> Float[Tensor, "Nc N"]:
        a = self.endpoint_a
        deltas = co[:, None, :] - a[None, :, :]
        seg = (self.endpoint_b - a)[None, :, :]
        seg_len_sq = (seg**2).sum(dim=-1).clamp(min=1e-8)
        t = (deltas * seg).sum(dim=-1) / seg_len_sq
        t_clamped = t.clamp(0, 1)
        closest = a[None, :, :] + t_clamped[:, :, None] * seg
        dist_sq = ((co[:, None, :] - closest) ** 2).sum(dim=-1)
        w = torch.exp(-dist_sq / (2 * self.width[None, :] ** 2 + 1e-8))
        return w * self.alphas[None, :]
