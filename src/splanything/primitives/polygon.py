from __future__ import annotations
from typing import Tuple, Dict

import math
import torch
from jaxtyping import Bool, Float, Integer
from torch import Tensor

from .base import Primitive, cached_property, ParamDef
from .initializers.base import Initializer


class PolygonInitializer(Initializer):
    def init_param(
        self, name: str, param_shape: Tuple[int], batched: bool
    ) -> Float[Tensor, "N ..."]:
        if name == "centerpoint":
            return (torch.rand(param_shape) - 0.5) * 0.6
        return super().init_param(name, param_shape, batched)


class PolygonPrimitive(Primitive):
    def __init__(
        self,
        size: int = 1,
        n_sides: int = 5,
        n_colors: int | None = None,
        **kwargs,
    ):
        self._n_sides = n_sides
        self._n_colors = n_colors if n_colors is not None else n_sides
        super().__init__(size=size, **kwargs)
        assign = torch.arange(n_sides) % self._n_colors
        self.register_buffer("color_assignments", assign)

    @property
    def default_params(self) -> Dict[str, ParamDef]:
        nc = self._n_colors
        return dict(
            centroids=ParamDef(True, True, (2,), 0.5),
            thetas=ParamDef(True, True, None),
            range=ParamDef(True, True, None, scalable=True),
            centerpoint=ParamDef(True, True, (2,)),
            colors=ParamDef(True, True, (nc, 3)),
            alphas=ParamDef(True, True, None),
        )

    @property
    def default_initializers(self) -> Dict[str, Initializer] | Initializer:
        return PolygonInitializer()

    @cached_property
    def vertices(self) -> Float[Tensor, "N S 2"]:
        thetas = self.thetas
        rng = self.range
        centroids = self.centroids
        n = self._n_sides
        angles = (
            thetas[:, None]
            + torch.linspace(
                0, 2 * math.pi, n + 1, device=thetas.device, dtype=thetas.dtype
            )[:-1][None, :]
        )
        dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        return centroids[:, None, :] + rng[:, None, None] * dirs

    @cached_property
    def cp(self) -> Float[Tensor, "N 2"]:
        return self.centroids + self.centerpoint * self.range[:, None]

    @cached_property
    def areas(self) -> Float[Tensor, "N"]:
        return math.pi * self.range**2

    @cached_property
    def scales(self) -> Tuple[Float[Tensor, "N"], Float[Tensor, "N"]]:
        return (self.range, self.range)

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
        return dists - unit_patches[:, None] < self.range[None, :]

    def sample_rgb(
        self,
        co: Float[Tensor, "Nc 2"],
        **kwargs,
    ) -> Float[Tensor, "Nc N 3"]:
        centroids = self.centroids
        thetas = self.thetas
        colors = self.colors
        assign = self.color_assignments
        n = self._n_sides

        deltas = co[:, None, :] - centroids[None, :, :]
        angles = torch.atan2(deltas[..., 1], deltas[..., 0])
        rel_angles = angles - thetas[None, :]
        rel_angles = (rel_angles + math.pi) % (2 * math.pi) - math.pi
        sector = (
            ((rel_angles + math.pi) / (2 * math.pi) * n).floor().long().clamp(0, n - 1)
        )

        color_idx = assign[sector]
        N = len(self)
        out = colors[
            torch.arange(N, device=co.device)[None, :, None],
            color_idx[:, :, None],
            torch.arange(3, device=co.device)[None, None, :],
        ]
        return out

    def sample_weights(
        self,
        co: Float[Tensor, "Nc 2"],
        **kwargs,
    ) -> Float[Tensor, "Nc N"]:
        centroids = self.centroids
        rng = self.range
        alpha = self.alphas
        n = self._n_sides
        N = len(self)
        Nc = co.shape[0]
        device = co.device

        deltas = co[:, None, :] - centroids[None, :, :]
        dists = deltas.norm(dim=-1)

        angles = torch.atan2(deltas[..., 1], deltas[..., 0])
        rel_angles = angles - self.thetas[None, :]
        rel_angles = (rel_angles + math.pi) % (2 * math.pi) - math.pi
        sector = (
            ((rel_angles + math.pi) / (2 * math.pi) * n).floor().long().clamp(0, n - 1)
        )

        v = self.vertices
        batch_idx = torch.arange(N, device=device)[:, None].expand(-1, Nc)
        sector_t = sector.T
        vi = v[batch_idx, sector_t]
        vin = v[batch_idx, (sector_t + 1) % n]

        e = vin - vi
        delta = vi - self.cp[:, None, :]
        d = co[None, :, :] - self.cp[:, None, :]

        det = e[..., 0] * d[..., 1] - e[..., 1] * d[..., 0]
        safe_det = det.sign() * det.abs().clamp(min=1e-8)
        t = (e[..., 0] * delta[..., 1] - e[..., 1] * delta[..., 0]) / safe_det
        s = (d[..., 0] * delta[..., 1] - d[..., 1] * delta[..., 0]) / safe_det

        inside = (t > 1e-6) & (s >= 0) & (s <= 1)
        tri_weight = torch.where(inside, (1 - 1 / t).clamp(0, 1), torch.zeros_like(t))
        radial = torch.exp(-(dists.T**2) / (2 * rng[:, None] ** 2 + 1e-8))
        weight_val = torch.maximum(tri_weight, radial * 0.3)

        return (weight_val * alpha[:, None]).T
