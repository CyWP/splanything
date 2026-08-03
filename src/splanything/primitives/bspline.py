from __future__ import annotations
from typing import Tuple, Dict

import math
import torch
from jaxtyping import Bool, Float, Integer
from torch import Tensor

from .base import Primitive, cached_property, ParamDef


class BSplinePrimitive(Primitive):
    def __init__(
        self,
        size: int = 1,
        n_vertices: int = 4,
        degree: int = 2,
        n_samples: int = 64,
        **kwargs,
    ):
        self._n_vertices = n_vertices
        self._degree = degree
        self._n_samples = n_samples
        super().__init__(size=size, **kwargs)

    @property
    def default_params(self) -> Dict[str, ParamDef]:
        V = self._n_vertices
        return dict(
            centroids=ParamDef(True, True, (V, 2)),
            width=ParamDef(True, True, None, scalable=True),
            color_1=ParamDef(True, True, (3,)),
            color_2=ParamDef(True, True, (3,)),
            alphas=ParamDef(True, True, None),
        )

    @cached_property
    def knots(self) -> Float[Tensor, "K"]:
        V = self._n_vertices
        p = self._degree
        device = self.device
        n_knots = V + p + 1
        k = torch.zeros(n_knots, device=device)
        n_internal = V - p - 1
        if n_internal > 0:
            mid = torch.linspace(0, 1, n_internal + 2, device=device)[1:-1]
            k[p + 1 : -(p + 1)] = mid
        k[-(p + 1) :] = 1.0
        return k

    @cached_property
    def curve_points(self) -> Float[Tensor, "N S 2"]:
        ctrl = self.centroids
        N, V, _ = ctrl.shape
        p = self._degree
        n_pts = self._n_samples
        device = ctrl.device
        knots = self.knots

        t = torch.linspace(0, 1, n_pts, device=device)
        n_basis = V + p
        basis = torch.zeros(n_pts, n_basis, device=device)
        for i in range(n_basis):
            lo = (t >= knots[i]).float()
            hi = (t < knots[i + 1]).float()
            tie = (t == 1.0).float() * float(i == n_basis - 1)
            basis[:, i] = lo * hi + tie

        for p_iter in range(1, p + 1):
            nb = n_basis - p_iter
            new_basis = torch.zeros(n_pts, nb, device=device)
            for i in range(nb):
                ln = t - knots[i]
                ld = knots[i + p_iter] - knots[i]
                left = torch.where(ld.abs() > 1e-10, ln / ld, torch.zeros_like(t))
                rn = knots[i + p_iter + 1] - t
                rd = knots[i + p_iter + 1] - knots[i + 1]
                right = torch.where(rd.abs() > 1e-10, rn / rd, torch.zeros_like(t))
                new_basis[:, i] = left * basis[:, i] + right * basis[:, i + 1]
            basis = new_basis

        return torch.einsum("nvc,sv->nsc", ctrl, basis)

    @cached_property
    def n_segments(self) -> int:
        return self._n_samples - 1

    @cached_property
    def segment_a(self) -> Float[Tensor, "N S 2"]:
        return self.curve_points[:, :-1, :]

    @cached_property
    def segment_b(self) -> Float[Tensor, "N S 2"]:
        return self.curve_points[:, 1:, :]

    @cached_property
    def segment_vec(self) -> Float[Tensor, "N S 2"]:
        return self.segment_b - self.segment_a

    @cached_property
    def segment_len_sq(self) -> Float[Tensor, "N S"]:
        return (self.segment_vec**2).sum(dim=-1).clamp(min=1e-8)

    @cached_property
    def areas(self) -> Float[Tensor, "N"]:
        total_len = self.segment_len_sq.sqrt().sum(dim=-1)
        return 2 * total_len * self.width + math.pi * self.width**2

    @cached_property
    def scales(self) -> Tuple[Float[Tensor, "N"], Float[Tensor, "N"]]:
        return (self.width, self.width)

    @torch.no_grad()
    def _raw_patch_mask(
        self,
        centers: Float[Tensor, "P 2"],
        patch_sizes: Integer[Tensor, "P"],
        H: Integer[Tensor, "P"],
        W: Integer[Tensor, "P"],
    ) -> Bool[Tensor, "P N"]:
        width = self.width
        total_len = self.segment_len_sq.sqrt().sum(dim=-1)
        radius = total_len / self.n_segments + width
        unit_patches = patch_sizes / torch.minimum(H, W)
        c = self.centroids.mean(dim=1)
        dists = (centers[:, None, :] - c[None, :, :]).norm(dim=2)
        return dists - unit_patches[:, None] < radius[None, :]

    def _segment_distances(
        self, co: Float[Tensor, "Nc 2"]
    ) -> Tuple[
        Float[Tensor, "Nc N"],
        Float[Tensor, "Nc N"],
        Float[Tensor, "Nc N"],
    ]:
        a = self.segment_a
        v = self.segment_vec
        v_len_sq = self.segment_len_sq

        d = co[:, None, None, :] - a[None, :, :, :]
        t = (d * v[None, :, :, :]).sum(dim=-1) / v_len_sq[None, :, :]
        t_clamped = t.clamp(0, 1)

        closest = a[None, :, :, :] + t_clamped[:, :, :, None] * v[None, :, :, :]
        dist_sq = ((co[:, None, None, :] - closest) ** 2).sum(dim=-1)
        best_dist_sq, best_seg = dist_sq.min(dim=-1)

        cross_val = d[..., 0] * v[None, :, :, 1] - d[..., 1] * v[None, :, :, 0]
        best_cross = cross_val.gather(-1, best_seg[:, :, None]).squeeze(-1)

        return best_dist_sq, best_cross, best_seg

    def sample_rgb(
        self,
        co: Float[Tensor, "Nc 2"],
        **kwargs,
    ) -> Float[Tensor, "Nc N 3"]:
        _, best_cross, _ = self._segment_distances(co)
        side = (best_cross > 0).float()[:, :, None]
        return side * self.color_1[None, :, :] + (1 - side) * self.color_2[None, :, :]

    def sample_weights(
        self,
        co: Float[Tensor, "Nc 2"],
        **kwargs,
    ) -> Float[Tensor, "Nc N"]:
        best_dist_sq, _, _ = self._segment_distances(co)
        w = torch.exp(-best_dist_sq / (2 * self.width[None, :] ** 2 + 1e-8))
        return w * self.alphas[None, :]
