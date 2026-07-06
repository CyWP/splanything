from __future__ import annotations

import logging
from typing import Tuple

import torch
from jaxtyping import Bool, Float, Integer
from torch import Tensor

from ..rendering import SampleOutput
from ..rendering.rasterizers import Rasterizer
from .base import Primitive, cached_property

_logger = logging.getLogger(__name__)


class MetaPrimitive(Primitive):
    def __init__(
        self,
        primitive: Primitive,
        size: int = 1,
        init_scale: float = 0.1,
        color: bool = True,
        alpha: bool = True,
        primitive_trainable: bool = False,
    ):
        super().__init__()
        self.primitive = primitive
        self.primitive.requires_grad_(primitive_trainable)
        self.primitive_trainable = primitive_trainable
        self.add_parameter(
            "centroids", torch.rand((size, 2)), batched=True, trainable=True
        )
        self.add_parameter(
            "transforms",
            torch.randn((size, 2, 2)) * init_scale,
            batched=True,
            trainable=True,
        )
        if color:
            self.add_parameter(
                "color_thetas",
                torch.rand((size,)) * 2 * torch.pi,
                batched=True,
                trainable=True,
            )
            self.add_parameter(
                "color_scales",
                1.0 + torch.randn((size,)) * 0.1,
                batched=True,
                trainable=True,
            )
        if alpha:
            self.add_parameter("alphas", 1.0 + torch.randn((size,)) * 0.1)

    @cached_property
    def transforms_components(self) -> Tuple[Float[Tensor, "N"]]:
        t = self.transforms
        return (t[:, 0, 0], t[:, 0, 1], t[:, 1, 0], t[:, 1, 1])

    @cached_property
    def transforms_determinants(self) -> Float[Tensor, "N"]:
        a, b, c, d = self.transforms_components
        return a * d - b * c

    @cached_property
    def transforms_determinants_inverse(self) -> Float[Tensor, "N"]:
        return 1 / self.transforms_determinants

    @cached_property
    def transforms_inverse(self) -> Float[Tensor, "N 2 2"]:
        a, b, c, d = self.transforms_components
        r1 = torch.stack([d, -b], dim=1)
        r2 = torch.stack([-c, a], dim=1)
        return self.transforms_determinants_inverse * torch.stack([r1, r2], dim=1)

    @torch.no_grad()
    def patch_mask(
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
            Bool tensor (P, N) indicating which primitives are valid for a given patch.
        """
        half_w = patch_sizes * W * 0.5
        half_h = patch_sizes * H * 0.5

        offsets = torch.stack(
            (
                torch.stack((-half_w, -half_h), dim=1),
                torch.stack((half_w, -half_h), dim=1),
                torch.stack((half_w, half_h), dim=1),
                torch.stack((-half_w, half_h), dim=1),
            ),
            dim=1,
        )  # (P,4,2)
        patch = centers[:, None, :] + offsets  # (P,4,2)
        rel = patch[:, None, :, :] - self.centroids[None, :, None, :]  # (P,N,4,2)
        local = (
            torch.einsum(
                "nij,pnkj->pnki",
                self.transforms_inverse,
                rel,
            )
            + 0.5
        )
        min_xy = local.min(dim=2).values  # (P,N,2)
        max_xy = local.max(dim=2).values  # (P,N,2)
        overlap = (
            (max_xy[..., 0] >= 0)
            & (min_xy[..., 0] <= 1)
            & (max_xy[..., 1] >= 0)
            & (min_xy[..., 1] <= 1)
        )
        edges = torch.roll(local, -1, dims=2) - local  # (P,N,4,2)
        normals = torch.stack(
            (-edges[..., 1], edges[..., 0]),
            dim=-1,
        )  # (P,N,4,2)
        normals = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        square = local.new_tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )
        for i in range(4):
            axis = normals[..., i, :]  # (P,N,2)
            # project patch
            proj_patch = (local * axis[..., None, :]).sum(-1)  # (P,N,4)
            pmin = proj_patch.min(dim=2).values
            pmax = proj_patch.max(dim=2).values
            # project square
            proj_sq = (square[None, None, :, :] * axis[..., None, :]).sum(-1)
            smin = proj_sq.min(dim=2).values
            smax = proj_sq.max(dim=2).values
            overlap &= (pmax >= smin) & (smax >= pmin)
        return overlap

    @torch.no_grad()
    def split(self, idx: Bool[Tensor, "N"]) -> Primitive:
        """Split instances at given indices"""
        updates = dict()
        A = self.transforms  # (N,2,2)
        axis_norms = (A**2).sum(dim=1)  # (N,2)
        longest = axis_norms.argmax(dim=1)  # (N,)
        eye = torch.eye(2, device=A.device, dtype=A.dtype)
        dirs_local = eye[longest]  # (N,2)
        dirs_world = torch.einsum(
            "nij,nj->ni",
            A,
            dirs_local,
        )
        dirs_world = dirs_world / dirs_world.norm(dim=1, keepdim=True).clamp_min(1e-12)
        for name, param in self.batched_parameters():
            new_param = torch.cat([param, param[idx]])
            split_mask = torch.cat(
                [torch.zeros_like(idx, dtype=torch.bool), idx], dim=0
            )
            if name == "transforms":
                # halve along dominant axis
                axis = longest[idx].repeat_interleave(1)
                scale = torch.ones_like(new_param)
                scale[split_mask, axis, axis] = 0.5
                new_param = new_param @ scale
            elif name == "centroids":
                offsets = 0.25 * dirs_world[idx]
                offsets = torch.stack([offsets, -offsets], dim=1).reshape(-1, 2)
                new_param[split_mask] = new_param[split_mask] + offsets
            updates[name] = new_param
        self.update_parameters(updates)
        return self

    def sample_rgb(
        self,
        co: Float[Tensor, "Nc 2"],
        **kwargs,
    ) -> Float[Tensor, "Nc Np 3"]:
        rgb = self.primitive.sample_rgb(co, **kwargs)  # (Nc, Np, 3)
        if not hasattr(self, "color_thetas"):
            return rgb
        Np = rgb.shape[1]
        device = rgb.device
        dtype = rgb.dtype
        theta = self.color_thetas.view(1, Np, 1)  # (1,Np,1)
        scale = self.color_scales.view(1, Np, 1)  # (1,Np,1)
        rgb_centered = rgb - 0.5
        c = torch.cos(theta)
        s = torch.sin(theta)
        # Rodrigues rotation around axis (1,1,1)/sqrt(3)
        axis = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype)
        axis = axis / axis.norm()
        x, y, z = axis
        # cross-product matrix
        K = torch.tensor(
            [
                [0, -z, y],
                [z, 0, -x],
                [-y, x, 0],
            ],
            device=device,
            dtype=dtype,
        )
        I = torch.eye(3, device=device, dtype=dtype)
        K2 = K @ K
        R = I[None, None] + s[..., None] * K + (1 - c)[..., None] * K2  # (1,Np,3,3)
        rgb_rot = torch.einsum("pnij,cnj->cni", R, rgb_centered)
        rgb_out = rgb_rot * scale + 0.5
        return rgb_out

    def sample_weights(
        self,
        co: Float[Tensor, "Nc 2"],
        **kwargs,
    ) -> Float[Tensor, "Nc Np"]:
        weights = self.primitive.sample_weights(co, **kwargs)
        if not hasattr(self, "alphas"):
            return weights
        return self.alphas[:, None] * weights

    def forward(self, co: Float[Tensor, "Nc 2"], rasterizer: Rasterizer):
        if len(self) == 0:
            return torch.zeros(co.shape[0], 4, device=co.device)
        with self.cache_properties():
            Nc = co.shape[0]
            N = len(self)
            Np = len(self.primitive)
            coords = (
                torch.einsum(
                    "nij,cj->cni",
                    self.transforms,
                    co - 0.5,
                )
                + self.centroids[None, :, :]
            )
            flat_coords = coords.reshape(Nc * N, 2)
            inside = ((flat_coords >= 0) & (flat_coords <= 1)).all(dim=-1)
            inside_coords = flat_coords[inside]
            rgb_flat = torch.zeros(
                (flat_coords.shape[0], 3), device=co.device, dtype=co.dtype
            )
            w_flat = torch.zeros(
                (flat_coords.shape[0],), device=co.device, dtype=co.dtype
            )
            rgb_flat[inside] = self.sample_rgb(inside_coords)
            w_flat[inside] = self.sample_weights(inside_coords)
            rgb = rgb_flat.view(Nc, N * Np, 3)
            weights = w_flat.view(Nc, N * Np)
            return rasterizer(SampleOutput(rgb=rgb, weights=weights, co=co))

    def requires_grad_(self, mode: bool = True) -> MetaPrimitive:
        super().requires_grad_(mode)
        self.primitive.requires_grad_(self.primitive_trainable and mode)
        return self

    def train(self, mode: bool = True) -> MetaPrimitive:
        super().train(mode)
        self.primitive.train(mode and self.primitive_trainable)
        return self
