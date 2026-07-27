from __future__ import annotations

import logging
from typing import Tuple, Optional, TYPE_CHECKING

import torch
from jaxtyping import Bool, Float, Integer
from torch import Tensor

from ..rendering.sample_output import SampleOutput
from .base import Primitive, cached_property, nomask, ParamDef
from ..training.splitters.base import Splitter

if TYPE_CHECKING:
    from ..training.regularizers.base import Regularizer
    from ..training.refinement.base import SplitRule, FilterRule
    from ..training.initializers import Initializer
    from ..training.splitters import Splitter

_logger = logging.getLogger(__name__)


class MetaSplitter(Splitter):
    def split_vals(
        self, name: str, primitive: Primitive, split_param: Float[Tensor, "N_split ..."]
    ) -> Tuple[Float[Tensor, "N_split ..."], Float[Tensor, "N_split ..."]]:
        if name not in ("centroids", "scales_1", "scales_2"):
            return super().split_vals(name, primitive, split_param)
        p = primitive
        longest = p.scales_1 > p.scales_2
        if name == "scales_1":
            new_param = split_param.clone()
            new_param[longest] *= 0.5
            return new_param, new_param
        if name == "scales_2":
            new_param = split_param.clone()
            new_param[~longest] *= 0.5
            return new_param, new_param
        if name == "centroids":
            # World-space axis directions
            c = torch.cos(p.thetas)
            s = torch.sin(p.thetas)
            axis_dirs = torch.stack(
                [torch.stack([c, s], dim=-1), torch.stack([-s, c], dim=-1)], dim=1
            )  # (N, 2, 2)
            dirs_world = axis_dirs[
                torch.arange(len(self), device=axis_dirs.device),
                longest,
            ]  # (N, 2)
            dirs_world = dirs_world / dirs_world.norm(dim=-1, keepdim=True).clamp_min(
                1e-12
            )
            split_lengths = scales[:, longest]
            offset = 0.25 * split_lengths[:, None] * dirs_world
            return split_param + offset, split_param - offset


class MetaPrimitive(Primitive):
    def __init__(
        self,
        primitive: Primitive,
        size: int = 1,
        initializers: Optional[Dict[str, Initializer] | Initializer] = None,
        splitters: Optional[Dict[str, Splitter] | Splitter] = None,
        param_defs: Optional[Dict[str, ParamDef]] = None,
        filter_rules: Optional[List[FilterRule]] = None,
        split_rules: Optional[List[SplitRule]] = None,
        sample_processors: Optional[List[SampleProcessor]] = None,
        regularizers: Optional[Dict[str, Tuple[Regularizer, float]]] = None,
        primitive_trainable: bool = False,
    ):
        super().__init__(
            size=size,
            initializers=initializers,
            splitters=splitters,
            param_defs=param_defs,
            filter_rules=filter_rules,
            split_rules=split_rules,
            sample_processors=sample_processors,
            regularizers=regularizers,
        )
        self.primitive = primitive
        self.primitive.requires_grad_(primitive_trainable)
        self.primitive_trainable = primitive_trainable
        self.add_parameter(
            "rgb_axis",
            torch.tensor([1 / 3**0.5] * 3),
            batched=False,
            trainable=False,
        )

    @property
    def default_params(self) -> Dict[str, ParamDef]:
        return dict(
            centroids=ParamDef(True, True, (2,), 0.5),
            thetas=ParamDef(True, True, None),
            scales_1=ParamDef(True, True, None),
            scales_2=ParamDef(True, True, None),
            color_thetas=ParamDef(True, True, None),
            color_scales=ParamDef(True, True, None),
            alphas=ParamDef(True, True, None),
        )

    @property
    def default_splitters(self) -> Dict[str, Splitter]:
        return MetaSplitter()

    @cached_property
    def transforms(self) -> Float[Tensor, "N 2 2"]:
        c = torch.cos(self.thetas)
        s = torch.sin(self.thetas)
        return torch.stack(
            [
                torch.stack([self.scales_1 * c, -self.scales_2 * s], dim=-1),
                torch.stack([self.scales_1 * s, self.scales_2 * c], dim=-1),
            ],
            dim=-2,
        )

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
        return self.transforms_determinants_inverse[:, None, None] * torch.stack(
            [r1, r2], dim=1
        )

    @cached_property
    def areas(self) -> Float[Tensor, "N"]:
        return self.scales_1 * self.scales_2

    @cached_property
    def scales(self) -> Tuple[Float[Tensor, "N"], Float[Tensor, "N"]]:
        return (self.scales_1, self.scales_2)

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

    def sample_rgb(
        self,
        co: Float[Tensor, "Nc 2"],
        meta_idx: Optional[Integer[Tensor, "Nc"]] = None,
        **kwargs,
    ) -> Float[Tensor, "Nc Np 3"]:
        rgb = self.primitive.sample_rgb(co, **kwargs)  # (M, Npi, 3)
        if meta_idx is None or not hasattr(self, "color_thetas"):
            return rgb
        theta = self.color_thetas[meta_idx]  # (M,)
        scale = self.color_scales[meta_idx]  # (M,)
        rgb_centered = rgb - 0.5  # (M, Npi, 3)
        c = torch.cos(theta)[:, None, None]  # (M, 1, 1)
        s = torch.sin(theta)[:, None, None]  # (M, 1, 1)
        axis = self.rgb_axis.to(rgb.device, dtype=rgb.dtype)  # (3,)
        x, y, z = axis.unbind(-1)
        zero = torch.zeros((), device=rgb.device, dtype=rgb.dtype)
        K = torch.stack(
            [
                torch.stack([zero, -z, y]),
                torch.stack([z, zero, -x]),
                torch.stack([-y, x, zero]),
            ]
        )  # (3, 3)
        I = torch.eye(3, device=rgb.device, dtype=rgb.dtype)
        K2 = K @ K
        R = I[None] + s * K[None] + (1 - c) * K2[None]  # (M, 3, 3)
        rgb_rot = torch.einsum("mij,mkj->mki", R, rgb_centered)  # (M, Npi, 3)
        rgb_out = rgb_rot * scale[:, None, None] + 0.5
        return rgb_out

    def sample_weights(
        self,
        co: Float[Tensor, "Nc 2"],
        meta_idx: Optional[Integer[Tensor, "Nc"]] = None,
        **kwargs,
    ) -> Float[Tensor, "Nc Np"]:
        weights = self.primitive.sample_weights(co, **kwargs)  # (M, Npi)
        if meta_idx is None or not hasattr(self, "alphas"):
            return weights
        return weights * self.alphas[meta_idx][:, None]  # (M, Npi)

    def forward(self, co: Float[Tensor, "Nc 2"]) -> SampleOutput:
        if len(self) == 0:
            return SampleOutput(
                rgb=torch.zeros((co.shape[0], 3), device=self.device, dtype=co.dtype),
                weights=torch.zeros(
                    (co.shape[0], 1), device=self.device, dtype=co.dtype
                ),
                co=co,
            )
        with self.cache_properties():
            Nc = co.shape[0]
            N = len(self)
            Np = len(self.primitive)
            coords = (
                torch.einsum(
                    "nij,cnj->cni",
                    self.transforms_inverse,
                    co[:, None, :] - self.centroids[None],
                )
                + 0.5
            )
            flat_coords = coords.reshape(Nc * N, 2)  # (Nc*N, 2)
            inside = ((flat_coords >= 0) & (flat_coords <= 1)).all(dim=-1)  # (Nc*N,)
            inside = torch.ones_like(inside)
            meta_idx = torch.arange(N, device=co.device).repeat(Nc)  # (Nc*N,)
            rgb_flat = torch.zeros((Nc * N, Np, 3), device=co.device, dtype=co.dtype)
            w_flat = torch.zeros((Nc * N, Np), device=co.device, dtype=co.dtype)
            if inside.any():
                rgb_flat[inside] = self.sample_rgb(
                    flat_coords[inside], meta_idx=meta_idx[inside]
                )
                w_flat[inside] = self.sample_weights(
                    flat_coords[inside], meta_idx=meta_idx[inside]
                )
            rgb = rgb_flat.view(Nc, N * Np, 3)
            weights = w_flat.view(Nc, N * Np)
            sample = SampleOutput(rgb=rgb, weights=weights, co=co)
            for proc in self._sample_processors:
                sample = proc(sample, self)
            return sample

    def compute_regularization(self) -> Dict[str, Float[Tensor, ""]]:
        regs = super().compute_regularization()
        if self.primitive_trainable:
            regs = {
                **{f"{name}(Parent)": r for name, r in regs.items()},
                **{
                    f"{name}(Child)": r
                    for name, r in self.primitive.compute_regularization().items()
                },
            }
        return regs

    def requires_grad_(self, mode: bool = True) -> MetaPrimitive:
        super().requires_grad_(mode)
        self.primitive.requires_grad_(self.primitive_trainable and mode)
        return self

    def train(self, mode: bool = True) -> MetaPrimitive:
        super().train(mode)
        self.primitive.train(mode and self.primitive_trainable)
        return self

    @nomask
    def param_groups(self) -> List[Dict[str, nn.Parameter]]:
        groups = super().param_groups()
        if self.primitive_trainable:
            pg = self.primitive.param_groups()
            for g in pg:
                g["name"] = f"^^{g['name']}"
            groups.extend(pg)
        return groups
