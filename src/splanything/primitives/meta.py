"""MetaPrimitive: per-splat affine transforms plus color/alpha modulation over a child primitive."""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

import torch
from jaxtyping import Bool, Float, Integer
from torch import Tensor

from ..rendering.sample_output import SampleOutput
from .base import Primitive, cached_property, nomask, ParamDef, SampleProcessor
from .splitters.base import Splitter

if TYPE_CHECKING:
    from ..training.regularizers.base import Regularizer
    from ..training.refinement.base import SplitRule, FilterRule
    from .initializers.base import Initializer
    from .splitters.base import Splitter

_logger = logging.getLogger(__name__)


class MetaSplitter(Splitter):
    """Splitter placing meta splat children along their longest axis.

    Halves the shorter scale and offsets the split centroids along the
    longest world-space axis direction; all other parameters fall back to
    the base splitter.
    """

    def split_vals(
        self, name: str, primitive: Primitive, split_param: Float[Tensor, "N_split ..."]
    ) -> Tuple[Float[Tensor, "N_split ..."], Float[Tensor, "N_split ..."]]:
        """Split values for a parameter, axis-aware for meta parameters.

        Args:
            name: Parameter name.
            primitive: Primitive being split.
            split_param: Parameter rows selected for splitting (N_split, ...).

        Returns:
            out: Two parameter tensors (N_split, ...) placed at the split positions.
        """
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
            )  # (N_split, 2, 2)
            Ns = split_param.shape[0]
            longest_idx = longest.long()
            dirs_world = axis_dirs[
                torch.arange(Ns, device=axis_dirs.device), longest_idx
            ]
            dirs_world = dirs_world / dirs_world.norm(dim=-1, keepdim=True).clamp_min(
                1e-12
            )
            all_scales = torch.stack([p.scales_1, p.scales_2], dim=-1)
            split_lengths = all_scales[
                torch.arange(Ns, device=all_scales.device), longest_idx
            ]
            offset = 0.25 * split_lengths[:, None] * dirs_world
            return split_param + offset, split_param - offset


class MetaPrimitive(Primitive):
    """Per-splat transforms and color/alpha modulation over a child primitive.

    Each meta splat owns a local unit-square frame (rotation ``thetas``,
    scales ``scales_1``/``scales_2``). Sampling transforms coordinates into
    every meta splat's local frame where the child primitive is evaluated;
    child RGB is rotated/scaled by ``color_thetas``/``color_scales`` and
    child weights are modulated by ``alphas``.

    Attributes:
        primitive (Primitive): Child primitive whose samples are transformed.
        primitive_trainable (bool): Whether the child's parameters are optimized.

    Construction:
        MetaPrimitive(primitive, size=..., ...) -> MetaPrimitive:
            Wrap an existing child primitive; typically the child is frozen
            (``primitive_trainable=False``) and only the meta transforms train.

    Notes:
        - SampleOutput rows are laid out per meta splat: (Nc, N * Np, ...) with
          N meta splats and Np child splats.
        - ``forward`` returns zeros if len(self) == 0.
    """

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
        modify_scale: bool = True,
        modify_rotation: bool = True,
        modify_color: bool = True,
        modify_alphas: bool = True,
    ):
        """
        Args:
            primitive: Child primitive to transform.
            size: Number of meta splats.
            initializers: Per-parameter Initializer overrides.
            splitters: Per-parameter Splitter overrides.
            param_defs: ParamDef overrides merged over ``default_params``.
            filter_rules: Filter rules attached at construction.
            split_rules: Split rules attached at construction.
            sample_processors: Sample processors attached at construction.
            regularizers: Name -> (regularizer, weight) attached at construction.
            primitive_trainable: Optimize the child's parameters as well.
            modify_scale: If False, meta scales are frozen (identity).
            modify_rotation: If False, meta rotations are frozen (identity).
            modify_color: Train ``color_thetas``/``color_scales``.
            modify_alphas: Train ``alphas`` modulation of child weights.
        """
        self._modify_scale = modify_scale
        self._modify_rotation = modify_rotation
        self._modify_color = modify_color
        self._modify_alphas = modify_alphas
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
        """Meta parameter declarations; scale/color/alpha trainability follows the ``modify_*`` flags."""
        ms = self._modify_scale
        mc = self._modify_color
        return dict(
            centroids=ParamDef(True, True, (2,), 0.5),
            thetas=ParamDef(True, True, None),
            scales_1=ParamDef(True, ms, None, scalable=True),
            scales_2=ParamDef(True, ms, None, scalable=True),
            color_thetas=ParamDef(True, mc, None),
            color_scales=ParamDef(True, mc, None),
            alphas=ParamDef(True, True, None),
        )

    @property
    def default_splitters(self) -> Dict[str, Splitter]:
        """Meta splitter applied to every parameter."""
        return MetaSplitter()

    @cached_property
    def transforms(self) -> Float[Tensor, "N 2 2"]:
        """Per-splat affine transform matrices (N, 2, 2): rotation x scale."""
        c = torch.cos(self.thetas)
        s = torch.sin(self.thetas)
        if not self._modify_rotation:
            c = torch.ones_like(c)
            s = torch.zeros_like(s)
        sx = self.scales_1 if self._modify_scale else torch.ones_like(self.scales_1)
        sy = self.scales_2 if self._modify_scale else torch.ones_like(self.scales_2)
        return torch.stack(
            [
                torch.stack([sx * c, -sy * s], dim=-1),
                torch.stack([sx * s, sy * c], dim=-1),
            ],
            dim=-2,
        )

    @cached_property
    def transforms_components(self) -> Tuple[Float[Tensor, "N"]]:
        """The four transform matrix entries per splat as separate (N,) tensors."""
        t = self.transforms
        return (t[:, 0, 0], t[:, 0, 1], t[:, 1, 0], t[:, 1, 1])

    @cached_property
    def transforms_determinants(self) -> Float[Tensor, "N"]:
        """Determinant of each transform matrix (N,)."""
        a, b, c, d = self.transforms_components
        return a * d - b * c

    @cached_property
    def transforms_determinants_inverse(self) -> Float[Tensor, "N"]:
        """Reciprocal of each transform determinant (N,)."""
        return 1 / self.transforms_determinants

    @cached_property
    def transforms_inverse(self) -> Float[Tensor, "N 2 2"]:
        """Inverse transform matrices (N, 2, 2), mapping world coordinates to each splat's local frame."""
        a, b, c, d = self.transforms_components
        r1 = torch.stack([d, -b], dim=1)
        r2 = torch.stack([-c, a], dim=1)
        return self.transforms_determinants_inverse[:, None, None] * torch.stack(
            [r1, r2], dim=1
        )

    @cached_property
    def areas(self) -> Float[Tensor, "N"]:
        """Approximate splat areas (N,): ``scales_1 * scales_2``."""
        return self.scales_1 * self.scales_2

    @cached_property
    def scales(self) -> Tuple[Float[Tensor, "N"], Float[Tensor, "N"]]:
        """Pair ``(scales_1, scales_2)`` of per-splat scales."""
        return (self.scales_1, self.scales_2)

    @torch.no_grad()
    def _raw_patch_mask(
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
        """Sample child RGB in local frames, with color rotation/scale applied.

        Args:
            co: Local-frame coordinates (Nc, 2).
            meta_idx: Meta splat index per coordinate row (Nc,). Required
                when ``modify_color`` is set.

        Returns:
            out: RGB values (Nc, Np, 3).
        """
        rgb = self.primitive.sample_rgb(co, **kwargs)  # (M, Npi, 3)
        if not self._modify_color or meta_idx is None:
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
        """Sample child weights in local frames, modulated by ``alphas``.

        Args:
            co: Local-frame coordinates (Nc, 2).
            meta_idx: Meta splat index per coordinate row (Nc,). Required
                when ``modify_alphas`` is set.

        Returns:
            out: Weights (Nc, Np).
        """
        weights = self.primitive.sample_weights(co, **kwargs)  # (M, Npi)
        if not self._modify_alphas or meta_idx is None:
            return weights
        return weights * self.alphas[meta_idx][:, None]  # (M, Npi)

    def forward(self, co: Float[Tensor, "Nc 2"]) -> SampleOutput:
        """Sample the child primitive under every meta splat's transform.

        Coordinates are transformed into each meta splat's local frame via
        ``transforms_inverse`` and the child is sampled there.

        Args:
            co: Coordinates to sample at (Nc, 2).

        Returns:
            out: SampleOutput with rgb (Nc, N * Np, 3) and weights
            (Nc, N * Np), laid out per meta splat.

        Notes:
            - Returns zeros if len(self) == 0.
            - Coordinates outside a meta splat's local frame contribute
              zero (weight 0) and skip child sampling.
        """
        if len(self) == 0:
            return SampleOutput(
                rgb=torch.zeros((co.shape[0], 3), device=self.device, dtype=co.dtype),
                weights=torch.zeros(
                    (co.shape[0], 1), device=self.device, dtype=co.dtype
                ),
                co=co,
            )
        co_t = self._co_transform(co)
        with self.cache_properties():
            Nc = co.shape[0]
            N = len(self)
            Np = len(self.primitive)
            coords = (
                torch.einsum(
                    "nij,cnj->cni",
                    self.transforms_inverse,
                    co_t[:, None, :] - self.centroids[None],
                )
                + 0.5
            )
            flat_coords = coords.reshape(Nc * N, 2)  # (Nc*N, 2)
            inside = ((flat_coords >= 0) & (flat_coords <= 1)).all(dim=-1)  # (Nc*N,)
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
        """Evaluate regularizers on self and (if trainable) the child.

        Returns:
            out: Weighted scalar terms keyed ``<name>(Parent)`` and
            ``<name>(Child)``.
        """
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
        """Enable/disable gradients; the child is additionally gated by ``primitive_trainable``."""
        super().requires_grad_(mode)
        self.primitive.requires_grad_(self.primitive_trainable and mode)
        return self

    def train(self, mode: bool = True) -> MetaPrimitive:
        """Set train/eval mode; the child follows only when ``primitive_trainable``."""
        super().train(mode)
        self.primitive.train(mode and self.primitive_trainable)
        return self

    @nomask
    def param_groups(self) -> List[Dict[str, nn.Parameter]]:
        """Optimizer param groups; child groups (if trainable) get ``^^``-prefixed names.

        Returns:
            out: Own groups plus child groups; the ``^^`` prefix lets
            OptimizerWrapper treat child params as a sub-primitive.
        """
        groups = super().param_groups()
        if self.primitive_trainable:
            pg = self.primitive.param_groups()
            for g in pg:
                g["name"] = f"^^{g['name']}"
            groups.extend(pg)
        return groups
