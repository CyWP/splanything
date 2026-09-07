"""Cubic fan primitive: oriented two-color gradient fans."""

from __future__ import annotations
from typing import Tuple, TYPE_CHECKING

import torch
from jaxtyping import Bool, Float, Integer
from torch import Tensor

from .base import Primitive, ParamDef, cached_property
from .splitters.base import Splitter


class CubicFanSplitter(Splitter):
    """Splitter splitting cubic fans along their dominant axis.

    Halves the dominant range, keeps the other range, and displaces child
    centroids by a quarter of the dominant axis extent; other parameters
    fall back to ``Splitter.split_vals``.
    """

    def split_vals(
        self, name: str, primitive: Primitive, split_param: Float[Tensor, "N_split ..."]
    ) -> Tuple[Float[Tensor, "N_split ..."], Float[Tensor, "N_split ..."]]:
        """Compute child parameter values for instances being split.

        Args:
            name: Parameter name being split.
            primitive: Primitive being split, with the split instances
                masked in.
            split_param: Parameter values of the instances being split.

        Returns:
            Tuple of two tensors (N_split, ...): values for the retained
            rows and the appended split rows.
        """
        if name not in ("range_1", "range_2", "centroids"):
            return super().split_vals(name, primitive, split_param)
        p = primitive
        r_mask = p.range_1 > p.range_2
        if name == "range_1":
            new_param = torch.where(r_mask, split_param / 2, split_param)
            return new_param, new_param
        if name == "range_2":
            new_param = torch.where(r_mask, split_param, split_param / 2)
            return new_param, new_param
        if name == "centroids":
            ax_1, ax_2 = p.axes
            ax_1 *= p.range_1[:, None]
            ax_2 *= p.range_2[:, None]
            disp = torch.where(r_mask[:, None].repeat(1, 2), ax_1, ax_2) / 4
            return split_param + disp, split_param - disp


class CubicFanPrimitive(Primitive):
    """Oriented two-color gradient fan.

    Each instance is a gradient rotated by ``thetas`` around
    ``centroids``: a coordinate takes ``color_1`` or ``color_2`` depending
    on which of the two perpendicular axes it aligns with, with weight
    falling off linearly along the dominant axis over its range, squared.

    Attributes:
        thetas: Rotation of each gradient in [0, 1] turns (N,).
        centroids: Center positions (N, 2).
        range_1: Extent along the first axis (N,).
        range_2: Extent along the second axis (N,).
        color_1: Color of the first axis half (N, 3).
        color_2: Color of the second axis half (N, 3).
        alphas: Opacity (N,).
        ref_axis: Fixed reference axis the rotation acts on (2,).
    """

    @property
    def default_params(self) -> Dict[str, ParamDef]:
        """Parameter definitions for this primitive."""
        return dict(
            thetas=ParamDef(True, True, None),
            centroids=ParamDef(True, True, (2,), 0.5),
            range_1=ParamDef(True, True, None, scalable=True),
            range_2=ParamDef(True, True, None, scalable=True),
            color_1=ParamDef(True, True, (3,)),
            color_2=ParamDef(True, True, (3,)),
            alphas=ParamDef(True, True, None),
            ref_axis=ParamDef(False, False, (2,)),
        )

    @property
    def default_splitters(self) -> Dict[str, Splitter]:
        """Default splitter: ``CubicFanSplitter`` for all parameters."""
        return CubicFanSplitter()

    @cached_property
    def axes(self) -> Tuple[Float[Tensor, "N 2"], Float[Tensor, "N 2"]]:
        """Compute gradient axes from rotation matrices.

        Returns:
            Tuple of (ax_1, ax_2) where each is (N, 2) representing the
            two perpendicular axes of each gradient. ax_2 is ax_1 rotated
            90 degrees counterclockwise.
        """
        ax_1 = self.R @ self.ref_axis
        ax_2 = torch.stack([ax_1[:, 1], -ax_1[:, 0]], dim=1)
        return ax_1, ax_2

    @cached_property
    def R(self) -> Float[Tensor, "N 2 2"]:
        """Rotation matrices for all primitives.

        Returns:
            Rotation matrices (N, 2, 2).
        """
        thetas = self.thetas
        thetapipi = (2 * torch.pi * thetas).unsqueeze(1)
        cos = torch.cos(thetapipi)
        sin = torch.sin(thetapipi)
        out = torch.stack([cos, -sin, sin, cos], dim=1).reshape(thetas.shape[0], 2, 2)
        return out

    @cached_property
    def areas(self) -> Float[Tensor, "N"]:
        """Approximate area of each primitive for normalization.

        Returns:
            Tensor of shape (N,) with area values (range_1 * range_2 * 2).
        """
        return torch.maximum(self.range_1, self.range_2) ** 2 * 4

    @cached_property
    def scales(self) -> Tuple[Float[Tensor, "N"], Float[Tensor, "N"]]:
        """Scale parameters used by refinement/splitting.

        Returns:
            Tuple of (range_1, range_2), each (N,).
        """
        return (self.range_1, self.range_2)

    @torch.no_grad()
    def _raw_patch_mask(
        self,
        centers: Float[Tensor, "P 2"],
        patch_sizes: Integer[Tensor, "P"],
        H: Integer[Tensor, "P"],
        W: Integer[Tensor, "P"],
    ) -> Bool[Tensor, "P N"]:
        range_max = torch.maximum(self.range_1, self.range_2)
        unit_patches = patch_sizes / torch.minimum(H, W)
        dists = (centers[:, None, :] - self.centroids[None, :, :]).norm(dim=2)
        return dists - unit_patches[:, None] < range_max[None, :]

    def sample_rgb(
        self,
        co: Float[Tensor, "Nc 2"],
        **kwargs,
    ) -> Float[Tensor, "Nc Np 3"]:
        """Sample per-primitive colors at coordinates.

        Returns ``color_1`` or ``color_2`` per primitive depending on
        which gradient axis the coordinate aligns with.

        Args:
            co: Coordinates to sample at (Nc, 2).

        Returns:
            RGB tensor (Nc, Np, 3).
        """
        ax_1, ax_2 = self.axes
        centroids = self.centroids
        color_1 = self.color_1
        color_2 = self.color_2
        deltas = co[:, None, :] - centroids[None, :, :]  # [Nc, N, 2]
        dot1 = (deltas * ax_1).sum(dim=-1).abs()  # [Nc, N]
        dot2 = (deltas * ax_2).sum(dim=-1).abs()  # [Nc, N]
        axmask = dot1 > dot2  # [Nc, N]
        return torch.where(axmask[:, :, None], color_1[None, :, :], color_2[None, :, :])

    def sample_weights(
        self,
        co: Float[Tensor, "Nc 2"],
        **kwargs,
    ) -> Float[Tensor, "Nc N"]:
        """Sample per-primitive weights at coordinates.

        Linear falloff from the centroid along the dominant axis over its
        range, times an angular-alignment factor, all squared.

        Args:
            co: Coordinates to sample at (Nc, 2).

        Returns:
            Weights tensor (Nc, Np).
        """
        ax_1, ax_2 = self.axes
        centroids = self.centroids
        range_1 = self.range_1
        range_2 = self.range_2
        deltas = co[:, None, :] - centroids[None, :, :]  # [Nc, N, 2]
        dists = deltas.norm(dim=-1)  # [Nc, N]
        dot1 = (deltas * ax_1).sum(dim=-1).abs()  # [Nc, N]
        dot2 = (deltas * ax_2).sum(dim=-1).abs()  # [Nc, N]
        axmask = dot1 > dot2  # [Nc, N]
        dots = torch.where(axmask, dot1, dot2)  # [Nc, N]
        ranges = torch.where(axmask, range_1[None, :], range_2[None, :])  # [Nc, N]
        weights = (
            (1 - dots / (ranges.abs() + 1e-6)).clamp(0, 1)
            * ((dots / dists.clamp(min=1e-6)) ** 2).clamp(min=1e-6)
        ) ** 2  # [Nc, N]
        return weights
