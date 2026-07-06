from typing import Tuple

import torch
from jaxtyping import Bool, Float, Integer
from torch import Tensor

from ..utils.pytorch import TensorIndex1D
from .base import Primitive, cached_property


class CubicFanPrimitive(Primitive):
    """Cubic gradient primitive for image reconstruction.

    Represents an image as a collection of cubic gradients with position,
    orientation, color, and falloff parameters. Each primitive contributes
    a smooth color transition that can be composited to reconstruct an image.

    Attributes:
        thetas: Rotation angles for each primitive (N,).
        centroids: Center positions (N, 2).
        range_1: Primary falloff range (N,).
        range_2: Secondary falloff range (N,).
        color_1: Primary color (N, 3).
        color_2: Secondary color (N, 3).
        alphas: Opacity values (N,).

    Construction:
        CubicFanPrimitive(size: int):
            Create primitive with specified number of elements.
    """

    _ref_axis = [-1.0, 0.0]

    def __init__(self, size: int = 1):
        """Initialize primitive parameters.

        Initializes thetas, centroids, ranges, colors, and alphas as
        learnable parameters with sensible starting values.

        Args:
            size: Number of gradient primitives to create.
            **kwargs: Additional arguments (unused).

        Notes:
            - Areas scale with 1/sqrt(size) for balanced initial coverage.
            - Ranges are initialized as squared random values for positive magnitudes.
            - Colors are initialized near 0.5 (gray) with small variance.
            - Alphas use squared random for bias toward higher opacity.
        """
        super().__init__()
        area_factor = 2 / size**0.5
        self.add_parameter("thetas", torch.rand((size,)), batched=True, trainable=True)
        self.add_parameter(
            "centroids", torch.rand((size, 2)), batched=True, trainable=True
        )
        self.add_parameter(
            "range_1",
            (1 + torch.randn((size,)) * 0.1) * area_factor + 1e-3,
            batched=True,
            trainable=True,
        )
        self.add_parameter(
            "range_2",
            (1 + torch.randn((size,)) * 0.1) * area_factor + 1e-3,
            batched=True,
            trainable=True,
        )
        self.add_parameter(
            "color_1", torch.rand((size, 3)), batched=True, trainable=True
        )
        self.add_parameter(
            "color_2", torch.rand((size, 3)), batched=True, trainable=True
        )
        self.add_parameter(
            "alphas", 1 - torch.rand((size,)) ** 2, batched=True, trainable=True
        )

    @cached_property
    def ref_axis(self) -> Float[Tensor, "N 2"]:
        return (
            torch.tensor(
                CubicFanPrimitive._ref_axis, device=self.device, dtype=self.dtype
            )
            .unsqueeze(0)
            .expand(len(self), 2)
        )

    @cached_property
    def axes(self) -> Tuple[Float[Tensor, "N 2"], Float[Tensor, "N 2"]]:
        """Compute gradient axes from rotation matrices.

        Returns:
            Tuple of (ax_1, ax_2) where each is (N, 2) representing the
            two perpendicular axes of each gradient. ax_2 is ax_1 rotated
            90 degrees counterclockwise.
        """
        ref = self.ref_axis
        ax_1 = (self.R @ ref.unsqueeze(-1)).squeeze(-1)
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
        return self.range_1 * self.range_2 * 2

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
        range_max = torch.maximum(self.range_1, self.range_2)
        unit_patches = patch_sizes / torch.minimum(H, W)
        dists = (centers[:, None, :] - self.centroids[None, :, :]).norm(dim=2)
        return dists - unit_patches[:, None] < range_max[None, :]

    def sample_rgb(
        self,
        co: Float[Tensor, "Nc 2"],
        **kwargs,
    ) -> Float[Tensor, "Nc Np 3"]:
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

    @torch.no_grad()
    def split(self, mask: TensorIndex1D):
        """Split selected primitives into two with offset centroids.

        Duplicates primitives at mask positions, splitting each into two
        with half-sized ranges and offset centroids along both axes.

        Args:
            mask: Boolean mask or integer indices selecting primitives to split.

        Notes:
            - Ranges are divided by sqrt(2).
            - Alphas are divided by sqrt(2) for split copies.
            - New centroids are offset by 0.5 * (ax_1 * range_1 + ax_2 * range_2).
        """
        r_1, r_2 = self.range_1, self.range_2
        r_mask = r_1 > r_2
        ax_1, ax_2 = self.axes
        ax_1 = ax_1[mask] * self.range_1[mask, None]
        ax_2 = ax_2[mask] * self.range_2[mask, None]
        disp = 0.25 * torch.where(r_mask[mask, None].repeat(1, 2), ax_1, ax_2)
        self.centroids[mask] -= disp
        new_centroids = torch.cat([self.centroids, self.centroids[mask] + disp], dim=0)
        new_thetas = torch.cat([self.thetas, self.thetas[mask]], dim=0)
        r1half = self.range_1 / 2
        r2half = self.range_2 / 2
        new_range_1 = torch.cat(
            [
                torch.where(r_mask & mask, r1half, self.range_1),
                r1half[mask],
            ],
            dim=0,
        )
        new_range_2 = torch.cat(
            [
                torch.where(~r_mask & mask, r2half, self.range_2),
                r2half[mask],
            ],
            dim=0,
        )
        new_color_1 = torch.cat([self.color_1, self.color_1[mask]], dim=0)
        new_color_2 = torch.cat([self.color_2, self.color_2[mask]], dim=0)
        new_alphas = torch.cat([self.alphas, self.alphas[mask]], dim=0)
        self.update_parameters(
            {
                "thetas": new_thetas,
                "centroids": new_centroids,
                "range_1": new_range_1,
                "range_2": new_range_2,
                "color_1": new_color_1,
                "color_2": new_color_2,
                "alphas": new_alphas,
            }
        )
