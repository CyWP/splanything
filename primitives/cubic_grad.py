import torch

from typing import Optional, Tuple
from jaxtyping import Float, Bool
from torch import Tensor

from utils.lazy import lazy_tree
from utils.math import soft_clamp
from utils.pytorch import TensorIndex
from .generic import Primitive
from .protocols import HasAlphas, HasAreas, Splittable, HasScales


class CubicGrad(Primitive, HasAlphas, HasAreas, Splittable, HasScales):
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
        CubicGrad(size: int):
            Create primitive with specified number of elements.

    Notes:
        - Uses exponential falloff for smooth gradient transitions.
        - Colors interpolate based on position relative to gradient axis.
        - Uses @lazy_tree for caching computed properties.
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

    @property
    def ref_axis(self) -> Float[Tensor, "N 2"]:
        return (
            torch.tensor(CubicGrad._ref_axis, device=self.device, dtype=self.dtype)
            .unsqueeze(0)
            .expand(len(self), 2)
        )

    @property
    def axes(self) -> Tuple[Float[Tensor, "N 2"], Float[Tensor, "N 2"]]:
        """Compute gradient axes from rotation matrices (cached).

        Returns:
            Tuple of (ax_1, ax_2) where each is (N, 2) representing the
            two perpendicular axes of each gradient.
        """
        return CubicGrad._compute_axes(self.R, ref=self.ref_axis)

    @classmethod
    def _compute_axes(
        cls, R: Float[Tensor, "N 2 2"], ref: Optional[Float[Tensor, "N 2"]] = None
    ) -> Tuple[Float[Tensor, "N 2"], Float[Tensor, "N 2"]]:
        """Compute gradient axes from rotation matrices.

        Args:
            R: Rotation matrices (N, 2, 2).

        Returns:
            Tuple of (ax_1, ax_2) where ax_2 is ax_1 rotated 90 degrees counterclockwise.
        """
        if ref is None:
            ref = torch.tensor(cls._ref_axis, device=R.device, dtype=R.dtype)
        ax_1 = (R @ ref.unsqueeze(-1)).squeeze(-1)
        ax_2 = torch.stack([ax_1[:, 1], -ax_1[:, 0]], dim=1)
        return ax_1, ax_2

    @property
    def R(self) -> Float[Tensor, "N 2 2"]:
        """Rotation matrices for all primitives (cached via lazy_tree).

        Returns:
            Rotation matrices (N, 2, 2).
        """
        return CubicGrad._compute_R(self.thetas)

    @classmethod
    def _compute_R(cls, thetas: Float[Tensor, "N"]) -> Float[Tensor, "N 2 2"]:
        """Compute rotation matrix for primitive orientations.

        Args:
            thetas: Rotation angles (N,).

        Returns:
            Rotation matrices (N, 2, 2).
        """
        thetapipi = (2 * torch.pi * thetas).unsqueeze(1)
        cos = torch.cos(thetapipi)
        sin = torch.sin(thetapipi)
        out = torch.stack([cos, -sin, sin, cos], dim=1).reshape(thetas.shape[0], 2, 2)
        return out

    @property
    def areas(self) -> Float[Tensor, "N"]:
        """Approximate area of each primitive for normalization.

        Returns:
            Tensor of shape (N,) with area values (range_1 * range_2 * 2).
        """
        return self.range_1 * self.range_2 * 2

    @property
    def scales(self) -> Tuple[Float[Tensor, "N"], Float[Tensor, "N"]]:
        return (self.range_1, self.range_2)

    def _sample(self, co: Float[Tensor, "N 2"]) -> Float[Tensor, "Nm 4"]:
        """Sample primitive values at coordinates.

        Args:
            co: Coordinates to sample at (N, 2).
            mask: Optional mask for active primitives (N,).

        Returns:
            Sampled RGBA values (N, 4).

        Notes:
            Nm is the sum of the mask (number of splats that are actually used for computation).
        """
        ax_1, ax_2 = self.axes
        return CubicGrad._sample_F(
            co,
            self.centroids,
            self.range_1,
            self.range_2,
            self.color_1,
            self.color_2,
            self.alphas,
            ax_1,
            ax_2,
        )

    @classmethod
    def _sample_F(
        cls,
        co: Float[Tensor, "Nc 2"],
        centroids: Float[Tensor, "N 2"],
        range_1: Float[Tensor, "N"],
        range_2: Float[Tensor, "N"],
        color_1: Float[Tensor, "N 3"],
        color_2: Float[Tensor, "N 3"],
        alpha: Float[Tensor, "N"],
        ax_1: Float[Tensor, "N 2"],
        ax_2: Float[Tensor, "N 2"],
    ) -> Float[Tensor, "Nm 4"]:
        """Sample cubic gradients at coordinates.

        Args:
            co: Coordinates to sample at (Nc, 2).
            mask: Optional mask for active primitives (N,).
            centroids: Center positions (N, 2).
            range_1: Primary falloff range (N,).
            range_2: Secondary falloff range (N,).
            color_1: Primary color (N, 3).
            color_2: Secondary color (N, 3).
            alpha: Opacity values (N,).
            thetas: Rotation angles (N,).
            R: Rotation matrix derived from thetas. Overrides use of thetas if provided. (N, 2, 2)

        Returns:
            Sampled RGBA values (Nc, 4).

        Notes:
            Nm is the sum of the mask (number of splats that are actually used for computation).
        """
        Nc = co.shape[0]
        c_mask = centroids
        Nm = c_mask.shape[0]
        deltas = co[:, None, :] - c_mask[None, :, :]  # [Nc, Nm, 2]
        dists = deltas.norm(dim=-1)  # [Nc, Nm]
        dot1 = (deltas * ax_1).sum(dim=-1).abs()  # [Nc, Nm]
        dot2 = (deltas * ax_2).sum(dim=-1).abs()  # [Nc, Nm]
        axmask = dot1 > dot2  # [Nc, Nm]
        dots = torch.where(
            axmask,
            dot1,
            dot2,
        )  # [Nc, Nm]
        ranges = torch.where(
            axmask,
            range_1[None, :].expand(Nc, Nm),
            range_2[None, :].expand(Nc, Nm),
        )  # [Nc, Nm]
        weights = (
            soft_clamp(1 - dots / (ranges + 1e-6), 0, 1, 0.1)
            * ((dots / (dists)) ** 2).clamp(min=1e-6)
        ) ** 2  # [Nc, Nm]
        c1 = color_1[None, :, :].expand(Nc, Nm, 3)  # [Nc, Nm, 3]
        c2 = color_2[None, :, :].expand(Nc, Nm, 3)  # [Nc, Nm, 3]
        rgb = torch.where(axmask[:, :, None], c1, c2)
        rgb = (rgb * weights.unsqueeze(-1)).sum(dim=1)  # [Nc, 3]
        weight_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)  # [Nc, 1]
        rgb = (rgb / weight_sum).clamp(0, 1)  # [Nc, 3]
        a = soft_clamp(
            (weights * alpha[None, :]).sum(dim=1),
            min_val=0.0,
            max_val=1.0,
            softness=0.1,
        )  # [Nc, 1]
        out = torch.cat([rgb, a.unsqueeze(-1)], dim=-1)  # [Nc, 4]
        return out

    @torch.no_grad()
    def split(self, mask: TensorIndex):
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
        r_mask = (r_1 > r_2) & mask
        ax_1, ax_2 = self.axes
        ax_1 = ax_1[mask] * self.range_1[mask, None]
        ax_2 = ax_2[mask] * self.range_2[mask, None]
        disp = 0.25 * torch.where(r_mask[mask, None].repeat(1, 2), ax_1, ax_2)
        self.centroids[mask] -= disp
        new_centroids = torch.cat([self.centroids, self.centroids[mask] + disp], dim=0)
        new_thetas = torch.cat([self.thetas, self.thetas[mask]], dim=0)
        new_range_1 = torch.cat(
            [
                torch.where(r_mask, self.range_1 / 2, self.range_1),
                self.range_1[mask],
            ],
            dim=0,
        )
        new_range_2 = torch.cat(
            [
                torch.where(r_mask, self.range_2 / 2, self.range_2),
                self.range_2[mask],
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
