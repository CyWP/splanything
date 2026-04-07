import torch

from typing import Optional
from jaxtyping import Float, Bool
from torch import Tensor
from utils.lazy import lazy_tree
from .generic import Primitive


@lazy_tree
class CubicGrad(Primitive):
    """Cubic gradient primitive for image reconstruction.

    Represents an image as a collection of cubic gradients with position,
    orientation, color, and falloff parameters. Each primitive contributes
    a smooth color transition that can be composited to reconstruct an image.

    Attributes:
        thetas: Rotation angles for each primitive (N,).
        centroids: Center positions (N, 2).
        range1: Primary falloff range (N,).
        range2: Secondary falloff range (N,).
        color1: Primary color (N, 3).
        color2: Secondary color (N, 3).
        alpha: Opacity values (N,).

    Construction:
        CubicGrad(size: int):
            Create primitive with specified number of elements.

    Notes:
        - Uses exponential falloff for smooth gradient transitions.
        - Colors interpolate based on position relative to gradient axis.
        - Uses @lazy_tree for caching computed properties.
    """

    def __init__(self, size: int):
        """Initialize cubic gradient primitive.

        Args:
            size: Number of gradient primitives to create.
        """
        super().__init__()

    def __len__(self) -> int:
        """Number of gradient primitives."""
        return self.thetas.shape[0]

    @torch.no_grad()
    def patch_mask(
        self, center: Float[Tensor, "N 2"], patch_size: int
    ) -> Bool[Tensor, "N"]:
        """Compute valid patch mask (always returns all valid)."""
        return (center - self.means).norm(dim=1) < patch_size

    @property
    def R(self) -> Float[Tensor, "N 2 2"]:
        """Rotation matrices for all primitives (cached via lazy_tree).

        Returns:
            Rotation matrices (N, 2, 2).
        """
        return CubicGrad._compute_R(self.thetas, self.device, self.dtype)

    @staticmethod
    def _compute_R(
        thetas: Float[Tensor, "N"],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Float[Tensor, "N 2 2"]:
        """Compute rotation matrix for primitive orientations.

        Args:
            thetas: Rotation angles (N,).
            device: Target device.
            dtype: Target dtype.

        Returns:
            Rotation matrices (N, 2, 2).
        """
        cos = torch.cos(thetas)
        sin = torch.sin(thetas)
        return torch.stack([cos, -sin, sin, cos], dim=0).reshape(-1, 2, 2)

    @staticmethod
    def _sample_impl(
        co: Float[Tensor, "N 2"],
        mask: Optional[Bool[Tensor, "N"]],
        thetas: Float[Tensor, "N"],
        centroids: Float[Tensor, "N 2"],
        range1: Float[Tensor, "N"],
        range2: Float[Tensor, "N"],
        color1: Float[Tensor, "N 3"],
        color2: Float[Tensor, "N 3"],
        alpha: Float[Tensor, "N"],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Float[Tensor, "N 4"]:
        """Sample cubic gradients at coordinates.

        Args:
            co: Coordinates to sample at (N, 2).
            mask: Optional mask for active primitives (N,).
            thetas: Rotation angles (N,).
            centroids: Center positions (N, 2).
            range1: Primary falloff range (N,).
            range2: Secondary falloff range (N,).
            color1: Primary color (N, 3).
            color2: Secondary color (N, 3).
            alpha: Opacity values (N,).
            device: Target device.
            dtype: Target dtype.

        Returns:
            Sampled RGBA values (N, 4).
        """
        # Apply mask to thetas if provided
        thetas_masked = thetas if mask is None else thetas[mask]
        R = CubicGrad._compute_R(thetas_masked, device, dtype)

        ax1 = R @ torch.tensor([-1, 0], device=device, dtype=dtype).unsqueeze(0).expand(
            len(co), 2
        )
        ax2 = torch.stack([ax1[:, 1], -ax1[:, 0]], dim=0)
        deltas = centroids[mask] - co
        dot1 = (ax1 * deltas).sum(dim=1).abs()
        dot2 = (ax2 * deltas).sum(dim=1).abs()
        axmask = dot1 > dot2
        dots = dot2 / range2[mask]
        dots[axmask] = dot1[axmask] / range1[mask]
        alphas = torch.exp(-dots)
        colors = color2.clone()
        colors[axmask] = color1[axmask]
        return torch.cat([colors, alphas.unsqueeze(1)], dim=1)

    def sample(
        self, co: Float[Tensor, "N 2"], mask: Optional[Bool[Tensor, "N"]] = None
    ) -> Float[Tensor, "N 4"]:
        """Sample primitive values at coordinates.

        Args:
            co: Coordinates to sample at (N, 2).
            mask: Optional mask for active primitives (N,).

        Returns:
            Sampled RGBA values (N, 4).
        """
        return self._sample_impl(
            co,
            mask,
            self.thetas,
            self.centroids,
            self.range1,
            self.range2,
            self.color1,
            self.color2,
            self.alpha,
            self.device,
            self.dtype,
        )
