import torch

from typing import Tuple
from jaxtyping import Float, Bool, Integer
from torch import Tensor

from splanything.utils.pytorch import TensorIndex1D
from .base import Primitive
from splanything.rasterizers import SampleOutput


class Gaussian(Primitive):
    """2D anisotropic Gaussian primitive for image reconstruction."""

    _ref_axis = [-1.0, 0.0]

    def __init__(self, size: int = 1):
        super().__init__()
        area_factor = 0.5 / size**0.5
        self.add_parameter("thetas", torch.rand((size,)), batched=True, trainable=True)
        self.add_parameter(
            "centroids", torch.rand((size, 2)), batched=True, trainable=True
        )
        self.add_parameter(
            "sigma_1",
            (1 + torch.randn((size,)) * 0.2) * area_factor + 1e-3,
            batched=True,
            trainable=True,
        )
        self.add_parameter(
            "sigma_2",
            (1 + torch.randn((size,)) * 0.2) * area_factor + 1e-3,
            batched=True,
            trainable=True,
        )
        self.add_parameter("color", torch.rand((size, 3)), batched=True, trainable=True)
        self.add_parameter("alphas", torch.rand((size,)), batched=True, trainable=True)

    @property
    def scales(self) -> Tuple[Float[Tensor, "N"], Float[Tensor, "N"]]:
        return (self.sigma_1, self.sigma_2)

    @property
    def ref_axis(self) -> Float[Tensor, "N 2"]:
        """Reference axis for rotations.

        Returns:
            Reference axis (N, 2)
        """
        N = len(self)
        return torch.tensor(
            self.__class__._ref_axis, device=self.thetas.device, dtype=self.thetas.dtype
        )

    @property
    def R(self) -> Float[Tensor, "N 2 2"]:
        """Rotation matrices for all primitives.

        Returns:
            Rotation matrices (N, 2, 2).
        """
        thetas = self.thetas
        thetapipi = (2 * torch.pi * thetas).unsqueeze(1)
        cos = torch.cos(thetapipi)
        sin = torch.sin(thetapipi)
        out = torch.stack([cos, -sin, sin, cos], dim=1).reshape(-1, 2, 2)
        return out

    @property
    def axes(self) -> Tuple[Float[Tensor, "N 2"], Float[Tensor, "N 2"]]:
        """Compute gradient axes from rotation matrices.

        Returns:
            Tuple of (ax_1, ax_2) where each is (N, 2) representing the
            two perpendicular axes of each gradient. ax_2 is ax_1 rotated
            90 degrees counterclockwise.
        """
        ref = self.ref_axis
        ax_1 = self.R @ ref
        ax_2 = torch.stack([ax_1[:, 1], -ax_1[:, 0]], dim=1)
        return ax_1, ax_2

    @property
    def areas(self) -> Float[Tensor, "N"]:
        return self.sigma_1 * self.sigma_2 * 3.14159

    @torch.no_grad()
    def patch_mask(
        self,
        centers: Float[Tensor, "P 2"],
        patch_sizes: Integer[Tensor, "P"],
        H: Integer[Tensor, "P"],
        W: Integer[Tensor, "P"],
    ) -> Bool[Tensor, "P N"]:
        sig = torch.maximum(self.sigma_1, self.sigma_2)
        unit_patches = patch_sizes / torch.minimum(H, W)
        dists = (centers[:, None, :] - self.centroids[None, :, :]).norm(dim=2)
        return dists - unit_patches[:, None] < 2.5 * sig[None, :]

    def sample_rgb(
        self,
        co: Float[Tensor, "Nc 2"],
        **kwargs,
    ) -> Float[Tensor, "Nc Np 3"]:
        return self.color[None, :, :].expand(co.shape[0], -1, -1)

    def sample_weights(
        self,
        co: Float[Tensor, "Nc 2"],
        **kwargs,
    ) -> Float[Tensor, "Nc N"]:
        centroids = self.centroids
        sigma_1 = self.sigma_1
        sigma_2 = self.sigma_2
        alpha = self.alphas
        ax_1, ax_2 = self.axes
        deltas = co[:, None, :] - centroids[None, :, :]
        dot1 = (deltas * ax_1).sum(dim=-1).abs()
        dot2 = (deltas * ax_2).sum(dim=-1).abs()
        weights = (
            torch.exp(-(dot1**2) / (2 * sigma_1**2 + 1e-8))
            * torch.exp(-(dot2**2) / (2 * sigma_2**2 + 1e-8))
            * alpha[None, :]
        )
        return weights

    @torch.no_grad()
    def split(self, mask: TensorIndex1D):
        """Split selected primitives along their main axis.

        Duplicates primitives at mask positions, splitting each into two
        along the principal axis (larger sigma). Only the principal axis
        variance is halved; the orthogonal axis is unchanged.

        Args:
            mask: Boolean mask or integer indices selecting primitives to split.

        Notes:
            - Only sigma_1 (principal axis) is divided by sqrt(2).
            - Sigma_2 (orthogonal axis) is unchanged.
            - Alphas are divided by sqrt(2) for split copies.
            - Centroids are offset by 0.5 * sigma_1 along the principal axis only.
        """
        ax_1, ax_2 = self.axes
        ax_1 = ax_1[mask] * self.sigma_1[mask, None]
        ax_2 = ax_2[mask] * self.sigma_2[mask, None]
        sigma_1_mask = self.sigma_1[mask]
        sigma_2_mask = self.sigma_2[mask]
        longest_is_1 = sigma_1_mask >= sigma_2_mask
        disp = torch.where(longest_is_1.unsqueeze(-1), ax_1, ax_2) * 0.5
        sq2 = 2**0.5

        # Compute new values for all parameters
        new_thetas = torch.cat([self.thetas, self.thetas[mask]], dim=0)

        # Note: centroids are modified in-place first, then concatenated
        centroids_copy = self.centroids
        centroids_copy[mask] -= disp
        new_centroids = torch.cat([centroids_copy, self.centroids[mask] + disp], dim=0)

        # Note: sigma_1 and sigma_2 are modified in-place first, then concatenated
        sigma_1_split = sigma_1_mask / sq2
        sigma_2_split = sigma_2_mask / sq2
        sigma_1_new = self.sigma_1
        sigma_2_new = self.sigma_2
        sigma_1_new[mask] = torch.where(
            longest_is_1.squeeze(-1), sigma_1_split, self.sigma_1[mask]
        )
        sigma_2_new[mask] = torch.where(
            longest_is_1.squeeze(-1), self.sigma_2[mask], sigma_2_split
        )
        new_sigma_1 = torch.cat([sigma_1_new, sigma_1_split], dim=0)
        new_sigma_2 = torch.cat([sigma_2_new, sigma_2_split], dim=0)

        new_color = torch.cat([self.color, self.color[mask]], dim=0)
        new_alphas = torch.cat([self.alphas, self.alphas[mask]], dim=0)

        self.update_parameters(
            {
                "thetas": new_thetas,
                "centroids": new_centroids,
                "sigma_1": new_sigma_1,
                "sigma_2": new_sigma_2,
                "color": new_color,
                "alphas": new_alphas,
            }
        )
