"""Anisotropic 2D Gaussian primitive."""

from typing import Tuple, Dict

import torch
from jaxtyping import Bool, Float, Integer
from torch import Tensor

from .base import Primitive, cached_property, ParamDef
from .splitters.base import Splitter


class GaussianSplitter(Splitter):
    """Splitter splitting Gaussians along their dominant axis.

    Halves the dominant sigma, keeps the other, and displaces child
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
        if name not in ("sigma_1", "sigma_2", "centroids"):
            return super().split_vals(name, primitive, split_param)
        p = primitive
        s_mask = p.sigma_1 > p.sigma_2
        if name == "centroids":
            ax_1, ax_2 = p.axes
            ax_1 = ax_1 * p.sigma_1[:, None]
            ax_2 = ax_2 * p.sigma_2[:, None]
            disp = torch.where(s_mask, ax_1, ax_2) / 4
            return split_param + disp, split_param - disp
        if name == "sigma_1":
            new_param = torch.where(s_mask, split_param / 2, split_param)
            return new_param, new_param
        if name == "sigma_2":
            new_param = torch.where(s_mask, split_param, split_param / 2)
            return new_param, new_param


class GaussianPrimitive(Primitive):
    """2D anisotropic Gaussian primitive for image reconstruction.

    Attributes:
        thetas: Rotation of each Gaussian in [0, 1] turns (N,).
        centroids: Center positions (N, 2).
        sigma_1: Standard deviation along the first axis (N,).
        sigma_2: Standard deviation along the second axis (N,).
        color: Per-primitive color (N, 3).
        alphas: Peak opacity (N,).
        ref_axis: Fixed reference axis the rotation acts on (2,).
    """

    _sigma_cutoff = 2.5

    @property
    def default_params(self) -> Dict[str, ParamDef]:
        """Parameter definitions for this primitive."""
        return dict(
            thetas=ParamDef(True, True, None),
            centroids=ParamDef(True, True, (2,), 0.5),
            sigma_1=ParamDef(True, True, None, scalable=True),
            sigma_2=ParamDef(True, True, None, scalable=True),
            color=ParamDef(True, True, (3,)),
            alphas=ParamDef(True, True, None),
            ref_axis=ParamDef(False, False, (2,)),
        )

    @property
    def default_splitters(self) -> Dict[str, Splitter]:
        """Default splitter: ``GaussianSplitter`` for all parameters."""
        return GaussianSplitter()

    @cached_property
    def scales(self) -> Tuple[Float[Tensor, "N"], Float[Tensor, "N"]]:
        """Scale parameters used by refinement/splitting.

        Returns:
            Tuple of (sigma_1, sigma_2), each (N,).
        """
        return (self.sigma_1, self.sigma_2)

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
        out = torch.stack([cos, -sin, sin, cos], dim=1).reshape(-1, 2, 2)
        return out

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
    def areas(self) -> Float[Tensor, "N"]:
        """Approximate area of each primitive.

        Returns:
            Tensor of shape (N,) with area values
            sigma_1 * sigma_2 * _sigma_cutoff**2 * pi.
        """
        return self.sigma_1 * self.sigma_2 * self._sigma_cutoff**2 * torch.pi

    @torch.no_grad()
    def _raw_patch_mask(
        self,
        centers: Float[Tensor, "P 2"],
        patch_sizes: Integer[Tensor, "P"],
        H: Integer[Tensor, "P"],
        W: Integer[Tensor, "P"],
    ) -> Bool[Tensor, "P N"]:
        sig = torch.maximum(self.sigma_1, self.sigma_2)
        unit_patches = patch_sizes / torch.minimum(H, W)
        dists = (centers[:, None, :] - self.centroids[None, :, :]).norm(dim=2)
        return dists - unit_patches[:, None] < self._sigma_cutoff * sig[None, :]

    def sample_rgb(
        self,
        co: Float[Tensor, "Nc 2"],
        **kwargs,
    ) -> Float[Tensor, "Nc Np 3"]:
        """Sample per-primitive colors at coordinates.

        Args:
            co: Coordinates to sample at (Nc, 2).

        Returns:
            RGB tensor (Nc, Np, 3): each primitive's constant color.
        """
        return self.color[None, :, :].expand(co.shape[0], -1, -1)

    def sample_weights(
        self,
        co: Float[Tensor, "Nc 2"],
        **kwargs,
    ) -> Float[Tensor, "Nc N"]:
        """Sample per-primitive weights at coordinates.

        Anisotropic Gaussian falloff along the rotated axes, scaled by
        alpha.

        Args:
            co: Coordinates to sample at (Nc, 2).

        Returns:
            Weights tensor (Nc, Np).
        """
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
