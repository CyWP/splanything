from typing import Tuple, Dict

import torch
from jaxtyping import Bool, Float, Integer
from torch import Tensor

from .base import Primitive, cached_property, ParamDef
from ..training.splitters import Splitter


class GaussianSplitter(Splitter):
    def split_vals(
        self, name: str, primitive: Primitive, split_param: Float[Tensor, "N_split ..."]
    ) -> Tuple[Float[Tensor, "N_split ..."], Float[Tensor, "N_split ..."]]:
        if name not in ("sigma_1", "sigma_2", "centroids"):
            return super().split_vals(name, primitive, split_param)
        s_mask = p.sigma_1 > p.sigma_2
        if name == "centroids":
            p = primitive
            ax_1, ax_2 = p.axes
            ax_1 *= p.sigma_1[:, None]
            ax_2 *= p.sigma_2[:, None]
            disp = torch.where(s_mask, ax_1, ax_2) / 4
            return split_param + disp, split_param - disp
        if name == "sigma_1":
            new_param = torch.where(s_mask, param_split / 2, param_split)
            return new_param, new_param
        if name == "sigma_2":
            new_param = torch.where(s_mask, param_split, param_split / 2)
            return new_param, new_param


class GaussianPrimitive(Primitive):
    """2D anisotropic Gaussian primitive for image reconstruction."""

    _sigma_cutoff = 2.5

    @property
    def default_params(self) -> Dict[str, ParamDef]:
        return dict(
            thetas=ParamDef(True, True, None),
            centroids=ParamDef(True, True, (2,), 0.5),
            sigma_1=ParamDef(True, True, None),
            sigma_2=ParamDef(True, True, None),
            color=ParamDef(True, True, (3,)),
            alphas=ParamDef(True, True, None),
            ref_axis=ParamDef(False, False, (2,)),
        )

    @property
    def default_splitters(self) -> Dict[str, Splitter]:
        return GaussianSplitter()

    @cached_property
    def scales(self) -> Tuple[Float[Tensor, "N"], Float[Tensor, "N"]]:
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
        ref = self.ref_axis[None, :]
        ax_1 = self.R @ ref
        ax_2 = torch.stack([ax_1[:, 1], -ax_1[:, 0]], dim=1)
        return ax_1, ax_2

    @cached_property
    def areas(self) -> Float[Tensor, "N"]:
        return self.sigma_1 * self.sigma_2 * self._sigma_cutoff**2 * torch.pi

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
        return dists - unit_patches[:, None] < self._sigma_cutoff * sig[None, :]

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
