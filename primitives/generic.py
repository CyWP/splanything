from __future__ import annotations
import torch
import torch.nn as nn

from typing import Dict, Optional, Any
from jaxtyping import Float, Bool
from torch import Tensor
from utils.img import ImgUtils


class Primitive(nn.Module):
    """Base class for trainable geometric image primitives.

    A Primitive represents a learnable geometric representation that can be
    optimized to reconstruct a target image through gradient descent.

    Attributes:
        device: Computed device (torch.device).
        dtype: Computed dtype (torch.dtype).

    Notes:
        - Subclasses must implement `_sample_impl()`, `__len__`, and `parameters` properties.
        - The instance method `sample()` calls `_sample_impl()` with extracted parameters.
        - Uses lazy evaluation via `@lazy_tree` for property caching.
    """

    def __init__(self, **kwargs):
        super().__init__(self.__class__)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Primitive:
        """Deserialize primitive from dict configuration.

        Args:
            data: Dict containing primitive config and optional state_dict path.

        Returns:
            Primitive instance initialized from config.
        """
        primitive = cls(**data)
        state_dict = data.get("state_dict", None)
        if state_dict is not None:
            with open(state_dict, "r") as f:
                state_dict = torch.load(f)
            primitive.load_state_dict(state_dict)
        return primitive

    @torch.no_grad()
    def prepare_for_optimization(
        self, target: Float[Tensor, "..."], patch_size: Optional[int] = None
    ):
        """Prepare buffers for optimization.

        Args:
            target: Target image tensor (C, H, W) or (B, C, H, W).
            patch_size: Optional patch size for patch-based rendering.
        """
        H, W = target.shape[-2:]
        self._buffer_patches, self._buffer_centers = ImgUtils.get_patches(
            H, W, target.device, patch_size=patch_size
        )
        self._buffer_H = H
        self._buffer_W = W

    @torch.no_grad()
    def patch_mask(
        self, center: Float[Tensor, "N 2"], patch_size: int
    ) -> Bool[Tensor, "N"]:
        """Compute mask for valid patches at given centers.

        Args:
            center: Patch center coordinates (N, 2).
            patch_size: Size of patch.

        Returns:
            Bool tensor (N,) indicating which patches are valid.
        """
        return torch.ones((len(self),), dtype=torch.bool, device=self.device)

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
        """Implementation of sampling logic. Subclasses must implement this.

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
        raise NotImplementedError()

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

    def __len__(self) -> int:
        """Number of primitives in this object."""
        raise NotImplementedError()

    def forward(
        self,
        H: int,
        W: int,
        patches: Float[Tensor, "P S 2"],
        centers: Float[Tensor, "P 2"],
    ) -> Float[Tensor, "B C H W"]:
        """Render primitive to full image.

        Args:
            H: Output height.
            W: Output width.
            patches: Patch coordinates (P, S, 2) where P=num_patches, S=patch_size^2.
            centers: Patch centers (P, 2).

        Returns:
            Rendered image (B, C, H, W).
        """
        gen_patches = []
        for patch_idx in range(len(patches)):
            gen_patches.append(
                self.sample(
                    patches[patch_idx], mask=self.patch_mask(centers[patch_idx])
                )
            )
        return ImgUtils.assemble_patches(torch.stack(gen_patches, dim=0), H, W)

    def optim_step(self) -> Float[Tensor, "B C H W"]:
        """Run one optimization step and return rendered output.

        Returns:
            Rendered image using cached buffers (B, C, H, W).
        """
        return self(
            self._buffer_H, self._buffer_W, self._buffer_patches, self._buffer_centers
        )

    def rasterize(
        self, H: int, W: int, patch_size: Optional[int] = None
    ) -> Float[Tensor, "B C H W"]:
        """Rasterize primitive to image tensor.

        Args:
            H: Output height.
            W: Output width.
            patch_size: Optional patch size.

        Returns:
            Image tensor (B, C, H, W).
        """
        patches, centers = ImgUtils.get_patches(
            H, W, self.device, patch_size=patch_size
        )
        return self(H, W, patches, centers)

    def image(
        self, H: int, W: int, patch_size: Optional[int] = None
    ) -> Float[Tensor, "B H W C"]:
        """Convert rasterized output to displayable image.

        Args:
            H: Output height.
            W: Output width.
            patch_size: Optional patch size.

        Returns:
            Image tensor (B, H, W, C) in [0, 1] range.
        """
        return ImgUtils.tensor2img(self.rasterize(H, W, patch_size=patch_size))

    @property
    def parameters(self) -> Dict[str, Float[Tensor, "..."]]:
        """All parameters (trainable and non-trainable)."""
        raise NotImplementedError()

    @property
    def trainable_parameters(self) -> Dict[str, Float[Tensor, "..."]]:
        """Only trainable parameters."""
        raise NotImplementedError()
