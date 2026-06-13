from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F

from typing import Optional, Tuple, Union, Sequence, List
from jaxtyping import Float
from PIL import Image
from torch import Tensor

from .lazy import lazy_tree


class ImgUtils:
    """Image processing utilities for tensor operations.

    Static methods for converting between image and tensor formats,
    patch extraction/assembly, Gaussian kernels, and SSIM computation.
    """

    @staticmethod
    def img2tensor(
        img: Float[Tensor, "B H W C"], normalize: bool = True
    ) -> Float[Tensor, "B C H W"]:
        """Convert [0,1] HWC image to [-1,1] CHW tensor.

        Args:
            img: Image tensor (B, H, W, C) in [0, 1].

        Returns:
            Tensor (B, C, H, W) in [-1, 1].
        """
        x = img.permute(0, 3, 1, 2)
        if normalize:
            return x * 2 - 1
        return x

    @staticmethod
    def tensor2img(
        x: Float[Tensor, "B C H W"],
        normalized: bool = True,
        clamp: bool = True,
        mode: str = "RGBA",
    ) -> Float[Tensor, "B H W C"]:
        """Convert [-1,1] CHW tensor to [0,1] HWC image.

        Args:
            x: Tensor (B, C, H, W) in [-1, 1].

        Returns:
            Image (B, H, W, C) in [0, 1].
        """
        B, C, H, W = x.shape
        img = x.permute(0, 2, 3, 1)
        if normalized:
            img = (img + 1) / 2
        if clamp:
            img = img.clamp(0, 1)
        return img

    @staticmethod
    def tensor2pil(
        x: Float[Tensor, "B C H W"], normalized: bool = True
    ) -> Union[Image.Image, List[Image.Image]]:
        """Convert tensor to PIL Image.

        Args:
            x: Tensor (B, C, H, W) in [-1, 1] or [0, 1].

        Returns:
            PIL Image as uint8 [0, 255].
        """
        B, C, H, W = x.shape
        mode = "RGB" if C == 3 else "RGBA"
        img = ImgUtils.tensor2img(x, normalized=normalized, clamp=True)
        img_np = (img.cpu().numpy() * 255).astype(np.uint8)
        imgs = []
        for i in range(B):
            imgs.append(Image.fromarray(img_np[i], mode=mode))
        if B == 1:
            return imgs[0]
        return imgs

    @staticmethod
    def resize(
        img: Float[Tensor, "B C H W"],
        H: int,
        W: int,
        mode: str = "bilinear",
        align_corners: Optional[bool] = None,
        antialias: bool = False,
    ) -> Float[Tensor, "B C H W"]:
        """Resize image tensor to target height and width.

        Args:
            img: Input image (B, C, H_in, W_in).
            H: Target height.
            W: Target width.
            mode: Interpolation mode for F.interpolate (default "bilinear").
            align_corners: Optional align_corners argument.
            antialias: Apply antialiasing on downscaling.

        Returns:
            Resized image (B, C, H, W).

        Notes:
            - Requires batch dimension; never squeezes.
            - align_corners is ignored for "nearest" and "area" modes.
        """
        if H < 1 or W < 1:
            raise Exception(f"Target size must be positive, got H={H}, W={W}.")

        kwargs = {"antialias": antialias}
        if mode not in ("nearest", "area"):
            kwargs["align_corners"] = align_corners

        return F.interpolate(img, size=(H, W), mode=mode, **kwargs)

    @staticmethod
    @torch.no_grad()
    def ensure_rgba(img: Float[Tensor, "B C H W"]) -> Float[Tensor, "B 4 H W"]:
        """Ensure image has 4 channels (RGBA).

        Args:
            img: Input tensor (B, C, H, W).

        Returns:
            RGBA tensor (B, 4, H, W).
        """
        B, C, H, W = img.shape
        if C == 4:
            return img
        elif C == 3:
            return torch.cat(
                [img, torch.ones((B, 1, H, W), device=img.device, dtype=img.dtype)],
                dim=1,
            )
        elif C == 1:
            return torch.cat(
                [
                    img.repeat(1, 3, 1, 1),
                    torch.ones((B, 1, H, W), device=img.device, dtype=img.dtype),
                ],
                dim=1,
            )
        else:
            raise Exception(
                f"Cannot recognize image format for tensor with shape {img.shape}"
            )

    @staticmethod
    @torch.no_grad()
    def ensure_rgb(img: Float[Tensor, "B C H W"]) -> Float[Tensor, "B 4 H W"]:
        """Ensure image has 3 channels (RGB).

        Args:
            img: Input tensor (B, C, H, W).

        Returns:
            RGBA tensor (B, 4, H, W).
        """
        B, C, H, W = img.shape
        if C == 3:
            return img
        elif C == 4:
            return img[:, :3] * img[:, 3].unsqueeze(1)
        elif C == 1:
            return img.repeat(1, 3, 1, 1)
        else:
            raise Exception(
                f"Cannot recognize image format for tensor with shape {img.shape}"
            )

    @staticmethod
    @torch.no_grad()
    def gen_px_coords(H: int, W: int, device: torch.device) -> Float[Tensor, "2 H W"]:
        """Generate normalized pixel coordinates.

        Args:
            H: Image height.
            W: Image width.
            device: Target device.

        Returns:
            Coordinates tensor (2, H, W) with values in [0, 1].
        """
        H_half = 0.5 / H
        W_half = 0.5 / W
        out = torch.stack(
            torch.meshgrid(
                torch.linspace(H_half, 1 - H_half, H, device=device),
                torch.linspace(W_half, 1 - W_half, W, device=device),
                indexing="ij",
            ),
            dim=0,
        )
        return out

    @staticmethod
    def extract_patches(
        co: Float[Tensor, "C H W"], patch_size: Optional[int] = None
    ) -> Tuple[Float[Tensor, "P S C"], Float[Tensor, "P C"]]:
        """Extract patches from coordinate grid.

        Args:
            co: Coordinate tensor (C, H, W).
            patch_size: Size of patches to extract.

        Returns:
            Tuple of (patches, centers):
                - patches: (P, S, C) where P=num_patches, S=patch_size^2
                - centers: (P, C)
        """
        if patch_size < 1:
            raise Exception("Patch size must be strictly positive integer.")
        C, H, W = co.shape
        if patch_size is None or all(patch_size > d for d in [H, W]):
            patches = co.permute(1, 2, 0).reshape(1, H * W, C)  # [1, S, C]
            centers = torch.tensor(
                [0.5, 0.5], device=co.device, dtype=co.dtype
            ).unsqueeze(0)
            return patches, centers
        S = patch_size**2
        pad_H = (patch_size - (H % patch_size)) % patch_size
        pad_W = (patch_size - (W % patch_size)) % patch_size
        co = ImgUtils.coords_pad(co, pad_H, pad_W)
        patches = (
            F.unfold(
                co.unsqueeze(0),
                kernel_size=patch_size,
                stride=patch_size,
            )
            .reshape(C, S, -1)
            .permute(2, 1, 0)
        )  # [P, S, C]
        centers = patches.mean(dim=1)
        return patches, centers

    @staticmethod
    def get_patches(
        H: int, W: int, device: torch.device, patch_size: Optional[int] = None
    ) -> Tuple[Float[Tensor, "P S 2"], Float[Tensor, "P 2"]]:
        """Get patches for image dimensions.

        Args:
            H: Image height.
            W: Image width.
            device: Target device.
            patch_size: Optional patch size.

        Returns:
            Tuple of (patches, centers).
        """
        return ImgUtils.extract_patches(
            ImgUtils.gen_px_coords(H, W, device), patch_size
        )

    @staticmethod
    def extract_image_patches(
        img: Float[Tensor, "B C H W"],
        patch_size: Optional[int],
        padding_mode: str = "replicate",
    ) -> Float[Tensor, "B P S C"]:
        """Extract image patches matching the layout of get_patches coordinates.

        Pads the image so H and W are divisible by patch_size, then extracts
        non-overlapping square patches using im2col/unfold.

        Args:
            img: Input image (B, C, H, W).
            patch_size: Size of square patches. If None or larger than both H and W,
                returns a single patch containing all H*W pixels.
            padding_mode: Padding mode for F.pad (default "replicate").

        Returns:
            Image patches (B, P, S, C). S is patch_size**2 normally, or H*W in
            the single-patch fallback.

        Notes:
            - Patch ordering matches ImgUtils.get_patches row-major layout.
            - Fallback behavior matches ImgUtils.extract_patches exactly.
            - Never squeezes batch dimension.
        """
        if patch_size is not None and patch_size < 1:
            raise Exception("Patch size must be strictly positive integer.")

        B, C, H, W = img.shape

        # Fallback: matches extract_patches when patch_size is None or > H and > W
        if patch_size is None or (patch_size > H and patch_size > W):
            return img.permute(0, 2, 3, 1).reshape(B, 1, H * W, C)  # (B, 1, H*W, C)

        S = patch_size * patch_size
        pad_H = (patch_size - (H % patch_size)) % patch_size
        pad_W = (patch_size - (W % patch_size)) % patch_size

        # F.pad order: (left, right, top, bottom) — pad only right/bottom
        padded = F.pad(
            img, (0, pad_W, 0, pad_H), mode=padding_mode
        )  # (B, C, H+pad_H, W+pad_W)

        patches = F.unfold(
            padded, kernel_size=patch_size, stride=patch_size
        )  # (B, C*S, P)
        patches = patches.view(B, C, S, -1).permute(0, 3, 2, 1)  # (B, P, S, C)
        return patches

    @staticmethod
    def coords_pad(
        co: Float[Tensor, "C H W"], pad_H: int, pad_W: int
    ) -> Float[Tensor, "C (H+pad_H) (W+pad_W)"]:
        C, H, W = co.shape
        y_coords = co[0, :, 0]
        x_coords = co[1, 0, :]
        y_step = (
            y_coords[1] - y_coords[0]
            if H > 1
            else torch.tensor(1.0 / H, device=co.device, dtype=co.dtype)
        )
        x_step = (
            x_coords[1] - x_coords[0]
            if W > 1
            else torch.tensor(1.0 / W, device=co.device, dtype=co.dtype)
        )
        y_extra = (
            y_coords[-1]
            + torch.arange(1, pad_H + 1, device=co.device, dtype=co.dtype) * y_step
        )
        x_extra = (
            x_coords[-1]
            + torch.arange(1, pad_W + 1, device=co.device, dtype=co.dtype) * x_step
        )
        y_full = torch.cat([y_coords, y_extra], dim=0)
        x_full = torch.cat([x_coords, x_extra], dim=0)
        yy, xx = torch.meshgrid(y_full, x_full, indexing="ij")
        return torch.stack([yy, xx], dim=0)

    @staticmethod
    def assemble_patches(
        sampled_patches: Float[Tensor, "P S C"],
        H: Optional[int] = None,
        W: Optional[int] = None,
    ) -> Float[Tensor, "B C H W"]:
        """Assemble sampled patches into full image.

        Args:
            sampled_patches: Sampled patches (P, S, C) where S=patch_size^2.
            H: Output height.
            W: Output width.

        Returns:
            Assembled image (B, C, H, W).
        """
        P, S, C = sampled_patches.shape
        patch_size = int(S**0.5)
        patches_H = math.ceil(H / patch_size)
        patches_W = math.ceil(W / patch_size)
        output_H = patch_size * patches_H
        output_W = patch_size * patches_W
        # if output_H * output_W != P:
        #     raise Exception(f"patches do not match image output size.")
        folded = F.fold(
            sampled_patches.permute(2, 1, 0).reshape(1, C * S, P),
            output_size=(output_H, output_W),
            kernel_size=patch_size,
            stride=patch_size,
        )
        return folded[:, :, :H, :W]

    @staticmethod
    @torch.no_grad()
    def gaussian_kernel(
        kernel_size: Union[int, Sequence[int]],
        sigma: Union[float, Sequence[float]],
    ) -> Float[Tensor, "1 1 KH KW"]:
        """Generate 2D Gaussian kernel.

        Args:
            kernel_size: Kernel size (int or [H, W]).
            sigma: Standard deviation (float or [H, W]).

        Returns:
            Gaussian kernel (1, 1, KH, KW).
        """
        if isinstance(kernel_size, int):
            kH, kW = kernel_size, kernel_size
        elif len(kernel_size) == 1:
            kH, kW = kernel_size[0], kernel_size[0]
        elif len(kernel_size) == 2:
            kH, kW = kernel_size
        else:
            raise Exception(
                f"2D gaussian kernel cannot accept more than 2 axes. Kernel '{kernel_size} is too long."
            )
        if isinstance(sigma, int):
            sH, sW = sigma, sigma
        elif len(sigma) == 1:
            sH, sW = sigma[0], sigma[0]
        elif len(sigma) == 2:
            sH, sW = sigma
        else:
            raise Exception(
                f"2D gaussian kernel cannot accept more than 2 axes. Sigmas '{kernel_size} are too many."
            )
        sq2pi = (2 * math.pi) ** 0.5
        muH = (kH - 1) / 2
        xH = torch.exp(-0.5 * ((torch.linspace(0, kH - 1, kH) - muH) / sH) ** 2) / (
            sH * sq2pi
        )
        if kH == kW and sH == sW:
            xW = xH
        else:
            muW = (kW - 1) / 2
            xW = torch.exp(-0.5 * ((torch.linspace(0, kW - 1, kW) - muW) / sW) ** 2) / (
                sW * sq2pi
            )
        return (xH.unsqueeze(1) @ xW.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

    @staticmethod
    def convolve(
        img: Float[Tensor, "B C H W"],
        kernel: Float[Tensor, "C G KH KW"],
        match_channels: bool = False,
        stride: Union[int, Tuple[int]] = 1,
        padding: str = "same",
    ) -> Float[Tensor, "B C H W"]:
        """Convolve image with kernel.

        Args:
            img: Input image (B, C, H, W).
            kernel: Convolution kernel (C, G, KH, KW).
            match_channels: If True, expand kernel to match input channels.
            stride: Convolution stride.
            padding: Padding mode ("same" or "valid").

        Returns:
            Convolved output (B, C, H, W).
        """
        Bi, Ci, Hi, Wi = img.shape
        Ck, Gk, Hk, Wk = kernel.shape
        if match_channels:
            if Ck != 1 or Gk != 1:
                raise Exception(
                    f"Cannot match channels for kernel with non-singleton dimensions.\nKernel shape: {kernel.shape}"
                )
            kernel = kernel.expand(Ci, Ci, -1, -1)
        return F.conv2d(img, kernel, stride=stride, padding=padding)

    @staticmethod
    def SSIM(
        img1: Float[Tensor, "B C H W"],
        img2: Float[Tensor, "B C H W"],
        kernel: Float[Tensor, "1 1 KH KW"],
        eps1: float = 0.0004,
        eps2: float = 0.0036,
    ) -> Float[Tensor, "B C H W"]:
        """Compute Structural Similarity Index (SSIM).

        Args:
            img1: First image (B, C, H, W).
            img2: Second image (B, C, H, W).
            kernel: Gaussian kernel (1, 1, KH, KW).
            eps1: Stability constant for means.
            eps2: Stability constant for variances.

        Returns:
            SSIM map (B, C, H, W) with values in [-1, 1].
        """
        mux = ImgUtils.convolve(img1, kernel, match_channels=True)
        muy = ImgUtils.convolve(img2, kernel, match_channels=True)
        mu2x = mux**2
        mu2y = muy**2
        sig2x = ImgUtils.convolve(img1**2, kernel, match_channels=True) ** 2 - mu2x
        sig2y = ImgUtils.convolve(img2**2, kernel, match_channels=True) ** 2 - mu2y
        sigxy = (
            ImgUtils.convolve(img1 * img2, kernel, match_channels=True) ** 2 - mux * muy
        )
        return (
            (2 * mux * muy + eps1)
            * (sigxy + eps2)
            / ((mu2x + mu2y + eps2) * (sig2x + sig2y + eps2))
        )

    @staticmethod
    def load_image(
        path: str, mode: str = "RGBA", normalize: bool = False
    ) -> Float[Tensor, "B C H W"]:
        """Load image from path as tensor.

        Args:
            path: Path to image file (PNG, JPG, etc.).
            mode: Color mode for PIL Image ("RGBA", "RGB", "L", etc.).
            normalize: If True, normalize to [-1, 1] instead of [0, 1].

        Returns:
            Image tensor (B, C, H, W) with values in [0, 1] or [-1, 1].
        """
        img = Image.open(path).convert(mode)
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)
        tensor = tensor.permute(0, 3, 1, 2)
        if normalize:
            tensor = tensor * 2 - 1
        return tensor

    @staticmethod
    def img2map(
        img: Float[Tensor, "B C H W"], min: float = -1.0, max: float = 1.0
    ) -> Float[Tensor, "B 1 H W"]:
        return (img.mean(dim=1).unsqueeze(1) + 1) / 2

    @staticmethod
    def load_map(path: str) -> Float[Tensor, "B 1 H W"]:
        return ImgUtils.img2map(ImgUtils.load_image(path, mode="RGBA", normalize=False))

    @staticmethod
    def same_size(*imgs: Float[Tensor, "B C H W"]) -> bool:
        if len(imgs) <= 1:
            raise ValueError(
                f"Function requires a minimum of 2 images to compare. Provided {len(imgs)}."
            )
        ref = imgs[0].shape[-2:]
        for i in imgs[1:]:
            if i.shape[-2:] != ref:
                return False
        return True
