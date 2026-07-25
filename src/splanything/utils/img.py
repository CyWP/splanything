from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from PIL import Image
from torch import Tensor


class ImgUtils:
    """Image processing utilities for tensor operations.

    Static methods for converting between image and tensor formats,
    patch extraction/assembly, Gaussian kernels, and SSIM computation.
    """

    @staticmethod
    def img2tensor(
        img: Float[Tensor, "B H W C"],
    ) -> Float[Tensor, "B C H W"]:
        """Convert [0,1] HWC image to CHW tensor.

        Args:
            img: Image tensor (B, H, W, C) in [0, 1].

        Returns:
            Tensor (B, C, H, W) in [0, 1].
        """
        return img.permute(0, 3, 1, 2)

    @staticmethod
    def tensor2img(
        x: Float[Tensor, "B C H W"],
        clamp: bool = True,
        mode: str = "RGBA",
    ) -> Float[Tensor, "B H W C"]:
        """Convert [0,1] CHW tensor to HWC image.

        Args:
            x: Tensor (B, C, H, W) in [0, 1].

        Returns:
            Image (B, H, W, C) in [0, 1].
        """
        img = x.permute(0, 2, 3, 1)
        if clamp:
            img = img.clamp(0, 1)
        return img

    @staticmethod
    def tensor2pil(
        x: Float[Tensor, "B C H W"],
    ) -> Union[Image.Image, List[Image.Image]]:
        """Convert tensor to PIL Image.

        Args:
            x: Tensor (B, C, H, W) in [0, 1].

        Returns:
            PIL Image as uint8 [0, 255].
        """
        B, C, H, W = x.shape
        mode = "RGB" if C == 3 else "RGBA"
        img = ImgUtils.tensor2img(x, clamp=True)
        img_np = (img.cpu().numpy() * 255).astype(np.uint8)
        imgs = []
        for i in range(B):
            imgs.append(Image.fromarray(img_np[i], mode=mode))
        if B == 1:
            return imgs[0]
        return imgs

    @staticmethod
    def pil2tensor(
        img: Union[Image.Image, List[Image.Image], Sequence[Image.Image]],
    ) -> Float[Tensor, "B C H W"]:
        """Convert PIL Image to tensor.

        Args:
            img: PIL Image or sequence of PIL Images as uint8 [0, 255].

        Returns:
            Tensor (B, C, H, W) in [0, 1].
        """
        if isinstance(img, Image.Image):
            imgs = [img]
        else:
            imgs = list(img)
        arrs = [np.asarray(im.convert("RGBA")) for im in imgs]
        stacked = np.stack(arrs, axis=0)
        return torch.from_numpy(stacked).permute(0, 3, 1, 2).float() / 255.0

    @staticmethod
    def pil2map(
        img: Union[Image.Image, List[Image.Image], Sequence[Image.Image]],
        mode: str = "mean",
    ) -> Float[Tensor, "B 1 H W"]:
        return ImgUtils.tensor2map(ImgUtils.pil2tensor(img), mode=mode)

    @staticmethod
    def tensor2map(
        x: Float[Tensor, "B C H W"],
        mode: str = "mean",
    ) -> Float[Tensor, "B 1 H W"]:
        """Reduce a multi-channel image tensor to a single-channel map.

        Args:
            x: Image tensor (B, C, H, W).
            mode: Reduction mode. One of "R", "G", "B", "A" (select the
                corresponding channel) or "mean" (average across channels).

        Returns:
            Single-channel map (B, 1, H, W).

        Raises:
            ValueError: If mode is not one of "R", "G", "B", "A", "mean".
        """
        channel_map = {"R": 0, "G": 1, "B": 2, "A": 3}
        if mode == "mean":
            return x.mean(dim=1, keepdim=True)
        if mode in channel_map:
            idx = channel_map[mode]
            return x[:, idx : idx + 1, :, :]
        raise ValueError(
            f"Unknown mode '{mode}'; expected one of 'R', 'G', 'B', 'A', 'mean'."
        )

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
    def gen_px_coords(
        H: int,
        W: int,
        device: torch.device,
        padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> Float[Tensor, "2 H W"]:
        """Generate normalized pixel coordinates.

        Args:
            H: Image height.
            W: Image width.
            device: Target device.

        Returns:
            Coordinates tensor (2, H, W) with values in [0, 1].
        """
        p_top, p_bot, p_left, p_right = padding
        H_frame = H - p_top - p_bot
        W_frame = W - p_left - p_right
        H_half = 0.5 / H_frame if H_frame != 0 else 0
        W_half = 0.5 / W_frame if W_frame != 0 else 0
        co = torch.stack(
            torch.meshgrid(
                torch.linspace(H_half, 1 - H_half, H, device=device),
                torch.linspace(W_half, 1 - W_half, W, device=device),
                indexing="ij",
            ),
            dim=0,
        )
        return ImgUtils.coords_pad(co, padding=padding)

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
        co = ImgUtils.coords_pad(co, padding=(0, pad_H, 0, pad_W))
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
        H: int,
        W: int,
        device: torch.device,
        patch_size: Optional[int] = None,
        padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
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
            ImgUtils.gen_px_coords(H, W, device, padding=padding), patch_size
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
        co: Float[Tensor, "C H W"],
        padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> Float[Tensor, "C (H+pad_top+pad_bottom) (W+pad_left+pad_right)"]:
        pad_top, pad_bottom, pad_left, pad_right = padding
        if padding == (0, 0, 0, 0):
            return co
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
        y_extra_top = (
            (
                y_coords[0]
                - torch.arange(1, pad_top + 1, device=co.device, dtype=co.dtype)
                * y_step
            ).flip(0)
            if pad_top > 0
            else torch.empty((0,), device=co.device, dtype=co.dtype)
        )
        y_extra_bottom = (
            (
                y_coords[-1]
                + torch.arange(1, pad_bottom + 1, device=co.device, dtype=co.dtype)
                * y_step
            )
            if pad_bottom > 0
            else torch.empty((0,), device=co.device, dtype=co.dtype)
        )
        x_extra_left = (
            (
                x_coords[0]
                - torch.arange(1, pad_left + 1, device=co.device, dtype=co.dtype)
                * x_step
            ).flip(0)
            if pad_left > 0
            else torch.empty((0,), device=co.device, dtype=co.dtype)
        )
        x_extra_right = (
            (
                x_coords[-1]
                + torch.arange(1, pad_right + 1, device=co.device, dtype=co.dtype)
                * x_step
            )
            if pad_right > 0
            else torch.empty((0,), device=co.device, dtype=co.dtype)
        )
        y_full = torch.cat([y_extra_top, y_coords, y_extra_bottom], dim=0)
        x_full = torch.cat([x_extra_left, x_coords, x_extra_right], dim=0)
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
        # Single-patch fallback (extract_image_patches returns S = H*W when
        # patch_size > H and > W; S is then not necessarily a perfect square,
        # so F.fold cannot reconstruct it). Reshape directly.
        if P == 1 and patch_size * patch_size != S:
            if H is None or W is None:
                raise Exception(
                    "assemble_patches needs H, W for the single-patch fallback."
                )
            if S != H * W:
                raise Exception(
                    f"single-patch fallback expects S == H*W, got S={S}, H*W={H * W}."
                )
            return sampled_patches.permute(2, 1, 0).reshape(1, C, H, W)  # (1, C, H, W)
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
    def load_image(path: str, mode: str = "RGBA") -> Float[Tensor, "B C H W"]:
        """Load image from path as tensor.

        Args:
            path: Path to image file (PNG, JPG, etc.).
            mode: Color mode for PIL Image ("RGBA", "RGB", "L", etc.).

        Returns:
            Image tensor (B, C, H, W) with values in [0, 1].
        """
        img = Image.open(path).convert(mode)
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0).permute(0, 3, 1, 2)

    @staticmethod
    def img2map(
        img: Float[Tensor, "B C H W"], min: float = -1.0, max: float = 1.0
    ) -> Float[Tensor, "B 1 H W"]:
        return (img.mean(dim=1).unsqueeze(1) + 1) / 2

    @staticmethod
    def load_map(path: str) -> Float[Tensor, "B 1 H W"]:
        return ImgUtils.img2map(ImgUtils.load_image(path, mode="RGBA"))

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

    @staticmethod
    def uv_sample(
        img: Float[Tensor, "B C H W"],
        uv_co: Float[Tensor, "N 2"],
    ) -> Float[Tensor, "B N C"]:
        """Bilinearly sample an image at normalized pixel-center coordinates.

        Uses the **pixel-center convention** matching :meth:`gen_px_coords`:
        the center of pixel ``i`` along an axis of size ``D`` sits at UV
        ``(i + 0.5) / D``. Equivalently, the pixel index for UV ``u`` is
        ``u * D - 0.5``. UVs outside ``[0, 1]`` (e.g. centroids that have
        drifted past the image frame after a split) are clamped to the
        nearest edge pixel via the fractional weights, so they return a
        valid sample rather than extrapolating.

        Args:
            img: Image tensor (B, C, H, W).
            uv_co: Normalized coordinates (N, 2) in pixel-center convention,
                typically produced by :meth:`gen_px_coords` or by
                ``primitive.centroids``.

        Returns:
            Sampled values (B, N, C).
        """
        B, C, H, W = img.shape
        N = uv_co.shape[0]

        y = uv_co[:, 0] * H - 0.5
        x = uv_co[:, 1] * W - 0.5

        x0 = x.floor().long().clamp(0, W - 1)
        y0 = y.floor().long().clamp(0, H - 1)
        x1 = (x0 + 1).clamp(0, W - 1)
        y1 = (y0 + 1).clamp(0, H - 1)

        # Clamp fractional weights so out-of-frame UVs clamp cleanly to the
        # nearest edge pixel instead of extrapolating with negative or >1
        # weights.
        fx = (x - x0.float()).clamp(0, 1).view(1, 1, N)
        fy = (y - y0.float()).clamp(0, 1).view(1, 1, N)
        inv_fx = 1 - fx
        inv_fy = 1 - fy

        tl = img[:, :, y0, x0]
        tr = img[:, :, y0, x1]
        bl = img[:, :, y1, x0]
        br = img[:, :, y1, x1]

        vals = (
            tl * (inv_fx * inv_fy)
            + tr * (fx * inv_fy)
            + bl * (inv_fx * fy)
            + br * (fx * fy)
        )
        return vals.permute(0, 2, 1)  # (B, N, C)

    @staticmethod
    @torch.no_grad()
    def sample_px_coords(
        map: Float[Tensor, "B 1 H W"],
        N: int,
        noise: bool = False,
    ) -> Float[Tensor, "N 2"]:
        """Sample N pixel coordinates weighted by a map.

        Args:
            map: Weight map (B, 1, H, W) with non-negative values.
            N: Number of coordinates to sample.
            noise: If True, jitter each sampled coordinate uniformly within
                half a pixel on either side of its pixel center.

        Returns:
            Sampled coordinates (N, 2) in pixel-center convention.

        Notes:
            - Sampling uses ``torch.multinomial`` with replacement, so
              coordinates may repeat.
            - Coordinates follow the pixel-center convention of
              :meth:`gen_px_coords` (centered at ``(i + 0.5) / D``).
            - With ``noise=True``, jitter is uniform in ``[-0.5/H, 0.5/H]`` for
              y and ``[-0.5/W, 0.5/W]`` for x, i.e. exactly one pixel width
              centered on each pixel.
        """
        B, _, H, W = map.shape
        if B != 1:
            raise ValueError(
                f"sample_px_coords expects a single-map batch (B=1), got B={B}."
            )
        weights = map.reshape(-1)  # (H*W,)
        if (weights < 0).any():
            raise ValueError("sample_px_coords expects non-negative map values.")
        indices = torch.multinomial(weights, N, replacement=True)  # (N,)
        co = ImgUtils.gen_px_coords(H, W, map.device)  # (2, H, W)
        co_flat = co.reshape(2, -1).T  # (H*W, 2)
        sampled = co_flat[indices]  # (N, 2)
        if noise:
            sampled = sampled.clone()
            sampled[:, 0].add_((torch.rand(N, device=map.device) - 0.5) / H)
            sampled[:, 1].add_((torch.rand(N, device=map.device) - 0.5) / W)
        return sampled
