from __future__ import annotations

import math
import torch
import torch.nn.functional as torchF

from typing import Optional, Tuple, Union, Sequence

F = torch.FloatTensor


class ImgUtils:

    @staticmethod
    def img2tensor(img: F) -> F:
        return (img * 2 - 1).permute(0, 3, 1, 2)

    @staticmethod
    def tensor2img(x: F) -> F:
        return (((x + 1) * 2).clamp(0, 1)).permute(0, 2, 3, 1)

    @staticmethod
    @torch.no_grad()
    def ensure_rgba(img: F) -> F:
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
    def gen_px_coords(H: int, W: int, device: torch.device) -> F:
        H_half = 0.5 / H
        W_half = 0.5 / W
        return torch.cat(
            torch.meshgrid(
                torch.linspace(H_half, 1 - H_half, H, device=device),
                torch.linspace(W_half, 1 - W_half, W, device=device),
                indexing="ij",
            ),
            dim=0,
        )

    @staticmethod
    @torch.no_grad()
    def extract_patches(co: F, patch_size: Optional[int]) -> Tuple[F, F]:
        if patch_size < 1:
            raise Exception("Patch size must be strictly positive integer.")
        C, H, W = co.shape
        if patch_size is None or all(patch_size < d for d in [H, W]):
            return co.reshape(C, -1).permute(0, 1), torch.tensor(
                [0.5, 0.5], device=co.device
            ).unsqueeze(0)
        pad_H = H % patch_size
        if pad_H != 0:
            pad_H == pad_H // 2 + 1
        pad_W = W % patch_size
        if pad_W != 0:
            pad_W == pad_W // 2 + 1
        patches = F.unfold(
            co.unsqueeze(0),
            kernel_size=patch_size,
            stride=patch_size,
            padding=(pad_H, pad_W),
        ).squeeze(
            0
        )  # [num_patches, patch_size**2, 2]
        centers = patches.mean(dim=1)  # [num_patches, 2]
        return patches, centers

    @staticmethod
    @torch.no_grad()
    def get_patches(
        H: int, W: int, device: torch.device, patch_size: Optional[int] = None
    ) -> Tuple[F, F]:
        return ImgUtils.extract_patches(ImgUtils.gen_coords(H, W, device), patch_size)

    @staticmethod
    @torch.no_grad()
    def assemble_patches(sampled_patches: F, H: int, W: int) -> F:
        num_patches, patch_size_sq, C = sampled_patches.shape
        patch_size = int(patch_size_sq**0.5)
        patches_H = math.ceil(H / patch_size)
        patches_W = math.ceil(W / patch_size)
        assembled = F.fold(
            sampled_patches,
            (patch_size * patches_H, patch_size * patches_W),
            kernel_size=patch_size,
            stride=patch_size,
        )
        assembled_H, assembled_W = assembled.shape[-2:]
        pad_H = 0 if assembled_H == H else (assembled_H - pad_H) // 2
        pad_W = 0 if assembled_W == W else (assembled_W - pad_W) // 2
        return assembled[..., pad_H : H + pad_H, pad_W : W + pad_W]

    @staticmethod
    @torch.no_grad()
    def gaussian_kernel(
        kernel_size: Union[int, Sequence[int]],
        sigma: Union[float, Sequence[float]],
    ) -> F:
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
        img: F,
        kernel: F,
        match_channels: bool = False,
        stride: Union[int, Tuple[int]] = 1,
        padding: str = "same",
    ) -> F:
        Bi, Ci, Hi, Wi = img.shape
        Ck, Gk, Hk, Wk = kernel.shape
        if match_channels:
            if Ck != 1 or Gk != 1:
                raise Exception(
                    f"Cannot match channels for kernel with non-singleton dimensions.\nKernel shape: {kernel.shape}"
                )
            kernel = kernel.expand(Ci, Ci, -1, -1)
        return torchF.conv2d(img, kernel, stride=stride, padding=padding)

    @staticmethod
    def SSIM(
        img1: F, img2: F, kernel: F, eps1: float = 0.0004, eps2: float = 0.0036
    ) -> F:
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
