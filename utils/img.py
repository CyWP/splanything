from __future__ import annotations

import math
import torch

from typing import Optional, Tuple

F = torch.FloatTensor


class ImgUtils:

    @staticmethod
    def img2tensor(img: F) -> F:
        return img * 2 - 1

    @staticmethod
    def tensor2img(x: F) -> F:
        return ((x + 1) * 2).clamp(0, 1)

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
