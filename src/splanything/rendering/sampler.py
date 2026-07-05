from __future__ import annotations

from typing import Iterator, Optional, Tuple

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from ..primitives import Primitive
from ..utils.img import ImgUtils
from .rasterizers import Rasterizer, WeightedRasterizer


class Sampler:
    def __init__(
        self,
        H: int,
        W: int,
        patch_size: Optional[int] = None,
        max_batch: Optional[int] = None,
        rasterizer: Optional[Rasterizer] = None,
        padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
        low_vram: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        self.H = H
        self.W = W
        self.max_batch = max_batch
        self.low_vram = low_vram
        self.set_patch_size(patch_size, padding)
        self.rasterizer = WeightedRasterizer() if rasterizer is None else rasterizer
        self.padding = padding
        self.device = device

    def set_patch_size(
        self, patch_size: int, padding: Tuple[int, int, int, int] = (0, 0, 0, 0)
    ):
        self.patch_size = patch_size
        self.co_patches, self.co_centers = ImgUtils.get_patches(
            self.H, self.W, device=self.device, patch_size=patch_size, padding=padding
        )

    def set_padding(self, padding: Tuple[int, int, int, int]):
        self.set_patch_size(self.patch_size, padding)
        self.padding = padding

    def to(self, device: torch.device) -> Sampler:
        self.co_patches = self.co_patches.to(device)
        self.co_centers = self.co_centers.to(device)

    @property
    def num_patches(self) -> int:
        if self.patch_size is None:
            return 1
        return self.co_patches.shape[0]

    def samples(
        self,
        p: Primitive,
    ) -> Iterator[Tuple[Float[Tensor, "S C"], Float[Tensor, "S 2"]]]:
        if p.device != self.device:
            raise ValueError(
                f"Sampler and primitive must be on same device. Currently: {self.device}, {p.device}."
            )
        patches = self.co_patches
        centers = self.co_centers
        rasterizer = self.rasterizer
        max_batch = self.max_batch
        P, S, C = patches.shape
        patch_sizes = torch.full(
            (P,), S if P == 1 else int(S**0.5), dtype=torch.long, device=patches.device
        )
        H = torch.full((P,), self.H, dtype=torch.long, device=patches.device)
        W = torch.full((P,), self.W, dtype=torch.long, device=patches.device)
        patch_masks = p.patch_mask(centers, patch_sizes, H, W)  # [P, N]

        def _compute_patch(batch: Float[Tensor, "S 2"], mask: Bool[Tensor, "N"]):
            nonlocal p
            nonlocal rasterizer
            with p.masked(mask):
                return p(batch, rasterizer)

        # If there is no max batch size, just compute per patch.
        if max_batch is None:
            for i in range(P):
                yield _compute_patch(patches[i], patch_masks[i])
        else:
            patch_mask_sums = patch_masks.sum(dim=1)  # [P,]
            i = 0
            mask = torch.empty((len(p),), dtype=torch.bool, device=patches.device)
            while i < P:
                acc_patches = []
                mask_size = 0
                co_size = 0
                batch_size = 0
                mask.zero_()
                while (
                    i < P
                    and batch_size + patches[i].shape[0] * patch_mask_sums[i]
                    < max_batch
                ):
                    mask = mask | patch_masks[i]
                    mask_size = mask.sum()
                    co_size += patches[i].shape[0]
                    batch_size = mask_size * co_size
                    acc_patches.append(patches[i])
                    i += 1
                # Only True if a single patch must be done in multiple passes
                if len(acc_patches) == 0:
                    b_patches = torch.chunk(
                        patches[i],
                        patch_mask_sums[i] * patches[i].shape[0] // max_batch,
                        dim=0,
                    )
                    mask = patch_masks[i]
                    for b in b_patches:
                        yield _compute_patch(b, mask)
                    i += 1
                else:
                    batch_co = torch.cat(acc_patches, dim=0)
                    yield _compute_patch(batch_co, mask), batch_co

    def rasterize(
        self,
        p: Primitive,
        max_batch: Optional[int] = None,
        low_vram: Optional[bool] = None,
    ) -> Float[Tensor, "B C H W"]:
        if p.device != self.device:
            raise ValueError(
                f"Sampler and primitive must be on same device. Currently: {self.device}, {p.device}."
            )
        low_vram = self.low_vram if low_vram is None else low_vram
        P, S, C = self.co.patches.shape
        gen = []
        for patch in self.samples(p, max_batch):
            if low_vram:
                patch = patch.cpu()
            gen.append(patch)
        patch_gen = torch.cat(gen, dim=0).reshape(P, S, 4)
        return ImgUtils.assemble_patches(patch_gen, self.H, self.W)
