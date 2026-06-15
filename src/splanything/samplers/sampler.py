import torch

from jaxtyping import Float, Bool
from torch import Tensor
from typing import Optional, Iterator
from splanything.primitives import Primitive
from splanything.rasterizers import Rasterizer, WeightedRasterizer
from splanything.utils.img import ImgUtils
from splanything.vars import LOW_VRAM, DEVICE


class Sampler:
    def __init__(
        self,
        H: int,
        W: int,
        patch_size: Optional[int] = None,
        max_batch: Optional[int] = None,
        rasterizer: Optional[Rasterizer] = None,
        device: Optional[torch.device] = None,
    ):
        self.H = H
        self.W = W
        self.max_batch = max_batch
        self.device = DEVICE if device is None else device
        self.set_patch_size(patch_size)
        self.rasterizer = WeightedRasterizer() if rasterizer is None else rasterizer

    def set_patch_size(self, patch_size: int):
        self.patch_size = patch_size
        self.co_patches, self.co_centers = ImgUtils.get_patches(
            self.H, self.W, device=self.device, patch_size=patch_size
        )

    @property
    def num_patches(self) -> int:
        if self.patch_size is None:
            return 1
        return self.co_patches.shape[0]

    def samples(
        self,
        p: Primitive,
    ) -> Iterator[Float[Tensor, "S C"]]:
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

        def _compute_patch(batch: Float[Tensor, "B C"], mask: Bool[Tensor, "N_splats"]):
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
                    yield _compute_patch(torch.cat(acc_patches, dim=0), mask)

    def rasterize(
        self,
        p: Primitive,
        max_batch: Optional[int] = None,
        low_vram: Optional[bool] = None,
    ) -> Float[Tensor, "B C H W"]:
        low_vram = LOW_VRAM if low_vram is None else low_vram
        P, S, C = self.co.patches.shape
        gen = []
        for patch in self.samples(p, max_batch):
            if low_vram:
                patch = patch.cpu()
            gen.append(patch)
        patch_gen = torch.cat(gen, dim=0).reshape(P, S, 4)
        return ImgUtils.assemble_patches(patch_gen, self.H, self.W)
