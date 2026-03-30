from __future__ import annotations
import torch
import torch.nn as nn

from typing import Dict, Optional
from utils.img import ImgUtils

F = torch.FloatTensor
L = torch.LongTensor
B = torch.BoolTensor


class Primitive(nn.Module):

    @torch.no_grad()
    def prepare_for_optimization(self, target: F, patch_size: Optional[int] = None):
        H, W = target.shape[-2:]
        self._buffer_patches, self._buffer_centers = ImgUtils.get_patches(
            H, W, target.device, patch_size=patch_size
        )
        self._buffer_H = H
        self._buffer_W = W

    @torch.no_grad()
    def patch_mask(self, center: F, patch_size: int) -> B:
        return torch.ones((len(self),), dtype=torch.bool, device=self.device)

    def sample(self, co: F, mask: Optional[B] = None) -> F:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def forward(self, H: int, W: int, patches: F, centers: F) -> F:
        gen_patches = []
        for patch_idx in range(len(patches)):
            gen_patches.append(
                self.sample(
                    patches[patch_idx], mask=self.patch_mask(centers[patch_idx])
                )
            )
        return ImgUtils.assemble_patches(torch.stack(gen_patches, dim=0), H, W)

    def optim_step(self) -> F:
        return self(
            self._buffer_H, self._buffer_W, self._buffer_patches, self._buffer_centers
        )

    def rasterize(self, H: int, W: int, patch_size: Optional[int] = None) -> F:
        patches, centers = ImgUtils.get_patches(
            H, W, self.device, patch_size=patch_size
        )
        return self(H, W, patches, centers)

    def image(self, H: int, W: int, patch_size: Optional[int] = None) -> F:
        return ImgUtils.tensor2img(self.rasterize(H, W, patch_size=patch_size))

    @property
    def parameters(self) -> Dict[F]:
        raise NotImplementedError()

    @property
    def trainable_parameters(self) -> Dict[F]:
        raise NotImplementedError()
