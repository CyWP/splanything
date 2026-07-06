from typing import Iterator, Optional, Tuple

import torch
from jaxtyping import Float
from torch import Tensor

from ..primitives import Primitive
from ..rendering.rasterizers import Rasterizer, WeightedRasterizer
from ..utils.img import ImgUtils
from .sampler import Sampler


class TrainSampler(Sampler):
    def __init__(
        self,
        target: Float[Tensor, "B C H W"],
        patch_size: Optional[int] = None,
        max_batch: Optional[int] = None,
        sampling_map: Optional[Float[Tensor, "B 1 H W"]] = None,
        rasterizer: Optional[Rasterizer] = None,
        low_vram: bool = False,
        epoch_size: Optional[int] = None,
    ):
        self.sampling_map = sampling_map
        self.max_batch = max_batch
        self.epoch_size = epoch_size
        self.low_vram = low_vram
        self.rasterizer = WeightedRasterizer() if rasterizer is None else rasterizer
        self.set_target(target, patch_size=patch_size)
        if sampling_map is not None:
            self.set_sampling_map(sampling_map)

    def set_target(
        self, target: Float[Tensor, "B C H W"], patch_size: Optional[int] = None
    ):
        device = target.device
        self.H, self.W = target.shape[-2:]
        self.patch_size = patch_size
        self.target_img = target
        self.target_patches = ImgUtils.extract_image_patches(
            target, patch_size, padding_mode="replicate"
        )
        self.co_patches, self.co_centers = ImgUtils.get_patches(
            self.H, self.W, device=device, patch_size=patch_size
        )
        if self.sampling_map is not None:
            self.set_sampling_map(self.sampling_map, patch_size=patch_size)

    def set_sampling_map(
        self,
        sampling_map: Float[Tensor, "B 1 H W"],
        patch_size: Optional[int] = None,
        epoch_size: Optional[int] = None,
    ):
        if not ImgUtils.same_size(sampling_map, self.target_img):
            sampling_map = ImgUtils.resize(sampling_map, self.H, self.W)
        e_size = self.epoch_size if epoch_size is None else epoch_size
        self.sampling_map = sampling_map
        self.sampling_patches = (
            ImgUtils.extract_image_patches(
                sampling_map, patch_size, padding_mode="constant"
            )
            .squeeze(-1)
            .squeeze(0)
        )
        if e_size is not None:
            self.sampling_patches = (
                self.sampling_patches / self.sampling_patches.sum() * e_size
            )

    def set_patch_size(self, patch_size: int):
        if self.target_img is not None:
            self.set_target(self.target_img, patch_size)
        else:
            self.patch_size = patch_size

    @property
    def num_patches(self) -> int:
        return self.target_patches.shape[0]

    def samples(
        self, p: Primitive
    ) -> Iterator[
        Tuple[Float[Tensor, "S C"], Float[Tensor, "S C"], Float[Tensor, "S 2"]]
    ]:
        sample_patches = []
        targets = []
        with torch.no_grad():
            for i in range(self.co_patches.shape[0]):
                mask = torch.bernoulli(self.sampling_patches[i]).bool()
                targets.append(self.target_patches[i][mask])
                sample_patches.append(self.co_patches[i][mask])
        tmp_co_patches = self.co_patches
        self.co_patches = sample_patches
        targets = torch.cat(targets, dim=0)
        t_i = 0
        for sample, batch_co in super().samples(p):
            s_l = sample.shape[0]
            target = targets[t_i : t_i + s_l]
            t_i += s_l
            yield sample, target, batch_co
        self.co_patches = tmp_co_patches
