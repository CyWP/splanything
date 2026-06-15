import torch

from jaxtyping import Float
from torch import Tensor
from typing import Optional, Iterator, Tuple

from splanything.primitives import Primitive
from splanything.rasterizers import Rasterizer, WeightedRasterizer
from splanything.utils.img import ImgUtils
from splanything.vars import LOW_VRAM

from .sampler import Sampler


class TrainSampler(Sampler):
    def __init__(
        self,
        target: Float[Tensor, "B C H W"],
        patch_size: Optional[int] = None,
        max_batch: Optional[int] = None,
        sampling_map: Optional[Float[Tensor, "B 1 H W"]] = None,
        rasterizer: Optional[Rasterizer] = None,
    ):
        self.sampling_map = sampling_map
        self.max_batch = max_batch
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
        self, sampling_map: Float[Tensor, "B 1 H W"], patch_size: Optional[int] = None
    ):
        if not ImgUtils.same_size(sampling_map, self.target_img):
            sampling_map = ImgUtils.resize(sampling_map, self.H, self.W)
        self.sampling_map = sampling_map
        self.sampling_patches = (
            ImgUtils.extract_image_patches(
                sampling_map, patch_size, padding_mode="constant"
            )
            .squeeze(-1)
            .squeeze(0)
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
    ) -> Iterator[Tuple[Float[Tensor, "S C"], Float[Tensor, "S C"]]]:
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
        for sample in super().samples(p):
            s_l = sample.shape[0]
            target = targets[t_i : t_i + s_l]
            t_i += s_l
            yield sample, target
        self.co_patches = tmp_co_patches

    def rasterize(self, p, max_batch=None, low_vram=None):
        low_vram = LOW_VRAM if low_vram is None else low_vram
        P, S, C = self.co.patches.shape
        gen = []
        for patch in super().samples(p, max_batch):
            if low_vram:
                patch = patch.cpu()
            gen.append(patch)
        patch_gen = torch.cat(gen, dim=0).reshape(P, S, 4)
        return ImgUtils.assemble_patches(patch_gen, self.H, self.W)
