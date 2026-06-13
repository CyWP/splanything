import torch

from jaxtyping import Float, Tensor
from typing import Optional, Generator, Tuple

from primitives import Primitive
from rasterizers import Rasterizer, WeightedRasterizer
from utils.img import ImgUtils


class TrainSampler:

    def __init__(
        self,
        primitive: Primitive,
        target: Float[Tensor, "B C H W"],
        patch_size: Optional[int] = None,
        sampling_map: Optional[Float[Tensor, "B 1 H W"]] = None,
        rasterizer: Optional[Rasterizer] = None,
    ):
        self.primitive = primitive
        self.sampling_map = sampling_map
        self.rasterizer = WeightedRasterizer() if rasterizer is None else rasterizer
        self.set_target(target, patch_size=patch_size)
        if sampling_map is not None:
            self.set_sampling_map(sampling_map)

    def set_target(
        self, target: Float[Tensor, "B C H W"], patch_size: Optional[int] = None
    ):
        device = target.device
        self.t_H, self.t_W = target.shape[-2:]
        self.patch_size = patch_size
        self.target_img = target
        self.target_patches = ImgUtils.extract_image_patches(
            target, patch_size, padding_mode="replicate"
        )
        self.co_patches, self.co_centers = ImgUtils.get_patches(
            self.t_H, self.t_W, device=device, patch_size=patch_size
        )
        if self.sampling_map is not None:
            self.set_sampling_map(self.sampling_map, patch_size=patch_size)

    def set_sampling_map(
        self, sampling_map: Float[Tensor, "B 1 H W"], patch_size: Optional[int] = None
    ):
        if not ImgUtils.same_size(sampling_map, self.target_img):
            sampling_map = ImgUtils.resize(*self.target_img.shape[-2:])
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

    def __iter__(self) -> Generator[Tuple[Float[Tensor, "S C"], Float[Tensor, "S C"]]]:
        p = self.primitive
        for i in range(self.num_patches):
            target = self.target_patches[i]
            co = self.co_patches[i]
            center = self.co_centers[i]
            if self.sampling_map is not None:
                sample_weights = self.sampling_patches[i]
                mask = torch.bernoulli(sample_weights).bool()
                target = target[mask]
                co = co[mask]
            if self.patch_size is None:
                yield p.sample(co, self.rasterizer), target
            else:
                with p.masked(
                    p.patch_mask(center[None, :], self.patch_size, self.t_H, self.t_W)
                ) as prim:
                    yield prim.sample(co, self.rasterizer), target
