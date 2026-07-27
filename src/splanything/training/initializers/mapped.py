import torch
from torch import Tensor
from typing import Tuple
from jaxtyping import Float

from .base import Initializer
from ...utils.img import ImgUtils


class MappedInitializer(Initializer):
    def __init__(
        self,
        map: Float[Tensor, "B 1 H W"],
        coordinate_key: str = "centroids",
        initializer: Optional[Initializer] = None,
        apply_func: Optional[
            Callable[
                [str, Float[Tensor, "N"], Float[Tensor, "N ..."]],
                Float[Tensor, "N ..."],
            ]
        ] = None,
    ):
        self.map = map
        self.initializer = Initializer() if initializer is None else initializer
        self.coordinate_key = coordinate_key
        self._sampled_cache = None
        self.apply_func = apply_func
        self._feats_cache = {}

    def init_param(
        self, name: str, param_shape: Tuple[int], batched: bool
    ) -> Float[Tensor, "Size ..."]:
        if not batched:
            return self.initializer.init_param(name, param_shape, batched)
        if name == self.coordinate_key:
            assert param_shape == (param_shape[0], 2), (
                f"The coordinate key '{name}' must represent 2D Coordinates."
            )
            co = self.generate_coords(param_shape[0])
            self._sampled_cache = ImgUtils.uv_sample(self.map, co)
            self.process_feats_cache()
            return co
        feat = self.initializer.init_param(name, param_shape, batched)
        if self._sampled_cache is None:
            self._feats_cache[name] = feat
            return feat
        else:
            return self.apply_map(name, self._sampled_cache, feat)

    def generate_coords(self, N: int) -> Float[Tensor, "N 2"]:
        return ImgUtils.sample_px_coords(self.map, N, noise=True)

    def process_feats_cache(self):
        # Reference to previously intialized tensors are kept,
        # so we can update in place once coordinates are computed/sampled.
        sampled = self._sampled_cache
        with torch.no_grad():
            for name, feat in self._feats_cache.items():
                feat.copy_(self.apply_map(name, sampled, feat))
        self._feats_cache = {}

    def apply_map(
        self, name: str, sampled: Float[Tensor, "N"], feat: Float[Tensor, "N ..."]
    ) -> Float[Tensor, "N ..."]:
        if self.apply_func is None:
            return sampled * feat
        return self.apply_func(name, sampled, feat)
