import torch
from torch import Tensor
from typing import List, Tuple
from jaxtyping import Float
from .base import Rasterizer
from ..sample_output import SampleOutput
from ...utils.img import ImgUtils


class MultiRasterizer(Rasterizer):
    def __init__(
        self,
        rasterizers: List[Tuple[Rasterizer, float | Float[Tensor, "B 1 H W"]]],
        normalize_weights: bool = False,
    ):
        self._rasterizers = [r for r, _ in rasterizers]
        self._weights = [w for _, w in rasterizers]
        self._normalize_weights = normalize_weights

    def rasterize(self, sample: SampleOutput, **kwargs) -> Float[Tensor, "Nc 4"]:
        Nc = sample.co.shape[0]
        out = torch.zeros((Nc, 4), device=sample.rgb.device, dtype=sample.rgb.dtype)
        if self._normalize_weights:
            cum_weights = torch.zeros(
                (Nc, 1), device=sample.rgb.device, dtype=sample.rgb.dtype
            )
        for r, w in zip(self._rasterizers, self._weights):
            rasterized = r(sample)
            if isinstance(w, torch.Tensor) and w.ndim == 4:
                weight = ImgUtils.uv_sample(w, sample.co)
            else:
                weight = w
            out += (weight * rasterized)[0]
            if self._normalize_weights:
                cum_weights += weight
        if self._normalize_weights:
            out /= cum_weights
        return out
