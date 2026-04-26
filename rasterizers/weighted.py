import torch

from jaxtyping import Float
from torch import Tensor

from .generic import Rasterizer
from .sample_out import SampleOutput

from utils.math import soft_clamp


class WeightedRasterizer(Rasterizer):

    def __init__(self, clamp_soft: float = 0.1):
        self.clamp_soft = clamp_soft

    def __call__(self, sample: SampleOutput, **kwargs) -> Float[Tensor, "N 4"]:
        # Normalize RGB by weight sum
        weight_sum = sample.weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        rgb = (sample.rgb * sample.weights[..., None]).sum(dim=1) / weight_sum
        rgb = rgb.clamp(0, 1)

        # Use soft_clamp for alpha with same logic as original
        a = (sample.weights * sample.alpha[None, :]).sum(dim=1).clamp(0, 1)
        return torch.cat([rgb, a[:, None]], dim=1)
