import torch

from jaxtyping import Float
from torch import Tensor

from .generic import Rasterizer
from .sample_out import SampleOutput

from utils.math import soft_clamp


class ProbabilisticRasterizer(Rasterizer):

    def __init__(self, clamp_soft: float = 0.1):
        self.clamp_soft = clamp_soft

    def __call__(self, sample: SampleOutput, **kwargs) -> Float[Tensor, "N 4"]:
        # Select one primitive per coordinate based on weights
        # sample.weights: (Nc, N), sample.rgb: (Nc, N, 3)
        selected = torch.multinomial(sample.weights, 1)  # (Nc, 1)
        Nc = selected.shape[0]
        
        # Gather RGB values using advanced indexing
        # selected.view(-1) gives flat indices for batch dim
        # selected.squeeze(-1) gives indices for primitive dim
        rgb_idx = selected.squeeze(-1)  # (Nc,)
        rgb = sample.rgb[torch.arange(Nc, device=sample.rgb.device), rgb_idx]  # (Nc, 3)
        
        a = soft_clamp(
            (sample.weights * sample.alpha[None, :]).sum(dim=1),
            min_val=0.0,
            max_val=1.0,
            softness=self.clamp_soft,
        )
        return torch.cat([rgb, a[:, None]], dim=1)
