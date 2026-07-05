import torch
from jaxtyping import Float
from torch import Tensor

from ..sample_output import SampleOutput
from .base import Rasterizer


class WeightedRasterizer(Rasterizer):
    """Weight-normalized aggregation rasterizer.

    Aggregates per-primitive RGB values by weight-normalized weighted average.
    Alpha is computed as weight-summed alphas.

    Attributes:
        None.

    Notes:
        - Default rasterizer used when none specified.
        - Weights are normalized by their sum per coordinate.
        - RGB normalized independently of alpha.
    """

    def rasterize(self, sample: SampleOutput, **kwargs) -> Float[Tensor, "N 4"]:
        """Aggregate via weight-normalized weighted average.

        Args:
            sample: SampleOutput with rgb (Nc, N, 3), weights (Nc, N).

        Returns:
            RGBA tensor (N, 4): RGB normalized by weight sum, alpha = sum(w * a).
        """
        # (Nc, N) -> (Nc, N, 1) for broadcasting
        weight_sum = sample.weights.sum(dim=1, keepdim=True).clamp(min=1e-6)  # (Nc, 1)
        rgb = (sample.rgb * sample.weights[..., None]).sum(
            dim=1
        ) / weight_sum  # (Nc, N, 3) -> (N, 3)
        rgb = rgb.clamp(0, 1)

        a = sample.weights.sum(dim=1).clamp(0, 1)  # (Nc, N) -> (N,)
        return torch.cat([rgb, a[:, None]], dim=1)  # (N, 3) + (N, 1) -> (N, 4)
