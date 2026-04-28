import torch

from jaxtyping import Float
from torch import Tensor

from .generic import Rasterizer
from .sample_out import SampleOutput


class InverseWeightedRasterizer(Rasterizer):
    """Inverse weight-normalized aggregation rasterizer.

    Uses (max_weight - weight) for RGB aggregation, promoting lower-weighted
    primitives. Alpha is computed as weight-summed alphas (same as WeightedRasterizer).

    Attributes:
        None.

    Notes:
        - Inverts weights relative to max before normalization.
        - Useful when high weights should contribute less to RGB.
    """

    def __call__(self, sample: SampleOutput, **kwargs) -> Float[Tensor, "N 4"]:
        """Aggregate via inverse-weight-normalized weighted average.

        Args:
            sample: SampleOutput with rgb (Nc, N, 3), alpha (N,), weights (Nc, N).

        Returns:
            RGBA tensor (N, 4): inverse-weighted RGB, alpha = sum(w * a).

        Shape:
            - weights: (Nc, N) -> (Nc, N) via max - weight
            - rgb: (Nc, N, 3) -> (N, 3) via inverse-weighted average
        """
        weight_sum = sample.weights.sum(dim=1, keepdim=True).clamp(min=1e-6)  # (Nc, 1)
        weights = sample.weights.max(dim=0).values[None, :] - sample.weights  # (Nc, N)
        rgb = (sample.rgb * weights[..., None]).sum(
            dim=1
        ) / weight_sum  # (Nc, N, 3) -> (N, 3)
        rgb = rgb.clamp(0, 1)

        a = (
            (sample.weights * sample.alpha[None, :]).sum(dim=1).clamp(0, 1)
        )  # (Nc, N) @ (N,) -> (N,)
        return torch.cat([rgb, a[:, None]], dim=1)  # (N, 3) + (N, 1) -> (N, 4)
