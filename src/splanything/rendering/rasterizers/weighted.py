from typing import Optional

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
        top_k: If set, only the ``top_k`` strongest-weight primitives per
            coordinate contribute to the aggregation. The remaining
            primitives are dropped via index selection on dim=1 so the
            downstream weighted average and alpha operate only on the
            selected ``top_k`` rows — both RGB and alpha reflect only
            that set. When ``None`` (default), all primitives
            contribute — preserving the original behaviour.

    Notes:
        - Default rasterizer used when none specified.
        - Weights are normalized by their sum per coordinate.
        - RGB normalized independently of alpha.
        - With ``top_k`` active, no (Nc, Np) mask or zeroed-weights
          tensor is materialised; only (Nc, top_k) rows are kept.
    """

    def __init__(self, top_k: Optional[int] = None):
        """Initialize the rasterizer.

        Args:
            top_k: Number of strongest-weight primitives to keep per
                coordinate. ``None`` keeps the existing full-weight
                aggregation behaviour. Values >= the number of
                primitives are equivalent to ``None``.
        """
        if top_k is not None and top_k < 1:
            raise ValueError(f"top_k must be >= 1 or None, got {top_k}.")
        self.top_k = top_k

    def rasterize(self, sample: SampleOutput, **kwargs) -> Float[Tensor, "Nc 4"]:
        """Aggregate via weight-normalized weighted average.

        Args:
            sample: SampleOutput with rgb (Nc, N, 3), weights (Nc, N).

        Returns:
            RGBA tensor (N, 4): RGB normalized by weight sum, alpha = sum(w * a).
        """
        weights = sample.weights  # (Nc, Np)
        rgb = sample.rgb  # (Nc, Np, 3)

        if self.top_k is not None and self.top_k < weights.shape[1]:
            _, top_idx = torch.topk(weights, self.top_k, dim=1)  # (Nc, top_k)
            weights = weights.gather(1, top_idx)  # (Nc, top_k)
            rgb = rgb.gather(
                1, top_idx.unsqueeze(-1).expand(-1, -1, rgb.shape[-1])
            )  # (Nc, top_k, 3)

        # (Nc, K) -> (Nc, K, 1) for broadcasting
        weight_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)  # (Nc, 1)
        rgb_out = (rgb * weights[..., None]).sum(
            dim=1
        ) / weight_sum  # (Nc, K, 3) -> (Nc, 3)
        rgb_out = rgb_out.clamp(0, 1)

        a = weights.sum(dim=1).clamp(0, 1)  # (Nc, K) -> (Nc,)
        return torch.cat([rgb_out, a[:, None]], dim=1)  # (Nc, 3) + (Nc, 1) -> (Nc, 4)
