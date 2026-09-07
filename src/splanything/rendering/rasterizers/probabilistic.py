"""Monte Carlo weighted-sampling aggregation rasterizer."""

import torch
from jaxtyping import Float
from torch import Tensor

from ..sample_output import SampleOutput
from .base import Rasterizer


class ProbabilisticRasterizer(Rasterizer):
    """Probabilistic weighted aggregation rasterizer.

    Samples primitives according to their weights and averages their colors.

    Args:
        top_k: Number of Monte Carlo samples per coordinate.

    Notes:
        - Uses torch.multinomial with replacement.
        - Approaches WeightedRasterizer as top_k increases.
        - Assumes weights already encode the desired selection distribution.
    """

    def __init__(self, top_k: int = 1, alpha_as_dropout: bool = True):
        """Initialize the rasterizer.

        Args:
            top_k: Number of Monte Carlo samples per coordinate.
            alpha_as_dropout: If True, binarize alpha by random thresholding.
        """
        self.top_k = top_k
        self._alpha_as_dropout = alpha_as_dropout

    def rasterize(self, sample: SampleOutput, **kwargs) -> Float[Tensor, "Nc 4"]:
        """Aggregate via Monte Carlo sampling of primitives per coordinate.

        Args:
            sample: SampleOutput with rgb (Nc, Np, 3), weights (Nc, Np).

        Returns:
            RGBA tensor (Nc, 4): mean of sampled RGBs per coordinate,
            alpha = clamped weight sum.

        Notes:
            - Coordinates with zero total weight keep RGB 0.
            - Sampling probabilities are the normalized (clamped) weights.
        """
        w = sample.weights.clamp(min=0)
        Nc, Np = w.shape
        valid = w.sum(dim=1) > 0
        rgb = torch.zeros(
            Nc,
            3,
            device=sample.rgb.device,
            dtype=sample.rgb.dtype,
        )
        if valid.any():
            w_valid = w[valid]
            # Convert weights into sampling probabilities
            probs = w_valid / w_valid.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1e-8)
            k = min(self.top_k, Np)
            # Sample primitives according to their weights
            selected = torch.multinomial(
                probs,
                k,
                replacement=True,
            )  # (Nv, k)
            # Gather sampled colors
            rgb_selected = torch.gather(
                sample.rgb[valid],
                1,
                selected[..., None].expand(-1, -1, 3),
            )  # (Nv, k, 3)
            # Monte Carlo estimate of expected color
            rgb[valid] = rgb_selected.mean(dim=1).clamp(0, 1)
        alpha = w.sum(dim=1).clamp(0, 1)
        if self._alpha_as_dropout:
            alpha = (alpha >= torch.rand_like(alpha)).to(alpha.dtype)
        return torch.cat([rgb, alpha[:, None]], dim=1)
