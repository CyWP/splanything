from __future__ import annotations
import torch

from jaxtyping import Float
from torch import Tensor
from typing import Union, Sequence


class SampleOutput:
    """Container for per-coordinate per-primitive sampling data.

    Attributes:
        rgb: RGB values per coordinate per primitive (Nc, N, 3).
        alpha: Alpha values per primitive (N,).
        weights: Weights per coordinate per primitive (Nc, N).
    """

    def __init__(
        self,
        rgb: Float[Tensor, "Nc N 3"],
        alpha: Float[Tensor, "N"],
        weights: Float[Tensor, "Nc N"],
    ):
        self.rgb = rgb
        self.alpha = alpha
        self.weights = weights

    def to(self, val: Union[torch.device, torch.dtype]) -> SampleOutput:
        """Move tensors to device/dtype."""
        return SampleOutput(
            rgb=self.rgb.to(val),
            alpha=self.alpha.to(val),
            weights=self.weights.to(val),
        )

    @staticmethod
    def cat(samples: Sequence[SampleOutput]) -> SampleOutput:
        """Concatenate multiple SampleOutputs along the coordinate dimension."""
        rgb = torch.cat([s.rgb for s in samples], dim=0)
        alpha = torch.cat([s.alpha for s in samples], dim=0)
        weights = torch.cat([s.weights for s in samples], dim=0)
        return SampleOutput(rgb, alpha, weights)