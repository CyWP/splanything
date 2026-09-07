"""Container for primitive sampling results passed to rasterizers."""

from __future__ import annotations
from dataclasses import dataclass
import torch
from jaxtyping import Float
from torch import Tensor


@dataclass
class SampleOutput:
    """Container for per-coordinate per-primitive sampling data.

    Attributes:
        rgb: RGB values per coordinate per primitive (Nc, N, 3).
        weights: Weights per coordinate per primitive (Nc, N).
        co: Coordinates used to sample.
    """

    rgb: Float[Tensor, "Nc N 3"]
    weights: Float[Tensor, "Nc N"]
    co: Float[Tensor, "Nc 2"]

    def to(self, val: torch.device | torch.dtype) -> SampleOutput:
        """Move tensors to device/dtype.

        Args:
            val: Target device or dtype.

        Returns:
            New SampleOutput with moved tensors.
        """
        return SampleOutput(
            rgb=self.rgb.to(val), weights=self.weights.to(val), co=self.co.to(val)
        )

    @staticmethod
    def cat(*samples: SampleOutput) -> SampleOutput:
        """Concatenate multiple SampleOutputs along the primitive dimension.

        Args:
            samples: SampleOutputs to concatenate.

        Returns:
            Concatenated SampleOutput with increased N dimension.

        Notes:
            - Concatenates rgb and weights along dim=1 (primitive dimension).
            - Coordinates are taken from the first SampleOutput.
        """
        rgb = torch.cat([s.rgb for s in samples], dim=1)  # (Nc_sum, N, 3)
        weights = torch.cat([s.weights for s in samples], dim=1)  # (Nc_sum, N_sum)
        return SampleOutput(rgb, weights, samples[0].co)
