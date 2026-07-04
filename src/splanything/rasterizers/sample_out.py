from __future__ import annotations
import torch

from jaxtyping import Float
from torch import Tensor
from typing import Union


class SampleOutput:
    """Container for per-coordinate per-primitive sampling data.

    Attributes:
        rgb: RGB values per coordinate per primitive (Nc, N, 3).
        weights: Weights per coordinate per primitive (Nc, N).
    """

    def __init__(
        self,
        rgb: Float[Tensor, "Nc N 3"],
        weights: Float[Tensor, "Nc N"],
    ):
        self.rgb = rgb
        self.weights = weights

    def to(self, val: Union[torch.device, torch.dtype]) -> SampleOutput:
        """Move tensors to device/dtype.

        Args:
            val: Target device or dtype.

        Returns:
            New SampleOutput with moved tensors.
        """
        return SampleOutput(
            rgb=self.rgb.to(val),
            weights=self.weights.to(val),
        )

    @staticmethod
    def cat(*samples: SampleOutput) -> SampleOutput:
        """Concatenate multiple SampleOutputs along the coordinate dimension.

        Args:
            samples: SampleOutputs to concatenate.

        Returns:
            Concatenated SampleOutput with increased Nc dimension.

        Notes:
            - Concatenates along dim=0 (coordinate dimension).
            - All other dimensions (N, 3) must match.
        """
        rgb = torch.cat([s.rgb for s in samples], dim=0)  # (Nc_sum, N, 3)
        weights = torch.cat([s.weights for s in samples], dim=0)  # (Nc_sum, N_sum)
        return SampleOutput(rgb, weights)
