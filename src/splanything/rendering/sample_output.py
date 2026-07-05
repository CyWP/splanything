from __future__ import annotations

from typing import Union

import torch
from jaxtyping import Float
from torch import Tensor


class SampleOutput:
    """Container for per-coordinate per-primitive sampling data.

    Attributes:
        rgb: RGB values per coordinate per primitive (Nc, N, 3).
        weights: Weights per coordinate per primitive (Nc, N).
        co: Coordinates used to sample.
    """

    def __init__(
        self,
        rgb: Float[Tensor, "Nc N 3"],
        weights: Float[Tensor, "Nc N"],
        co: Float[Tensor, "Nc 2"],
    ):
        self.rgb = rgb
        self.weights = weights
        self.co = co

    def to(self, val: Union[torch.device, torch.dtype]) -> SampleOutput:
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
        co = torch.cat([s.co for s in samples], dim=0)  # (Nc_sum, 2)
        return SampleOutput(rgb, weights, co)
