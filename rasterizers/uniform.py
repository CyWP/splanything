import torch

from jaxtyping import Float
from torch import Tensor

from .generic import Rasterizer
from .sample_out import SampleOutput


class UniformRasterizer(Rasterizer):
    """Uniform aggregation rasterizer.

    Averages RGB across all primitives equally. Alpha is computed as
    weight-summed alphas (same as WeightedRasterizer).

    Attributes:
        None.

    Notes:
        - Ignores weight magnitudes, treats all primitives equally.
        - Different from WeightedRasterizer which normalizes by weight sum.
    """

    def __call__(self, sample: SampleOutput, **kwargs) -> Float[Tensor, "N 4"]:
        """Aggregate via uniform averaging.

        Args:
            sample: SampleOutput with rgb (Nc, N, 3), alpha (N,), weights (Nc, N).

        Returns:
            RGBA tensor (N, 4): uniformly averaged RGB, alpha = sum(w * a).

        Shape:
            - rgb: (Nc, N, 3) -> (N, 3) via mean across N
            - alpha: same as WeightedRasterizer
        """
        rgb = sample.rgb.mean(dim=1)  # (Nc, N, 3) -> (N, 3)
        rgb = rgb.clamp(0, 1)

        a = (
            (sample.weights * sample.alpha[None, :]).sum(dim=1).clamp(0, 1)
        )  # (Nc, N) @ (N,) -> (N,)
        return torch.cat([rgb, a[:, None]], dim=1)  # (N, 3) + (N, 1) -> (N, 4)
