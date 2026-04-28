import torch

from jaxtyping import Float
from torch import Tensor

from .generic import Rasterizer
from .sample_out import SampleOutput


class ExponentialWeightedRasterizer(Rasterizer):
    """Exponential weight-based aggregation rasterizer.

    Raises weights to a power before normalization. Higher exponents
    create sharper distinctions between high/low weighted primitives.

    Attributes:
        exponent (float): Power to raise weights to (default 2.0).

    Construction:
        ExponentialWeightedRasterizer(exponent: float = 2.0):
            Create with specified exponent.

    Notes:
        - exponent=1.0 is equivalent to WeightedRasterizer.
        - exponent=0.0 is equivalent to UniformRasterizer.
        - Higher exponent = more aggressive weight concentration.
    """

    def __init__(self, exponent: float = 2.0):
        """Initialize rasterizer.

        Args:
            exponent: Power to raise weights to (must be positive).
        """
        if exponent <= 0:
            raise ValueError(f"Exponent for rasterizer must be positive.")
        self.exponent = exponent

    def __call__(self, sample: SampleOutput, **kwargs) -> Float[Tensor, "N 4"]:
        """Aggregate via exponentially-powered weights.

        Args:
            sample: SampleOutput with rgb (Nc, N, 3), alpha (N,), weights (Nc, N).

        Returns:
            RGBA tensor (N, 4): exponent-weighted RGB, alpha = sum(w * a).

        Shape:
            - weights: (Nc, N) -> (Nc, N) via **exponent
            - rgb: (Nc, N, 3) -> (N, 3) via exponent-weighted average
        """
        weights = sample.weights**self.exponent  # (Nc, N)
        weight_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)  # (Nc, 1)
        rgb = (sample.rgb * weights[..., None]).sum(
            dim=1
        ) / weight_sum  # (Nc, N, 3) -> (N, 3)
        rgb = rgb.clamp(0, 1)

        a = (
            (sample.weights * sample.alpha[None, :]).sum(dim=1).clamp(0, 1)
        )  # (Nc, N) @ (N,) -> (N,)
        return torch.cat([rgb, a[:, None]], dim=1)  # (N, 3) + (N, 1) -> (N, 4)
