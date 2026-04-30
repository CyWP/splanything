import torch

from jaxtyping import Float
from torch import Tensor

from .generic import Rasterizer
from .sample_out import SampleOutput

from utils.math import soft_clamp


class InverseExponentialProbabilisticRasterizer(Rasterizer):
    """Inverse probabilistic selection rasterizer.

    Selects one primitive per coordinate via inverse-weight random sampling.
    RGB comes from selected primitive; alpha is soft-clamped weighted sum.

    Attributes:
        clamp_soft (float): Softness parameter for soft_clamp on alpha
            (inherited from ProbabilisticRasterizer).

    Construction:
        InverseProbabilisticRasterizer(clamp_soft: float = 0.1):
            Create with specified softness for alpha clamping.

    Notes:
        - Uses (max_weight - weight) for selection probabilities.
        - Same soft_clamp behavior as ProbabilisticRasterizer.
    """

    def __init__(self, exponent: float = 2):
        """Initialize rasterizer.

        Args:
            clamp_soft: Softness for alpha soft_clamp (higher = softer transition).
        """
        self.exponent = exponent

    def __call__(self, sample: SampleOutput, **kwargs) -> Float[Tensor, "N 4"]:
        """Aggregate via inverse-probabilistic selection.

        Args:
            sample: SampleOutput with rgb (Nc, N, 3), alpha (N,), weights (Nc, N).

        Returns:
            RGBA tensor (N, 4): RGB from inverse-sampled primitive, soft-clamped alpha.

        Shape:
            - weights: (Nc, N) -> (Nc, N) via max - weight
            - selected: (Nc, 1) indices from inverse-weight multinomial
            - rgb: gather along N -> (Nc, 3)
        """
        # Compute inverse weights and sample
        weights = sample.weights**self.exponent
        weights = weights.max(dim=0).values[None, :] - weights  # (Nc, N)
        selected = torch.multinomial(weights, 1)  # (Nc, 1)
        Nc = selected.shape[0]

        # Gather RGB using selected indices
        rgb_idx = selected.squeeze(-1)  # (Nc,)
        rgb = sample.rgb[torch.arange(Nc, device=sample.rgb.device), rgb_idx]  # (Nc, 3)

        # Soft-clamped alpha
        a = (sample.weights * sample.alpha[None, :]).sum(dim=1).clamp(0, 1)
        return torch.cat([rgb, a[:, None]], dim=1)  # (N, 3) + (N, 1) -> (N, 4)
