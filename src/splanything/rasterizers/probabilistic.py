import torch

from jaxtyping import Float
from torch import Tensor

from .generic import Rasterizer
from .sample_out import SampleOutput

from splanything.utils.math import soft_clamp


class ProbabilisticRasterizer(Rasterizer):
    """Probabilistic selection rasterizer.

    Selects one primitive per coordinate via weighted random sampling.
    RGB comes from selected primitive; alpha is soft-clamped weighted sum.

    Attributes:
        clamp_soft (float): Softness parameter for soft_clamp on alpha.

    Construction:
        ProbabilisticRasterizer(clamp_soft: float = 0.1):
            Create with specified softness for alpha clamping.

    Notes:
        - Uses torch.multinomial for selection (replacement=False).
        - Alpha uses soft_clamp instead of hard clamp.
    """

    def __init__(self, clamp_soft: float = 0.1):
        """Initialize rasterizer.

        Args:
            clamp_soft: Softness for alpha soft_clamp (higher = softer transition).
        """
        self.clamp_soft = clamp_soft

    def __call__(self, sample: SampleOutput, **kwargs) -> Float[Tensor, "N 4"]:
        """Aggregate via probabilistic selection.

        Args:
            sample: SampleOutput with rgb (Nc, N, 3), alpha (N,), weights (Nc, N).

        Returns:
            RGBA tensor (N, 4): RGB from sampled primitive, soft-clamped alpha.

        Shape:
            - selected: (Nc, 1) indices from multinomial
            - rgb: gather along N -> (Nc, 3)
            - alpha: soft_clamp(sum(w * a)) -> (N,)
        """
        # (Nc, N) -> (Nc, 1) via multinomial selection
        selected = torch.multinomial(sample.weights, 1)  # (Nc, 1)
        Nc = selected.shape[0]

        # Gather RGB using selected indices
        rgb_idx = selected.squeeze(-1)  # (Nc,)
        rgb = sample.rgb[torch.arange(Nc, device=sample.rgb.device), rgb_idx]  # (Nc, 3)

        # Soft-clamped alpha
        a = soft_clamp(
            (sample.weights * sample.alpha[None, :]).sum(dim=1),
            min_val=0.0,
            max_val=1.0,
            softness=self.clamp_soft,
        )  # (N,)
        return torch.cat([rgb, a[:, None]], dim=1)  # (N, 3) + (N, 1) -> (N, 4)
