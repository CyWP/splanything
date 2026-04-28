import torch

from jaxtyping import Float
from torch import Tensor

from .generic import Rasterizer
from .sample_out import SampleOutput


class MaxWeightedRasterizer(Rasterizer):
    """Maximum weight selection rasterizer.

    Selects RGB from the primitive with highest weight for each coordinate.
    Alpha is computed as weight-summed alphas (same as WeightedRasterizer).

    Attributes:
        None.

    Notes:
        - Deterministic selection based on max weight.
        - Different from ProbabilisticRasterizer which uses random sampling.
    """

    def __call__(self, sample: SampleOutput, **kwargs) -> Float[Tensor, "N 4"]:
        """Aggregate via maximum weight selection.

        Args:
            sample: SampleOutput with rgb (Nc, N, 3), alpha (N,), weights (Nc, N).

        Returns:
            RGBA tensor (N, 4): RGB from max-weighted primitive, alpha = sum(w * a).

        Shape:
            - idx: (Nc,) indices of max weight per coordinate
            - rgb: gather along N -> (N, 3)
            - alpha: same as WeightedRasterizer
        """
        Nc, Np, C = sample.rgb.shape
        idx = sample.weights.max(dim=1).indices  # (Nc,)
        rgb = torch.gather(sample.rgb, 0, idx[:, None, None].expand(Nc, 1, C)).squeeze(
            1
        )  # (Nc, N, 3) -> (Nc, 3)
        rgb = rgb.clamp(0, 1)

        a = (
            (sample.weights * sample.alpha[None, :]).sum(dim=1).clamp(0, 1)
        )  # (Nc, N) @ (N,) -> (N,)
        return torch.cat([rgb, a[:, None]], dim=1)  # (N, 3) + (N, 1) -> (N, 4)
