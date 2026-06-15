from abc import ABC, abstractmethod
from jaxtyping import Float
from torch import Tensor

from .sample_out import SampleOutput


class Rasterizer(ABC):
    """Abstract base class for rasterizing sample outputs to RGBA images.

    Transforms per-coordinate per-primitive sampling data into final
    RGBA output by aggregating RGB, alpha, and weight information.

    Attributes:
        None.

    Notes:
        - Implement `__call__` to define custom aggregation strategy.
        - All rasterizers produce output in (N, 4) format: RGB + alpha.
    """

    @abstractmethod
    def __call__(self, sample: SampleOutput, **kwargs) -> Float[Tensor, "N 4"]:
        """Aggregate sample data to RGBA output.

        Args:
            sample: SampleOutput with rgb (Nc, N, 3), alpha (N,), weights (Nc, N).
            **kwargs: Additional rasterizer-specific arguments.

        Returns:
            RGBA tensor (N, 4) in [0, 1] range.
        """
        pass
