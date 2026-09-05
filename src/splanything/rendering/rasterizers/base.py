"""Abstract base interface for SampleOutput -> RGBA aggregation."""

from abc import ABC, abstractmethod

from jaxtyping import Float
from torch import Tensor

from ..sample_output import SampleOutput


class Rasterizer(ABC):
    """Abstract base class for rasterizing sample outputs to RGBA images.

    Transforms per-coordinate per-primitive sampling data into final
    RGBA output by aggregating RGB, alpha, and weight information.

    Attributes:
        None.

    Notes:
        - Implement rasterize to define custom aggregation strategy.
        - All rasterizers produce output in (Nc, 4) format: RGB + alpha.
        - Processor functions must both accept as an argument and return a SampleOutput instance.
    """

    def __call__(
        self,
        sample: SampleOutput,
        **kwargs,
    ) -> Float[Tensor, "Nc 4"]:
        """Aggregate sample data to RGBA via ``rasterize``.

        Args:
            sample: SampleOutput to aggregate.
            **kwargs: Forwarded to ``rasterize``.

        Returns:
            RGBA tensor (Nc, 4).
        """
        return self.rasterize(sample, **kwargs)

    @abstractmethod
    def rasterize(self, sample: SampleOutput, **kwargs) -> Float[Tensor, "Nc 4"]:
        """Aggregate sample data to RGBA output.

        Args:
            sample: SampleOutput with rgb (Nc, Np, 3), weights (Nc, Np),
                co (Nc, 2).
            **kwargs: Additional rasterizer-specific arguments.

        Returns:
            RGBA tensor (Nc, 4) in [0, 1] range.
        """
        pass
