from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from jaxtyping import Float
from torch import Tensor

from ..sample_output import SampleOutput


class SampleProcessor(ABC):
    """
    Callback function for processing SampleOutputs before rasterization.
    """

    @abstractmethod
    def process(self, sample: SampleOutput, **kwargs) -> SampleOutput:
        pass

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)


class Rasterizer(ABC):
    """Abstract base class for rasterizing sample outputs to RGBA images.

    Transforms per-coordinate per-primitive sampling data into final
    RGBA output by aggregating RGB, alpha, and weight information.

    Attributes:
        None.

    Notes:
        - Implement rasterize to define custom aggregation strategy.
        - All rasterizers produce output in (N, 4) format: RGB + alpha.
        - Processor functions must both accept as an argument and return a SampleOutput instance.
    """

    def __init__(self, processors: Optional[List[SampleProcessor]] = None):
        self.processors = [] if processors is None else processors

    def add_processor(self, processor: SampleProcessor):
        self.processors.append(processor)

    def __call__(
        self,
        sample: SampleOutput,
        processors: Optional[List[Callable]] = None,
        **kwargs,
    ) -> Float[Tensor, "N 4"]:
        for proc in self.processors:
            sample = proc(sample, **kwargs)
        if processors is not None:
            for proc in processors:
                sample = proc(sample, **kwargs)
        return self.rasterize(sample, **kwargs)

    @abstractmethod
    def rasterize(self, sample: SampleOutput, **kwargs) -> Float[Tensor, "N 4"]:
        """Aggregate sample data to RGBA output.

        Args:
            sample: SampleOutput with rgb (Nc, N, 3), alpha (N,), weights (Nc, N).
            **kwargs: Additional rasterizer-specific arguments.

        Returns:
            RGBA tensor (N, 4) in [0, 1] range.
        """
        pass
