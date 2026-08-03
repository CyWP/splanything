from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from ..sample_output import SampleOutput

if TYPE_CHECKING:
    from ...primitives.base import Primitive


class SampleProcessor(ABC):
    """
    Callback function for processing SampleOutputs before rasterization.
    """

    @abstractmethod
    def process(
        self,
        sample: SampleOutput,
        primitive: Primitive,
    ) -> SampleOutput:
        pass

    def __call__(
        self,
        sample: SampleOutput,
        primitive: Primitive,
    ) -> SampleOutput:
        return self.process(sample, primitive)
