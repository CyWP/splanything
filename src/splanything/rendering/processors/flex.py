"""Callable-based sample processor."""

from __future__ import annotations
from typing import Callable, TYPE_CHECKING

from .base import SampleProcessor
from ..sample_output import SampleOutput

if TYPE_CHECKING:
    from ...primitives.base import Primitive


class FlexibleSampleProcessor(SampleProcessor):
    """Sample processor wrapping a plain (sample, primitive) callable."""

    def __init__(self, proc_func: Callable[[SampleOutput, Primitive], SampleOutput]):
        """Initialize the processor.

        Args:
            proc_func: Callable (sample, primitive) -> SampleOutput.
        """
        self.proc = proc_func

    def process(
        self,
        sample: SampleOutput,
        primitive: Primitive,
    ) -> SampleOutput:
        """Apply the wrapped callable.

        Args:
            sample: SampleOutput to transform.
            primitive: Primitive the sample was generated from.

        Returns:
            Transformed SampleOutput.
        """
        return self.proc(sample, primitive)
