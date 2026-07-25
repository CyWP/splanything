from __future__ import annotations
from typing import Callable, TYPE_CHECKING

from .base import SampleProcessor
from ..sample_output import SampleOutput

if TYPE_CHECKING:
    from ...primitives.base import Primitive


class FlexibleSampleProcessor(SampleProcessor):
    def __init__(self, proc_func: Callable[[SampleOutput, Primitive], SampleOutput]):
        self.proc = proc_func

    def process(
        self,
        sample: SampleOutput,
        primitive: Primitive,
    ) -> SampleOutput:
        return self.proc(sample, primitive)
