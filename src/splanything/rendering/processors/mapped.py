from __future__ import annotations
from torch import Tensor
from typing import Callable, Optional
from jaxtyping import Float

from .base import SampleProcessor, SampleOutput
from ...utils.img import ImgUtils


class MappedSampleProcessor(SampleProcessor):
    def __init__(
        self,
        processor: SampleProcessor,
        proc_map: Float[Tensor, "B 1 H W"],
        proc_fn: Optional[
            Callable[[SampleOutput, Primitive, Float[Tensor, "Np"]], SampleOutput]
        ] = None,
    ):
        self._processor = processor
        self._map = proc_map
        self._proc_fn = proc_fn

    def process(
        self,
        sample: SampleOutput,
        primitive: Primitive,
    ) -> SampleOutput:
        sampled = ImgUtils.uv_sample(self._map, primitive.centroids).squeeze(-1)
        return self.process_map(self._processor(sample, primitive), primitive, sampled)

    def process_map(
        self,
        sample: SampleOutput,
        primitive: Primitive,
        sampled_vals: Float[Tensor, "Np"],
    ) -> SampleOutput:
        if self._proc_fn is not None:
            return self._proc_fn(sample, primitive, distances)
        return SampleOutput(
            rgb=sample.rgb, weights=sample.weights * sampled_vals[:, None], co=sample.co
        )
