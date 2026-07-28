from __future__ import annotations
from torch import Tensor
from typing import Callable, Optional, TYPE_CHECKING
from jaxtyping import Float

from .base import SampleProcessor
from ..sample_output import SampleOutput

if TYPE_CHECKING:
    from ...primitives.base import Primitive


class DistanceSampleProcessor(SampleProcessor):
    def __init__(
        self,
        processor: SampleProcessor,
        proc_fn: Optional[
            Callable[[SampleOutput, Primitive, Float[Tensor, "Nc Np"]], SampleOutput]
        ] = None,
        ref_coords: Optional[Float[Tensor, "Nc 2"]] = None,
    ):
        self._processor = processor
        self._proc_fn = proc_fn
        self._ref_coords = ref_coords

    def process(
        self,
        sample: SampleOutput,
        primitive: Primitive,
    ) -> SampleOutput:
        if self._ref_coords is None:
            dists = (
                (primitive.centroids[None, :, :] - sample.co[:, None, :]) ** 2
            ).sum(dim=-1)
        else:
            dists = (
                (primitive.centroids[None, :, :] - self._ref_coords[:, None, :]) ** 2
            ).sum(dim=-1)
        return self.process_distances(
            self._processor(sample, primitive), primitive, dists
        )

    def process_distances(
        self,
        sample: SampleOutput,
        primitive: Primitive,
        distances: Float[Tensor, "Nc Np"],
    ) -> SampleOutput:
        if self._proc_fn is not None:
            return self._proc_fn(sample, primitive, distances)
        return SampleOutput(
            rgb=sample.rgb,
            weights=torch.exp(-distances.min(dim=0))[None, :] * sample.weights,
            co=sample.co,
        )
