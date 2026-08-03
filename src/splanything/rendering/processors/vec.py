from __future__ import annotations
import torch
from torch import Tensor
from typing import Callable, Optional, TYPE_CHECKING
from jaxtyping import Float

from .base import SampleProcessor
from ..sample_output import SampleOutput

if TYPE_CHECKING:
    from ...primitives.base import Primitive


class VecSampleProcessor(SampleProcessor):
    def __init__(
        self,
        processor: SampleProcessor,
        proc_fn: Optional[
            Callable[
                [SampleOutput, Primitive, Float[Tensor, "Nc Np"], Float[Tensor, "Nc Np"]],
                SampleOutput,
            ]
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
            diff = primitive.centroids[None, :, :] - sample.co[:, None, :]
        else:
            diff = primitive.centroids[None, :, :] - self._ref_coords[:, None, :]
        dx = diff[..., 0]
        dy = diff[..., 1]
        return self.process_vec(
            self._processor(sample, primitive), primitive, dx, dy
        )

    def process_vec(
        self,
        sample: SampleOutput,
        primitive: Primitive,
        dx: Float[Tensor, "Nc Np"],
        dy: Float[Tensor, "Nc Np"],
    ) -> SampleOutput:
        if self._proc_fn is not None:
            return self._proc_fn(sample, primitive, dx, dy)
        dists = dx ** 2 + dy ** 2
        return SampleOutput(
            rgb=sample.rgb,
            weights=torch.exp(-dists.min(dim=0).values)[None, :] * sample.weights,
            co=sample.co,
        )
