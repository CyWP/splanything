from __future__ import annotations
from typing import List, Tuple, TYPE_CHECKING

import torch

from .base import SampleProcessor
from ..sample_output import SampleOutput
from ...utils.img import Splimage

if TYPE_CHECKING:
    from ...primitives.base import Primitive


class MultiSampleProcessor(SampleProcessor):
    def __init__(
        self,
        processors: List[Tuple[SampleProcessor, float | Splimage]],
        normalize_weights: bool = False,
    ):
        self._processors = [p for p, _ in processors]
        self._weights = [w for _, w in processors]
        self._normalize_weights = normalize_weights

    def process(
        self,
        sample: SampleOutput,
        primitive: Primitive,
    ) -> SampleOutput:
        rgb = torch.zeros_like(sample.rgb)
        weights = torch.zeros_like(sample.weights)
        if self._normalize_weights:
            cum_weights = torch.zeros(
                (sample.rgb.shape[0], 1),
                device=sample.rgb.device,
                dtype=sample.rgb.dtype,
            )
        for proc, w in zip(self._processors, self._weights):
            processed = proc(sample, primitive)
            if isinstance(w, Splimage):
                weight = w.mask_sample(sample.co).squeeze(0)
            else:
                weight = torch.tensor(
                    w, device=sample.rgb.device, dtype=sample.rgb.dtype
                )
            rgb += weight[..., None] * processed.rgb
            weights += weight * processed.weights
            if self._normalize_weights:
                cum_weights += weight
        if self._normalize_weights:
            rgb /= cum_weights[..., None]
            weights /= cum_weights
        return SampleOutput(rgb=rgb, weights=weights, co=sample.co)
