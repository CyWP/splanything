"""Splimage-mask-modulated weight processor."""

from __future__ import annotations
from torch import Tensor
from typing import Callable, Optional
from jaxtyping import Float

from .base import SampleProcessor, SampleOutput
from ...utils.img import Splimage


class MappedSampleProcessor(SampleProcessor):
    """Scales weights by per-primitive lookups of a Splimage map.

    Samples a Splimage map at each primitive's coordinates (attribute
    selectable) and multiplies the sample weights by the sampled values.
    """

    def __init__(
        self,
        processor: SampleProcessor,
        proc_map: Splimage,
        proc_fn: Optional[
            Callable[[SampleOutput, Primitive, Float[Tensor, "Np"]], SampleOutput]
        ] = None,
        coords_attr: str = "adjusted_coords",
    ):
        """Initialize the processor.

        Args:
            processor: Upstream processor applied before this one.
            proc_map: Splimage map sampled at the primitive coordinates.
            proc_fn: Optional custom (sample, primitive, sampled_vals) ->
                SampleOutput; defaults to multiplying weights by the
                sampled values.
            coords_attr: Primitive attribute providing coordinates (N, 2)
                for sampling ``proc_map``.
        """
        self._processor = processor
        self._map = proc_map
        self._proc_fn = proc_fn
        self._coords_attr = coords_attr

    def process(
        self,
        sample: SampleOutput,
        primitive: Primitive,
    ) -> SampleOutput:
        """Apply the upstream processor, then scale weights by the map lookup.

        Args:
            sample: SampleOutput to transform.
            primitive: Primitive the sample was generated from.

        Returns:
            Transformed SampleOutput.
        """
        sampled = self._map.mask_sample(getattr(primitive, self._coords_attr)).squeeze(
            -1
        )
        return self.process_map(self._processor(sample, primitive), primitive, sampled)

    def process_map(
        self,
        sample: SampleOutput,
        primitive: Primitive,
        sampled_vals: Float[Tensor, "Np"],
    ) -> SampleOutput:
        """Scale weights by per-primitive sampled map values.

        Args:
            sample: SampleOutput to transform.
            primitive: Primitive the sample was generated from.
            sampled_vals: Per-primitive map samples (Np,).

        Returns:
            Transformed SampleOutput with weights scaled per primitive.
        """
        if self._proc_fn is not None:
            return self._proc_fn(sample, primitive, distances)
        return SampleOutput(
            rgb=sample.rgb, weights=sample.weights * sampled_vals[:, None], co=sample.co
        )
