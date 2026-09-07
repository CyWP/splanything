"""Centroid-distance-based weight modulation processor."""

from __future__ import annotations
from torch import Tensor
from typing import Callable, Optional, TYPE_CHECKING
from jaxtyping import Float

from .base import SampleProcessor
from ..sample_output import SampleOutput

if TYPE_CHECKING:
    from ...primitives.base import Primitive


class DistanceSampleProcessor(SampleProcessor):
    """Modulates weights by squared distances to primitive centroids.

    Computes squared centroid-to-coordinate distances (optionally against
    fixed reference coordinates) and delegates to ``proc_fn``, or falls
    back to a Gaussian falloff on the minimum squared distance.
    """

    def __init__(
        self,
        processor: SampleProcessor,
        proc_fn: Optional[
            Callable[[SampleOutput, Primitive, Float[Tensor, "Nc Np"]], SampleOutput]
        ] = None,
        ref_coords: Optional[Float[Tensor, "Nc 2"]] = None,
    ):
        """Initialize the processor.

        Args:
            processor: Upstream processor applied before this one.
            proc_fn: Optional custom (sample, primitive, distances) ->
                SampleOutput; defaults to Gaussian falloff on the minimum
                squared distance.
            ref_coords: Optional fixed reference coordinates (Nc, 2);
                defaults to the sample's own coordinates.
        """
        self._processor = processor
        self._proc_fn = proc_fn
        self._ref_coords = ref_coords

    def process(
        self,
        sample: SampleOutput,
        primitive: Primitive,
    ) -> SampleOutput:
        """Apply the upstream processor, then modulate weights by distances.

        Args:
            sample: SampleOutput to transform.
            primitive: Primitive the sample was generated from.

        Returns:
            Transformed SampleOutput.
        """
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
        """Modulate weights with precomputed squared distances.

        Args:
            sample: SampleOutput to transform.
            primitive: Primitive the sample was generated from.
            distances: Squared centroid-to-coordinate distances (Nc, Np).

        Returns:
            Transformed SampleOutput; by default weights are scaled by
            exp(-min_c(distances)) per primitive, broadcast (1, Np).
        """
        if self._proc_fn is not None:
            return self._proc_fn(sample, primitive, distances)
        return SampleOutput(
            rgb=sample.rgb,
            weights=torch.exp(-distances.min(dim=0))[None, :] * sample.weights,
            co=sample.co,
        )
