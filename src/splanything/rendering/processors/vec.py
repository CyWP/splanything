"""Axis-distance-based weight modulation processor."""

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
    """Modulates weights using per-axis distances to primitive centroids.

    Computes centroid-to-coordinate difference vectors (optionally against
    fixed reference coordinates) and delegates to ``proc_fn`` with their
    dx/dy components, or falls back to a Gaussian falloff on the minimum
    squared axis distance.
    """

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
        """Initialize the processor.

        Args:
            processor: Upstream processor applied before this one.
            proc_fn: Optional custom (sample, primitive, dx, dy) ->
                SampleOutput; defaults to Gaussian falloff on the minimum
                squared axis distance.
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
        """Apply the upstream processor, then modulate weights by axis distances.

        Args:
            sample: SampleOutput to transform.
            primitive: Primitive the sample was generated from.

        Returns:
            Transformed SampleOutput.
        """
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
        """Modulate weights with precomputed dx/dy components.

        Args:
            sample: SampleOutput to transform.
            primitive: Primitive the sample was generated from.
            dx: Per-primitive x-distance components (Nc, Np).
            dy: Per-primitive y-distance components (Nc, Np).

        Returns:
            Transformed SampleOutput; by default weights are scaled by
            exp(-min_c(dx^2 + dy^2)) per primitive, broadcast (1, Np).
        """
        if self._proc_fn is not None:
            return self._proc_fn(sample, primitive, dx, dy)
        dists = dx ** 2 + dy ** 2
        return SampleOutput(
            rgb=sample.rgb,
            weights=torch.exp(-dists.min(dim=0).values)[None, :] * sample.weights,
            co=sample.co,
        )
