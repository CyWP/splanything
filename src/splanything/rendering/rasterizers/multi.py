"""Weighted blending of multiple rasterizers."""

import torch
from torch import Tensor
from typing import List, Tuple
from jaxtyping import Float
from .base import Rasterizer
from ..sample_output import SampleOutput
from ...utils.img import Splimage


class MultiRasterizer(Rasterizer):
    """Blends multiple rasterizers with per-coordinate weights.

    Each sub-rasterizer's RGBA output is weighted either by a scalar or
    by per-coordinate weights sampled from a Splimage mask at the sample
    coordinates, then summed (optionally normalized by the weight sum).
    """

    def __init__(
        self,
        rasterizers: List[Tuple[Rasterizer, float | Splimage]],
        normalize_weights: bool = False,
    ):
        """Initialize the rasterizer.

        Args:
            rasterizers: List of (rasterizer, weight) pairs; weight is a
                float or a Splimage mask sampled at the sample coordinates.
            normalize_weights: If True, divide the blended output by the
                summed weights per coordinate.
        """
        self._rasterizers = [r for r, _ in rasterizers]
        self._weights = [w for _, w in rasterizers]
        self._normalize_weights = normalize_weights

    def rasterize(self, sample: SampleOutput, **kwargs) -> Float[Tensor, "Nc 4"]:
        """Blend sub-rasterizer outputs by their weights.

        Args:
            sample: SampleOutput with rgb (Nc, Np, 3), weights (Nc, Np),
                co (Nc, 2).

        Returns:
            RGBA tensor (Nc, 4) blended across sub-rasterizers.
        """
        Nc = sample.co.shape[0]
        out = torch.zeros((Nc, 4), device=sample.rgb.device, dtype=sample.rgb.dtype)
        if self._normalize_weights:
            cum_weights = torch.zeros(
                (Nc, 1), device=sample.rgb.device, dtype=sample.rgb.dtype
            )
        for r, w in zip(self._rasterizers, self._weights):
            rasterized = r(sample)
            if isinstance(w, Splimage):
                weight = w.mask_sample(sample.co)
            else:
                weight = w
            out += (weight * rasterized)[0]
            if self._normalize_weights:
                cum_weights += weight
        if self._normalize_weights:
            out /= cum_weights
        return out
