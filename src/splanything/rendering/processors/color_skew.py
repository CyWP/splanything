from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from jaxtyping import Float
from torch import Tensor

from .base import SampleProcessor
from ..sample_output import SampleOutput

if TYPE_CHECKING:
    from ...primitives.base import Primitive

_REDUCTION_FN = {
    "MIN": lambda x: x.min(dim=-1).values,
    "MAX": lambda x: x.max(dim=-1).values,
    "MEAN": lambda x: x.mean(dim=-1),
}


class ColorSkewSampleProcessor(SampleProcessor):
    def __init__(
        self,
        target_colors: Float[Tensor, "Nt 3"],
        sigma: float = 1.0,
        reduction: str = "MIN",
        rescale: bool = True,
    ):
        if reduction not in _REDUCTION_FN:
            raise ValueError(
                f"Unknown reduction '{reduction}'. Expected one of {list(_REDUCTION_FN)}."
            )
        self._target_colors = target_colors
        self._sigma = sigma
        self._reduction = reduction
        self._rescale = rescale

    def process(
        self,
        sample: SampleOutput,
        primitive: Primitive,
    ) -> SampleOutput:
        diff = sample.rgb[..., None, :] - self._target_colors  # (Nc, Np, Nt, 3)
        dist_sq = diff.square().sum(dim=-1)  # (Nc, Np, Nt)
        dist_sq = _REDUCTION_FN[self._reduction](dist_sq)  # (Nc, Np)

        scale = torch.exp(-dist_sq / self._sigma)  # (Nc, Np)

        w_orig_sum = sample.weights.sum(dim=1, keepdim=True)
        new_weights = sample.weights * scale
        if self._rescale:
            w_new_sum = new_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
            new_weights = new_weights * (w_orig_sum / w_new_sum)

        return SampleOutput(rgb=sample.rgb, weights=new_weights, co=sample.co)
