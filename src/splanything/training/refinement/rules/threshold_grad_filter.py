from __future__ import annotations

from typing import Literal, Optional, Union

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from ....primitives.base import Primitive
from ..base import FilterRule, RefinementRule


class ThresholdGradFilter(FilterRule):
    """Filter primitives by thresholding the gradient magnitude of a named attribute.

    Reads the gradient of ``attr_name`` from the primitive, computes its
    absolute magnitude (summing over trailing dims), then applies the same
    threshold logic as ``ThresholdFilter``.

    Attributes:
        attr_name: Attribute whose gradient is evaluated (e.g. ``"alphas"``).
        threshold: Scalar threshold for the gradient magnitude.
        comparison: ``"OVER"`` (grad_mag > threshold) or ``"UNDER"`` (grad_mag < threshold).
        method: ``"ALL"`` or ``"ANY"`` for trailing dimensions of the gradient.
        interval: Fire every N invocations.
    """

    def __init__(
        self,
        attr_name: str,
        threshold: Union[float, Float[Tensor, "..."]],
        comparison: Literal["OVER", "UNDER"] = "OVER",
        method: Literal["ALL", "ANY"] = "ALL",
        interval: int = 10,
    ):
        super().__init__(interval=interval)
        self.attr_name = attr_name
        self.threshold = threshold
        self.comparison = comparison
        self.method = method

    def criterion(self, primitive: Primitive, **kwargs) -> Float[Tensor, "N ..."]:
        grad = getattr(primitive, self.attr_name).grad
        return grad.abs()

    def judge(self, criterion: Float[Tensor, "N ..."]) -> Optional[Bool[Tensor, "N"]]:
        if self.comparison == "OVER":
            result: Bool[Tensor, "N ..."] = criterion > self.threshold
        else:
            result = criterion < self.threshold
        if result.ndim == 1:
            return result
        reduce_dims = tuple(range(1, result.ndim))
        if self.method == "ALL":
            return result.all(dim=reduce_dims)
        return result.any(dim=reduce_dims)

    apply = RefinementRule.apply
