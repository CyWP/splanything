from __future__ import annotations

from typing import Callable, Dict, Literal

from jaxtyping import Float
from torch import Tensor

from ..base import CriterionProcessor, RefinementRule
from ....primitives.base import Primitive

_REDUCTION_FN: Dict[str, Callable] = {
    "max": lambda x, dim: x.flatten(start_dim=1).max(dim=1).values,
    "min": lambda x, dim: x.flatten(start_dim=1).min(dim=1).values,
    "mean": lambda x, dim: x.flatten(start_dim=1).mean(dim=1),
    "var": lambda x, dim: x.flatten(start_dim=1).var(dim=1),
    "std": lambda x, dim: x.flatten(start_dim=1).std(dim=1),
    "sum": lambda x, dim: x.flatten(start_dim=1).sum(dim=1),
}


class CriterionReduction(CriterionProcessor):
    """Reduce trailing dimensions of the criterion tensor.

    Applies a reduction operation over all dimensions beyond the first
    (batch) dimension, collapsing a criterion of shape ``(N, D1, D2, ...)``
    to ``(N,)``.

    Supported reductions:
        ``"max"``, ``"min"``, ``"mean"``, ``"var"``, ``"std"``, ``"sum"``

    Attributes:
        reduction: Name of the reduction operation.
    """

    def __init__(
        self,
        reduction: Literal["max", "min", "mean", "var", "std", "sum"] = "max",
    ):
        self._fn = _REDUCTION_FN[reduction]
        self.reduction = reduction

    def apply(
        self,
        primitive: Primitive,
        rule: RefinementRule,
        criterion: Float[Tensor, "N ..."],
        **kwargs,
    ) -> Float[Tensor, "N"]:
        if criterion.ndim <= 1:
            return criterion
        return self._fn(criterion, dim=None)
