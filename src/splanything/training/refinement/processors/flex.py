"""Criterion processor applying a user-supplied function."""

from __future__ import annotations

from typing import Callable

from jaxtyping import Float
from torch import Tensor

from ..base import CriterionProcessor, RefinementRule
from ....primitives.base import Primitive


class FlexibleCriterionProcessor(CriterionProcessor):
    """Flexible criterion processor that applies a user-supplied function.

    The function is called with the criterion tensor and must return a
    modified criterion of the same leading dimension ``(N, ...)``.

    Attributes:
        proc_fn: Callable ``(criterion) -> modified_criterion``.
    """

    def __init__(
        self,
        proc_fn: Callable[[Float[Tensor, "N ..."]], Float[Tensor, "N ..."]],
    ):
        """Store the processing function.

        Args:
            proc_fn: Callable ``(criterion) -> modified_criterion``.
        """
        self.proc_fn = proc_fn

    def apply(
        self,
        primitive: Primitive,
        rule: RefinementRule,
        criterion: Float[Tensor, "N ..."],
        **kwargs,
    ) -> Float[Tensor, "N ..."]:
        """Apply ``proc_fn`` to the criterion.

        Args:
            primitive: Unused.
            rule: Unused.
            criterion: Per-primitive criterion (N, ...).

        Returns:
            Modified criterion (N, ...).
        """
        return self.proc_fn(criterion)
