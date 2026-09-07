"""Refinement rule base classes and criterion processors."""

from __future__ import annotations

import logging
import weakref
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from jaxtyping import Bool, Float
from torch import Tensor

from ...primitives import Primitive

_logger = logging.getLogger(__name__)


class CriterionProcessor(ABC):
    """Transform a refinement criterion before judgement.

    Subclasses implement ``apply`` to modify the per-primitive criterion
    tensor (e.g., spatially weighting it via a map). Multiple processors
    can be chained on a single ``RefinementRule``.

    Notes:
        - The criterion shape is ``(N, ...)`` where ``N`` is the number
          of primitives. Processors must preserve this leading dimension.
    """

    @abstractmethod
    def apply(
        self,
        primitive: Primitive,
        rule: RefinementRule,
        criterion: Float[Tensor, "N ..."],
        **kwargs,
    ) -> Float[Tensor, "N ..."]:
        """Modify the criterion.

        Args:
            primitive: Primitive being evaluated.
            rule: The refinement rule this processor belongs to.
            criterion: Per-primitive criterion (N, ...).

        Returns:
            Modified criterion (N, ...).
        """

    def __call__(
        self,
        primitive: Primitive,
        rule: RefinementRule,
        criterion: Float[Tensor, "N ..."],
        **kwargs,
    ) -> Float[Tensor, "N ..."]:
        """Dispatch to :meth:`apply`."""
        return self.apply(primitive, rule, criterion, **kwargs)


class RefinementRule(ABC):
    """Base class for primitive refinement rules.

    A rule computes a per-primitive criterion, optionally transforms it
    through a chain of ``CriterionProcessor`` s, then judges which
    primitives to modify. The rule may be shared across multiple
    primitives; per-primitive state is tracked via a weak-key registry.

    Attributes:
        interval: Fire every N invocations of ``__call__``.
        processors: List of ``CriterionProcessor`` modifying the criterion.

    Notes:
        - ``calls(primitive)`` counts executions of ``apply`` on that
          primitive, not invocations of ``__call__``.
        - A primitive is registered either explicitly via
          ``register(primitive)`` or lazily on first ``__call__``.
        - Registry keys are weak; entries vanish when the primitive is
          garbage-collected.

    Warnings:
        - Subclasses must implement ``criterion`` and ``judge``.
    """

    def __init__(
        self,
        interval: int = 1,
        processors: Optional[List[CriterionProcessor]] = None,
        **kwargs,
    ):
        """
        Args:
            interval: Fire every N invocations of ``__call__`` (default 1).
            processors: Optional list of ``CriterionProcessor`` applied
                to the criterion before ``judge``.
        """
        self.interval = interval
        self.processors: List[CriterionProcessor] = (
            [] if processors is None else list(processors)
        )
        self._calls: weakref.WeakKeyDictionary[Primitive, int] = (
            weakref.WeakKeyDictionary()
        )
        self._ticks: weakref.WeakKeyDictionary[Primitive, int] = (
            weakref.WeakKeyDictionary()
        )

    def register(self, primitive: Primitive) -> None:
        """Register a primitive, initialising per-primitive counters.

        Idempotent: re-registering an already-known primitive is a no-op.

        Args:
            primitive: Primitive to register.
        """
        if primitive not in self._calls:
            self._calls[primitive] = 0
            self._ticks[primitive] = 0

    def unregister(self, primitive: Primitive) -> None:
        """Remove a primitive's per-primitive state.

        Args:
            primitive: Primitive to unregister.
        """
        self._calls.pop(primitive, None)
        self._ticks.pop(primitive, None)

    def calls(self, primitive: Primitive) -> int:
        """Number of times ``apply`` has executed on this primitive.

        Args:
            primitive: Registered primitive.

        Returns:
            Execution count. Returns 0 for unregistered primitives.
        """
        return self._calls.get(primitive, 0)

    def add_processor(self, processor: CriterionProcessor) -> None:
        """Append a criterion processor.

        Args:
            processor: Processor to append.
        """
        self.processors.append(processor)

    def can_apply(self, primitive: Primitive, **kwargs) -> bool:
        """Hook for additional gating logic.

        Args:
            primitive: Primitive being considered.

        Returns:
            True if the rule may execute on this primitive.
        """
        return True

    def log_result(self, result: Any) -> str:
        """Format the rule's result for logging.

        Args:
            result: Return value of ``apply``.

        Returns:
            Log message string.
        """
        return f"{self.__class__.__name__}: called"

    @abstractmethod
    def criterion(self, primitive: Primitive, **kwargs) -> Float[Tensor, "N ..."]:
        """Generate the criterion by which application is judged."""

    def processed_criterion(
        self, primitive: Primitive, **kwargs
    ) -> Float[Tensor, "N ..."]:
        """Generate the criterion by which application is judged, with processors applied."""
        crit = self.criterion(primitive, **kwargs)
        for proc in self.processors:
            crit = proc(primitive, self, crit, **kwargs)
        return crit

    @abstractmethod
    def judge(self, criterion: Float[Tensor, "N ..."], **kwargs) -> Any:
        """Process the criterion into the output for rule application."""

    def apply(self, primitive: Primitive, **kwargs) -> Any:
        """Execute the rule: criterion -> processors -> judge.

        Args:
            primitive: Primitive to refine.

        Returns:
            Whatever ``judge`` returns.
        """
        return self.judge(self.processed_criterion(primitive, **kwargs))

    def __call__(self, primitive: Primitive, **kwargs) -> Optional[Any]:
        """Invoke the rule, gating on interval and ``can_apply``.

        Per-primitive ticks increment on every invocation. The rule
        fires only when ``ticks % interval == 0`` and ``can_apply``
        returns True. On a successful fire, ``calls`` is incremented
        after ``apply`` runs.

        Args:
            primitive: Primitive to evaluate.

        Returns:
            Result of ``apply(primitive)`` on fire, else ``None``.
        """
        self.register(primitive)
        self._ticks[primitive] += 1
        if self._ticks[primitive] % self.interval == 0 and self.can_apply(
            primitive, **kwargs
        ):
            result = self.apply(primitive, **kwargs)
            if result is not None:
                _logger.info(self.log_result(result))
            self._calls[primitive] += 1
            return result
        return None


class FilterRule(RefinementRule, ABC):
    """Refinement rule that produces a boolean keep/cull mask.

    ``judge`` returns ``True`` == KEEP, ``False`` == REMOVE.
    """

    @abstractmethod
    def judge(self, criterion: Float[Tensor, "N ..."]) -> Optional[Bool[Tensor, "N"]]:
        """Threshold the criterion into a keep mask.

        Args:
            criterion: Per-primitive criterion (N, ...).

        Returns:
            keep: Boolean mask (N,). True = KEEP, False = REMOVE.
        """

    def log_result(self, result: Optional[Bool[Tensor, "N"]]) -> str:
        """Format the number of primitives marked for culling."""
        n = 0 if result is None else (~result).sum()
        return f"{self.__class__.__name__}: {n} primitives marked for filtering."


class SplitRule(RefinementRule, ABC):
    """Refinement rule that produces a boolean split/ignore mask.

    ``judge`` returns ``True`` == SPLIT, ``False`` == IGNORE.
    """

    @abstractmethod
    def judge(self, criterion: Float[Tensor, "N ..."]) -> Bool[Tensor, "N"]:
        """Threshold the criterion into a split mask.

        Args:
            criterion: Per-primitive criterion (N, ...).

        Returns:
            split: Boolean mask (N,). True = SPLIT, False = IGNORE.
        """

    def log_result(self, result: Optional[Bool[Tensor, "N"]]) -> str:
        """Format the number of primitives marked for splitting."""
        n = 0 if result is None else result.sum()
        return f"{self.__class__.__name__}: {n} primitives marked for splitting."
