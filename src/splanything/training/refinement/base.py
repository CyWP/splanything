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
    @abstractmethod
    def apply(
        self,
        primitive: Primitive,
        rule: "RefinementRule",
        criterion: Float[Tensor, "N ..."],
        **kwargs,
    ) -> Float[Tensor, "N ..."]:
        """Modify the criterion."""

    def __call__(
        self,
        primitive: Primitive,
        rule: "RefinementRule",
        criterion: Float[Tensor, "N ..."],
        **kwargs,
    ) -> Float[Tensor, "N ..."]:
        return self.apply(primitive, rule, criterion, **kwargs)


class RefinementRule(ABC):
    """Base class for primitive refinement rules.

    A rule may be shared across multiple primitives. Per-primitive state
    (call counts, invocation ticks) is tracked via a weak-key registry so
    that primitive lifetime controls registry entries.

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
            processors: Optional list of ``CriterionProcessor`` instances
                applied to the criterion before ``judge``.
        """
        self.interval = interval
        self.processors: List[CriterionProcessor] = (
            [] if processors is None else list(processors)
        )
        self._calls: "weakref.WeakKeyDictionary[Primitive, int]" = (
            weakref.WeakKeyDictionary()
        )
        self._ticks: "weakref.WeakKeyDictionary[Primitive, int]" = (
            weakref.WeakKeyDictionary()
        )

    def register(self, primitive: Primitive) -> None:
        """Register a primitive with this rule.

        Initialises the per-primitive call and tick counters to 0.
        Idempotent: re-registering an already-known primitive is a no-op.

        Args:
            primitive: Primitive to register.
        """
        if primitive not in self._calls:
            self._calls[primitive] = 0
            self._ticks[primitive] = 0

    def unregister(self, primitive: Primitive) -> None:
        """Remove a primitive's per-primitive state from this rule.

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
        return f"{self.__class__.__name__}: called"

    @abstractmethod
    def criterion(self, primitive: Primitive, **kwargs) -> Float[Tensor, "N ..."]:
        """Generate the criterion by which application is judged."""

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
        crit = self.criterion(primitive, **kwargs)
        for proc in self.processors:
            crit = proc(primitive, self, crit, **kwargs)
        return self.judge(crit)

    def __call__(self, primitive: Primitive, **kwargs) -> Optional[Any]:
        """Invoke the rule, gating on interval and ``can_apply``.

        Per-primitive ticks increment on every invocation; the rule
        fires only when ``ticks % interval == 0`` and ``can_apply``
        returns True. On a successful fire, ``calls`` is incremented
        after ``apply`` runs, and the result is returned.

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
    @abstractmethod
    def judge(self, criterion: Float[Tensor, "N ..."]) -> Optional[Bool[Tensor, "N"]]:
        """Returns boolean mask, True==KEEP, False==REMOVE."""

    def log_result(self, result: Optional[Bool[Tensor, "N"]]) -> str:
        return f"{self.__class__.__name__}: {0 if result is None else (~result).sum()} primitives marked for filtering."


class SplitRule(RefinementRule, ABC):
    @abstractmethod
    def judge(self, criterion: Float[Tensor, "N ..."]) -> Bool[Tensor, "N"]:
        """Returns a boolean mask, True==SPLIT, False==IGNORE."""

    def log_result(self, result: Optional[Bool[Tensor, "N"]]) -> str:
        return f"{self.__class__.__name__}: {0 if result is None else (result).sum()} primitives marked for splitting."
