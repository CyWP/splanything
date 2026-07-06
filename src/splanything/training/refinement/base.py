from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from jaxtyping import Bool, Float
from torch import Tensor

from ...primitives import Primitive


class CriterionProcessor(ABC):
    @abstractmethod
    def apply(
        self,
        primitive: Primitive,
        rule: RefinementRule,
        criterion: Float[Tensor, "N ..."],
        **kwargs,
    ) -> Float[Tensor, "N ..."]:
        """Modify the criterion."""

    def __call__(
        self,
        primitive: Primitive,
        rule: RefinementRule,
        criterion: Float[Tensor, "N ..."],
        **kwargs,
    ) -> Float[Tensor, "N ..."]:
        return self.apply(primitive, rule, criterion, **kwargs)


class RefinementRule(ABC):
    def __init__(
        self,
        interval: int = 1,
        processors: Optional[List[CriterionProcessor]] = None,
        **kwargs,
    ):
        self.interval = interval
        self.processors = [] if processors is None else processors
        self.calls = 0

    def add_processor(self, processor: CriterionProcessor):
        self.processors.append(processor)

    def can_apply(self, primitive: Primitive, **kwargs) -> bool:
        return True

    @abstractmethod
    def criterion(self, primitive: Primitive, **kwargs) -> Float[Tensor, "N ..."]:
        """Generate the criterion by which the application of the rule will be judged."""

    @abstractmethod
    def judge(self, criterion: Float[Tensor, "N ..."], **kwargs) -> Any:
        """Process the criterion into the necessary output for rule application."""

    def apply(self, primitive: Primitive, **kwargs) -> Any:
        """Apply the rule."""
        crit = self.criterion(primitive, **kwargs)
        for proc in self.processors:
            crit = proc(primitive, self, crit, **kwargs)
        return self.judge(crit)

    def __call__(self, primitive: Primitive, **kwargs) -> Optional[Any]:
        self.calls += 1
        if self.calls % self.interval == 0 and self.can_apply(primitive):
            return self.apply(primitive)
        return None


class FineTuneRule(RefinementRule, ABC):
    def criterion(self, *args, **kwargs):
        raise NotImplementedError("Not used for FineTuneRule.")

    def judge(self, *args, **kwargs):
        raise NotImplementedError("Not used for FineTuneRule.")

    @abstractmethod
    def apply(self, primitive, **kwargs) -> bool:
        """
        Edit a primitive object in place. Returns true if obejct was edited, False if not.
        """
        pass


class FilterRule(RefinementRule, ABC):
    @abstractmethod
    def judge(self, criterion: Float[Tensor, "N ..."]) -> Bool[Tensor, "N"]:
        """Returns boolean mask, True==KEEP, False==REMOVE"""
        pass


class SplitRule(RefinementRule, ABC):
    @abstractmethod
    def judge(self, criterion: Float[Tensor, "N ..."]) -> Bool[Tensor, "N"]:
        """Returns a boolean mask, True==SPLIT, False==IGNORE."""
        pass
