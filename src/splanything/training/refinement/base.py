from abc import ABC, abstractmethod
from typing import Any, Optional

from jaxtyping import Bool
from torch import Tensor

from ...primitives import Primitive


class RefinementRule(ABC):
    def __init__(self, interval: int = 1, **kwargs):
        self.interval = interval
        self.calls = 0

    def can_apply(self, primitive: Primitive, **kwargs) -> bool:
        return True

    @abstractmethod
    def apply(
        self,
        primtive: Primitive,
    ) -> Any:
        """Apply the rule."""
        pass

    def __call__(self, primitive: Primitive, **kwargs) -> Optional[Any]:
        self.calls += 1
        if self.calls % self.interval == 0 and self.can_apply(primitive):
            return self.apply(primitive)
        return None


class FineTuneRule(RefinementRule, ABC):
    @abstractmethod
    def apply(self, primitive, **kwargs) -> Primitive:
        """
        Edit a primitive object in place.
        """
        pass


class FilterRule(RefinementRule, ABC):
    @abstractmethod
    def apply(self, primitive: Primitive, **kwargs) -> Optional[Bool[Tensor, "N"]]:
        """Define which primitives to keep.

        Args:
            primitive: Primitive from which primitive and trainer state can be accessed.

        Returns:
            keep: Boolean tensor of shape (len(primitive),). True values keep, False values remove.
        """
        pass


class SplitRule(RefinementRule, ABC):
    @abstractmethod
    def apply(self, primitive: Primitive, **kwargs) -> Bool[Tensor, "N"]:
        """Define which primitives to split.

        Args:
            primitive: Primitive from which primitive and trainer state can be accessed.

        Returns:
            split: Boolean tensor of shape (len(primitive),). True values split, False values ignore.
        """
        pass
