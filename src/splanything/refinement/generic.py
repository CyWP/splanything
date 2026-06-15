from abc import ABC, abstractmethod
from typing import Any
from jaxtyping import Bool, Integer
from torch import Tensor

from splanything.training import Trainer, EPOCH_END


class RefinementRule(ABC):
    _stages = [EPOCH_END]

    @abstractmethod
    def apply(
        self,
        trainer: Trainer,
    ) -> Any:
        """Apply the rule."""
        pass

    def __call__(self, trainer: Trainer) -> Any:
        return self.apply(trainer)


class FilterRule(RefinementRule, ABC):

    _filter_rule = True

    @abstractmethod
    def apply(
        self,
        trainer: Trainer,
    ) -> Bool[Tensor, "N"]:
        """Define which primitives to keep.

        Args:
            trainer: Trainer from which primitive and trainer state can be accessed.

        Returns:
            keep: Boolean tensor of shape (len(primitive),). True values keep, False values remove.
        """
        pass


class SplitRule(RefinementRule, ABC):

    _split_rule = True

    @abstractmethod
    def apply(
        self,
        trainer: Trainer,
    ) -> Bool[Tensor, "N"]:
        """Define which primitives to split.

        Args:
            trainer: Trainer from which primitive and trainer state can be accessed.

        Returns:
            split: Boolean tensor of shape (len(primitive),). True values split, False values ignore.
        """
        pass


class CombineRule(RefinementRule, ABC):

    _combine_rule = True

    @abstractmethod
    def apply(
        self,
        trainer: Trainer,
    ) -> Integer[Tensor, "num_collections collection_size"]:
        """Define which collections of primitives to combine.

        Args:
            trainer: Trainer from which primitive and trainer state can be accessed.

        Returns:
            combine: Integer tensor of shape (num_collections, collection_size), for each pair to combine.
        """
        pass
