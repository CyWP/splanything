from abc import ABC, abstractmethod
from jaxtyping import Bool
from torch import Tensor
from typing import Tuple, Optional, Any

from callbacks import Callback
from primitives import Primitive
from trainers import Trainer, EPOCH_END


class RefinementRule(Callback, ABC):
    """Abstract base class for primitive refinement rules.

        Refinement rules are callbacks that modify primitives during training
    to improve reconstruction quality. They run at EPOCH_END and can perform
    operations like splitting, merging, or culling primitives.

        Attributes:
            primitive: The primitive being refined.
            _stages: Fixed to [EPOCH_END] - rules always run at epoch end.

        Notes:
            - Subclasses must implement `apply()` method.
            - Subclasses can specify required behaviors via `required_behaviours`.
            - The `check_apply()` method validates primitive compatibility.
    """

    _stages = [EPOCH_END]

    def __init__(self, primitive: Primitive, **kwargs):
        """Initialize refinement rule.

        Args:
            primitive: The primitive to refine.
            **kwargs: Additional rule-specific arguments.

        Raises:
            Exception: If primitive does not meet required behaviors.
        """
        super().__init__()
        if not self.check_apply(primitive):
            raise Exception(
                f"Provided arguments are invalid for rule of type '{self.__class__}'"
            )
        self.primitive = primitive

    @property
    def required_behaviours(self) -> Tuple[type[Any], ...]:
        """Protocol classes that the primitive must implement.

        Returns:
            Tuple of behavior protocol classes.
        """
        return tuple()

    def check_apply(self, primitive: Primitive) -> bool:
        """Validate that primitive supports required behaviors.

        Args:
            primitive: The primitive to validate.

        Returns:
            True if valid, raises TypeError otherwise.

        Raises:
            TypeError: If primitive missing required behaviors.
        """
        for r in self.required_behaviours:
            if not isinstance(primitive, r):
                raise TypeError(
                    f"Primitive '{primitive.__class__.__name__}' must inherit "
                    f"behaviour from '{r.__name__}' in order to apply rule "
                    f"'{self.__class__.__name__}'."
                )
        return True

    @abstractmethod
    def apply(
        self,
        trainer: Optional[Trainer] = None,
    ):
        """Apply the refinement rule in-place.

        Args:
            trainer: Optional trainer for accessing epoch state.
        """
        pass

    def run(self, trainer: Trainer, stage: str):
        """Callback interface - delegates to apply()."""
        return self.apply(trainer)


class FilterRule(RefinementRule, ABC):

    _filter_rule = True

    def __init__(self, primitive, **kwargs):
        super().__init__(primitive, **kwargs)

    @abstractmethod
    def apply(
        self,
        trainer: Optional[Trainer] = None,
    ) -> Bool[Tensor, "N"]:
        """Define which primitives to keep.

        Args:
            trainer: Optional trainer for accessing epoch state.

        Returns:
            keep: Boolean tensor of shape (len(primitive),). True values keep, False values remove.
        """
        pass


class SplitRule(RefinementRule, ABC):

    _split_rule = True

    def __init__(self, primitive, **kwargs):
        super().__init__(primitive, **kwargs)

    @abstractmethod
    def apply(
        self,
        trainer: Optional[Trainer] = None,
    ) -> Bool[Tensor, "N"]:
        """Define which primitives to split.

        Args:
            trainer: Optional trainer for accessing epoch state.

        Returns:
            split: Boolean tensor of shape (len(primitive),). True values split, False values ignore.
        """
        pass
