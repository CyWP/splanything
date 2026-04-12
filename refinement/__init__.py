"""Refinement rules for adaptive primitive optimization.

Refinement rules are callbacks that modify primitives during training
to improve reconstruction quality. They run at EPOCH_END and can split,
merge, or cull primitives based on various criteria.

Exposes:
- RefinementRule: Abstract base class for refinement rules
- FilterRule: Rules that return a boolean mask of primitives to keep
- SplitRule: Rules that return a boolean mask of primitives to split
- GradSplit: Split primitives with high gradient-to-area ratios
- AreaSplit: Split primitives exceeding area/scale threshold
- AlphaCull: Remove primitives with low alpha values
- IsoSplit: Split primitives that are too anisotropic
- get_refinement_rule: Factory function to instantiate a single rule from config
"""

from typing import Dict, Any

from .generic import RefinementRule, FilterRule, SplitRule
from .grad_split import GradSplit
from .area_split import AreaSplit
from .alpha_cull import AlphaCull
from .iso_split import IsoSplit

__all__ = [
    "RefinementRule",
    "FilterRule",
    "SplitRule",
    "GradSplit",
    "AreaSplit",
    "AlphaCull",
    "IsoSplit",
]

CLASSES = [GradSplit, AreaSplit, AlphaCull, IsoSplit]

REFINEMENTS = {c.__name__.lower(): c for c in CLASSES}


def register_refinement(cls: type[RefinementRule]):
    """Register a refinement rule class for use in the framework.

    Args:
        cls: RefinementRule class to register.

    Returns:
        The registered class.
    """
    CLASSES.append(cls)
    REFINEMENTS[cls.__name__.lower()] = cls
    return cls


def get_refinement_rule(name: str, kwargs: Dict[str, Any], primitive) -> RefinementRule:
    """Instantiate a single refinement rule from name and kwargs.

    Args:
        name: Rule class name (e.g., "GradSplit").
        kwargs: Constructor arguments (primitive will be injected).
        primitive: The primitive to apply the rule to.

    Returns:
        RefinementRule instance.

    Raises:
        KeyError: If name is not a valid refinement rule class.
    """
    rcls = REFINEMENTS.get(name.lower(), None)
    if rcls is None:
        raise KeyError(
            f"'{name}' is an invalid refinement rule class.\n"
            f"Valid classes: {list(REFINEMENTS.keys())}"
        )
    kwargs["primitive"] = primitive
    return rcls(**kwargs)
