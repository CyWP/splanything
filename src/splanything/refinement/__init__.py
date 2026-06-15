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
"""

from .generic import FilterRule, SplitRule, CombineRule
from .grad_split import GradSplit
from .area_split import AreaSplit
from .alpha_cull import AlphaCull
from .iso_split import IsoSplit

__all__ = [
    "FilterRule",
    "SplitRule",
    "CombineRule",
    "GradSplit",
    "AreaSplit",
    "AlphaCull",
    "IsoSplit",
]
