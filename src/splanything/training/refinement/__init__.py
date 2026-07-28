"""Refinement rules for adaptive primitive modification during training.

Rules modify a ``Primitive`` in-place (filtering, splitting) based on
per-primitive criteria. Processors transform the criterion before
judgement.

Exposes:
- FilterRule
- SplitRule
- MapFilter
- MapSplit
- rules
- processors
"""

from . import processors, rules
from .base import FilterRule, SplitRule
from .rules.map_filter import MapFilter
from .rules.map_split import MapSplit

__all__ = [
    "FilterRule",
    "SplitRule",
    "MapFilter",
    "MapSplit",
    "rules",
    "processors",
]
