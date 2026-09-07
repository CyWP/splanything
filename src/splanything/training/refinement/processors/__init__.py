"""Criterion processors that transform the criterion tensor before judgement.

Exposes:
- MapCriterionProcessor
- FlexibleCriterionProcessor
- CriterionReduction
"""

from .flex import FlexibleCriterionProcessor
from .map_processor import MapCriterionProcessor
from .reduction import CriterionReduction

__all__ = [
    "MapCriterionProcessor",
    "FlexibleCriterionProcessor",
    "CriterionReduction",
]
