"""Criterion processors that transform the criterion tensor before judgement.

Exposes:
- MapCriterionProcessor
"""

from .map_processor import MapCriterionProcessor

__all__ = ["MapCriterionProcessor"]
