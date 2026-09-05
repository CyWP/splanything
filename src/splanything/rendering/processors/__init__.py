"""Sample processors transforming SampleOutputs before rasterization."""

from .base import SampleProcessor
from .flex import FlexibleSampleProcessor
from .dist import DistanceSampleProcessor
from .mapped import MappedSampleProcessor
from .multi import MultiSampleProcessor
from .vec import VecSampleProcessor
from .color_skew import ColorSkewSampleProcessor

__all__ = [
    "SampleProcessor",
    "FlexibleSampleProcessor",
    "DistanceSampleProcessor",
    "MappedSampleProcessor",
    "MultiSampleProcessor",
    "VecSampleProcessor",
    "ColorSkewSampleProcessor",
]
