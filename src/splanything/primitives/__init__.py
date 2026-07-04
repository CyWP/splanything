"""Primitive classes for splatting."""

from .base import Primitive
from .cubic_grad import CubicGrad
from .gaussian import Gaussian

__all__ = ["Primitive", "CubicGrad", "Gaussian"]
