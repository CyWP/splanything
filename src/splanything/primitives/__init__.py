"""Primitive classes for splatting."""

from .base import Primitive
from .cubic_grad import CubicFanPrimitive
from .gaussian import GaussianPrimitive

__all__ = ["Primitive", "CubicFanPrimitive", "GaussianPrimitive"]
