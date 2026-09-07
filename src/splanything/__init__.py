"""Splat any differentiable function: primitives, rendering and training subpackages.

Exposes:
- primitives
- rendering
- training
- ImgUtils
"""
__version__ = "0.1.0"
from . import primitives, rendering, training
from .utils.img import ImgUtils

__all__ = ["__version__", "primitives", "rendering", "training", "ImgUtils"]
