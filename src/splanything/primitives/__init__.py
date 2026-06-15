"""Geometric primitives for image reconstruction.

Exposes:
- Primitive: Base class for trainable image primitives
- CubicGrad: Cubic gradient primitives
- Gaussian: Simple Gaussian primitives
- protocols:
    - Splittable: Requires split(idx) method for refinement
    - HasAreas: Requires areas property returning (N,) tensor
    - HasAlphas: Requires alphas attribute of shape (N,)
    - HasScales: Requires scales property returning tuple of 2 (N,) tensors
"""

from .protocols import HasAlphas, HasAreas, Splittable, HasScales
from .generic import Primitive
from .cubic_grad import CubicGrad
from .gaussian import Gaussian

__all__ = [
    "Primitive",
    "CubicGrad",
    "Gaussian",
    "HasAlphas",
    "HasAreas",
    "Splittable",
    "HasScales",
]
