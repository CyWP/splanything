"""Geometric primitives for image reconstruction.

Exposes:
- Primitive: Base class for trainable image primitives
- CubicGrad: Cubic gradient primitives
- Gaussian: Simple Gaussian primitives
- get_primitive: Factory function to instantiate primitive from config
- load_from_file: Factory function to load primitive from checkpoint
- protocols:
    - Splittable: Requires split(idx) method for refinement
    - HasAreas: Requires areas property returning (N,) tensor
    - HasAlphas: Requires alphas attribute of shape (N,)
    - HasScales: Requires scales property returning tuple of 2 (N,) tensors
"""

import torch

from typing import Dict, Any

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

CLASSES = [CubicGrad, Gaussian]

PRIMITIVES = {c.__name__.lower(): c for c in CLASSES}


def register_primitive(cls: type[Primitive]):
    """Register a primitive class for use in the framework.

    Args:
        cls: Primitive class to register.

    Returns:
        The registered class.
    """
    CLASSES.append(cls)
    PRIMITIVES[cls.__name__.lower()] = cls
    return cls


def get_primitive(name: str, kwargs: Dict[str, Any]) -> Primitive:
    """Instantiate a single primitive from class name and kwargs.

    Args:
        name: Primitive class name (e.g., "CubicGrad").
        kwargs: Constructor arguments for the primitive.

    Returns:
        Primitive instance.

    Raises:
        KeyError: If name is not a valid primitive class.
    """
    pcls = PRIMITIVES.get(name.lower(), None)
    if pcls is None:
        raise KeyError(
            f"'{name}' is an invalid primitive class.\n"
            f"Valid classes: {list(PRIMITIVES.keys())}"
        )
    if "state_dict" in kwargs:
        p = pcls()
        p.load_state_dict(torch.load(open(kwargs["state_dict"])))
        return p
    return pcls(**kwargs)


def load_from_file(path: str) -> Primitive:
    """Load primitive from checkpoint file.

    Args:
        path: Path to checkpoint file.

    Returns:
        Loaded Primitive instance.
    """
    state = torch.load(path, weights_only=False)
    class_name = state.pop("_class")
    pcls = PRIMITIVES.get(class_name, None)
    if pcls is None:
        raise KeyError(f"Unknown primitive class: {class_name}")
    primitive = pcls()
    primitive.load_state_dict(state)
    return primitive
