"""Geometric primitives for image reconstruction.

Exposes:
- Primitive: Base class for trainable image primitives
- CubicGrad: Cubic gradient primitive implementation
- get_primitives: Factory function to instantiate primitives from config
"""

from typing import Dict, Any

from .generic import Primitive
from .cubic_grad import CubicGrad

CLASSES = [CubicGrad]

PRIMITIVES = {c.__name__.lower(): c for c in CLASSES}


def get_primitives(data: Dict[str, Any]) -> Dict[str, Primitive]:
    """Instantiate primitives from configuration dict.

    Args:
        data: Dict mapping primitive name to config kwargs.

    Returns:
        Dict mapping name to Primitive instance.
    """
    losses = {}
    for l in data.keys():
        lcls = PRIMITIVES.get(l, None)
        if lcls is None:
            raise KeyError(
                f"Class '{l}' is an invalid callback class.\n Valid classes:\n{LOSSES.keys()}"
            )
        kwargs = data[l]
        losses[l] = lcls.from_dict(**kwargs)
    return losses
