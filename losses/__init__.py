"""Loss functions for image reconstruction optimization.

Exposes:
- Loss: Base class for loss functions
- L1Loss: L1 (MAE) loss
- L2Loss: L2 (MSE) loss
- get_loss: Factory function to instantiate a single loss from config
"""

from typing import Dict, Any

from .generic import Loss
from .l1 import L1Loss
from .l2 import L2Loss
from .ssim import SSIMLoss

CLASSES = [L1Loss, L2Loss, SSIMLoss]

LOSSES = {c._name.lower(): c for c in CLASSES}


def register_loss(cls: type[Loss]):
    """Register a loss class for use in the framework.

    Args:
        cls: Loss class to register.

    Returns:
        The registered class.
    """
    CLASSES.append(cls)
    LOSSES[cls._name.lower()] = cls
    return cls


def get_loss(name: str, kwargs: Dict[str, Any]) -> Loss:
    """Instantiate a single loss from name and kwargs.

    Args:
        name: Loss class name (e.g., "L1", "L2").
        kwargs: Constructor arguments including 'weight'.

    Returns:
        Loss instance.

    Raises:
        KeyError: If name is not a valid loss class.
    """
    lcls = LOSSES.get(name.lower(), None)
    if lcls is None:
        raise KeyError(
            f"'{name}' is an invalid loss class.\n"
            f"Valid classes: {list(LOSSES.keys())}"
        )
    return lcls(**kwargs)
