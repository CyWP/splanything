"""Loss functions for image reconstruction optimization.

Exposes:
- Loss: Base class for loss functions
- L1Loss: L1 (MAE) loss
- L2Loss: L2 (MSE) loss
- get_losses: Factory function to instantiate losses from config
"""

from typing import Dict, List, Any

from .generic import Loss
from .l1 import L1Loss
from .l2 import L2Loss

CLASSES = [L1Loss, L2Loss]

LOSSES = {c._name: c for c in CLASSES}


def get_losses(data: Dict[str, Any]) -> Dict[str, Loss]:
    """Instantiate losses from configuration dict.

    Args:
        data: Dict mapping loss name to config kwargs (must include 'weight').

    Returns:
        Dict mapping name to Loss instance.
    """
    losses = {}
    for l in data.keys():
        lcls = LOSSES.get(l, None)
        if lcls is None:
            raise KeyError(
                f"Class '{l}' is an invalid callback class.\n Valid classes:\n{LOSSES.keys()}"
            )
        kwargs = data[l]
        losses[l] = lcls(**kwargs)
    return losses
