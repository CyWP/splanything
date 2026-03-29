from typing import Dict, List, Any

from .generic import Loss
from .l1 import L1Loss
from .l2 import L2Loss

CLASSES = [L1Loss, L2Loss]

LOSSES = {c._name: c for c in CLASSES}


def get_losses(data: Dict[str, Any]) -> Dict[str, Loss]:
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
