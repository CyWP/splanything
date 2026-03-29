from typing import Dict, List, Any

from .generic import Loss
from .l1 import L1Loss
from .l2 import L2Loss

CLASSES = [L1Loss, L2Loss]

LOSSES = {c._name: c for c in CLASSES}


def get_losses(data: Dict[str, Any]) -> List[Loss]:
    losses = []
    for l in data.keys():
        lcls = LOSSES.get(c, None)
        if lcls is None:
            raise KeyError(
                f"Class '{c}' is an invalid callback class.\n Valid classes:\n{LOSSES.keys()}"
            )
        kwargs = data[l]
        losses.append(lcls(**kwargs))
    return losses
