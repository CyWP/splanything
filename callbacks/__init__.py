"""Callbacks for training monitoring and control.

Exposes:
- Callback: Base class for training callbacks
- LoopControl: Epoch counting and interrupt handling
- ImgUpdate: Real-time image preview during training
- LossLogger: Logs loss values per epoch
"""

from typing import Any, Dict, List

from .generic import Callback
from .loop_control import LoopControl
from .img_update import ImgUpdate
from .losses_log import LossLogger

CLASSES = [LoopControl, ImgUpdate, LossLogger]
CALLBACKS = {c._name: c for c in CLASSES}


def get_callbacks(data: Dict[str, Any]) -> List[Callback]:
    """Instantiate callbacks from configuration dict.

    Args:
        data: Dict mapping callback name to config kwargs.

    Returns:
        List of Callback instances.
    """
    callbacks = []
    for c in data.keys():
        ccls = CALLBACKS.get(c, None)
        if ccls is None:
            raise KeyError(
                f"Class '{c}' is an invalid callback class.\n Valid classes:\n{CALLBACKS.keys()}"
            )
        kwargs = data[c]
        callbacks.append(ccls(**kwargs))
    return callbacks
