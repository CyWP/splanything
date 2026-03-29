from typing import Any, Dict, List

from .generic import Callback

CLASSES = []
CALLBACKS = {c._name: c for c in CLASSES}


def get_callbacks(data: Dict[str, Any]) -> List[Callback]:
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
