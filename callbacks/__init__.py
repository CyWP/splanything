"""Callbacks for training monitoring and control.

Exposes:
- Callback: Base class for training callbacks
- LoopControl: Epoch counting and progress bar (tqdm)
- ComfyUIControl: ComfyUI-specific loop control with interrupt handling
- NodePreview: Real-time image preview during training
- PreviewWindow: Preview window with optional target side-by-side
- PrimitiveCheckpoint: Save primitive checkpoint at interval
- LossLogger: Logs loss values per epoch
- get_callback: Factory function to instantiate a single callback from config
"""

from typing import Any, Dict

from .generic import Callback
from .loop_control import LoopControl
from .comfy_control import ComfyUIControl
from .img_update import NodePreview
from .preview_window import PreviewWindow
from .primitive_checkpoint import PrimitiveCheckpoint
from .losses_log import LossLogger

CLASSES = [
    LoopControl,
    ComfyUIControl,
    NodePreview,
    PreviewWindow,
    PrimitiveCheckpoint,
    LossLogger,
]

CALLBACKS = {c.__name__.lower(): c for c in CLASSES}


def register_callback(cls: type[Callback]):
    """Register a callback class for use in the framework.

    Args:
        cls: Callback class to register.

    Returns:
        The registered class.
    """
    CLASSES.append(cls)
    CALLBACKS[cls.__name__.lower()] = cls
    return cls


def get_callback(name: str, kwargs: Dict[str, Any]) -> Callback:
    """Instantiate a single callback from name and kwargs.

    Args:
        name: Callback class name (e.g., "NodePreview").
        kwargs: Constructor arguments.

    Returns:
        Callback instance.

    Raises:
        KeyError: If name is not a valid callback class.
    """
    ccls = CALLBACKS.get(name.lower(), None)
    if ccls is None:
        raise KeyError(
            f"'{name}' is an invalid callback class.\n"
            f"Valid classes: {list(CALLBACKS.keys())}"
        )
    return ccls(**kwargs)
