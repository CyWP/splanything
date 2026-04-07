import torch
import comfy.model_management as mm

from comfy.utils import ProgressBar
from typing import Optional, Any
from jaxtyping import Float
from torch import Tensor


class ComfyUtils:
    """ComfyUI integration utilities.

    Static methods for device management, interrupt handling, progress bars,
    preview output, and seed management.
    """

    @staticmethod
    def active_device() -> torch.device:
        """Get current active compute device.

        Returns:
            torch.device for current ComfyUI device.
        """
        return mm.get_torch_device()

    @staticmethod
    def is_interrupted() -> bool:
        """Check if ComfyUI processing was interrupted.

        Returns:
            True if user triggered interrupt.
        """
        return bool(mm.interrupt_processing)

    @staticmethod
    def check_interrupt():
        """Raise if ComfyUI processing was interrupted."""
        if mm.interrupt_processing:
            raise InterruptedError("ComfyUI processing interrupted")

    @staticmethod
    def make_progress(total_steps: int):
        """Create ComfyUI progress bar.

        Args:
            total_steps: Total number of steps.

        Returns:
            ProgressBar instance.
        """
        return ProgressBar(total_steps)

    @staticmethod
    def update_progress(pbar, step: int):
        """Update progress bar.

        Args:
            pbar: ProgressBar instance.
            step: Current step.
        """
        if pbar is not None:
            pbar.update_absolute(step)

    @staticmethod
    def preview_image(node: Any, image: Float[Tensor, "B H W C"]):
        """Send preview image to ComfyUI node.

        Args:
            node: ComfyUI node with send_preview method.
            image: Image tensor (B, H, W, C) in [0, 1].
        """
        if node is None:
            return
        fn = getattr(node, "send_preview", None)
        if callable(fn):
            try:
                fn(image)
            except Exception:
                pass

    @staticmethod
    def preview_text(node, text: str):
        """Print text preview to console.

        Args:
            node: Unused (reserved for future text preview).
            text: Text to print.
        """
        print(f"[ComfyUI TEXT PREVIEW] {text}")

    @staticmethod
    def preview(node, image=None, text=None):
        """Send preview to ComfyUI.

        Args:
            node: ComfyUI node.
            image: Optional image tensor.
            text: Optional text string.
        """
        if image is not None:
            ComfyUtils.preview_image(node, image)
        if text is not None:
            ComfyUtils.preview_text(node, text)

    @staticmethod
    def seeded(seed: int, devices=None):
        """Context manager for deterministic operations.

        Args:
            seed: Random seed.
            devices: Optional device list.

        Returns:
            RNG context manager.
        """
        return torch.random.fork_rng(devices=devices or ComfyUtils.active_device())

    @staticmethod
    def set_seed(seed: int):
        """Set random seed for reproducibility.

        Args:
            seed: Random seed value.
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def safe_call(fn, *args, **kwargs):
        """Safely call function, returning None on error.

        Args:
            fn: Function to call.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Function result or None on exception.
        """
        if fn is None:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception:
            return None

    @staticmethod
    def ensure_batch(x: Float[Tensor, "..."]) -> Float[Tensor, "B ..."]:
        """Ensure tensor has batch dimension.

        Args:
            x: Tensor (C, H, W) or (B, C, H, W).

        Returns:
            Tensor (B, C, H, W).
        """
        if isinstance(x, torch.Tensor) and x.ndim == 3:
            return x.unsqueeze(0)
        return x
