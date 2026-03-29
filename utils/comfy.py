import torch
import comfy.model_management as mm

from comfy.utils import ProgressBar
from typing import Optional, Any


class ComfyUtils:

    @staticmethod
    def active_device() -> torch.device:
        return mm.get_torch_device()

    @staticmethod
    def is_interrupted() -> bool:
        return bool(mm.interrupt_processing)

    @staticmethod
    def check_interrupt():
        if mm.interrupt_processing:
            raise InterruptedError("ComfyUI processing interrupted")

    @staticmethod
    def make_progress(total_steps: int):
        return ProgressBar(total_steps)

    @staticmethod
    def update_progress(pbar, step: int):
        if pbar is not None:
            pbar.update_absolute(step)

    @staticmethod
    def preview_image(node: Any, image: torch.Tensor):
        """
        Safely send preview image if available.
        """
        if node is None:
            return
        fn = getattr(node, "send_preview", None)
        if callable(fn):
            try:
                fn(image)
            except Exception:
                pass  # fail silently

    @staticmethod
    def preview_text(node, text: str):
        """
        There is no official text preview channel,
        so we fallback to printing (visible in console/log).
        """
        print(f"[ComfyUI TEXT PREVIEW] {text}")

    @staticmethod
    def preview(node, image=None, text=None):
        if image is not None:
            ComfyUtils.preview_image(node, image)
        if text is not None:
            ComfyUtils.preview_text(node, text)

    @staticmethod
    def seeded(seed: int, devices=None):
        """
        Context manager for deterministic blocks.
        """
        return torch.random.fork_rng(devices=devices or ComfyUtils.active_device())

    @staticmethod
    def set_seed(seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def safe_call(fn, *args, **kwargs):
        if fn is None:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception:
            return None

    @staticmethod
    def ensure_batch(x):
        if isinstance(x, torch.Tensor) and x.ndim == 3:
            return x.unsqueeze(0)
        return x
