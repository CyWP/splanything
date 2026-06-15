from __future__ import annotations

import logging
import torch

from pathlib import Path
from PIL import Image
from typing import List, Optional, Dict

from splanything.training import Trainer, EPOCH_END
from splanything.utils.img import ImgUtils

from .generic import Callback

_logger = logging.getLogger(__name__)


class PreviewWindow(Callback):
    """Training preview window with side-by-side target comparison.

    Displays training output alongside target image for visual comparison.
    Updates at specified interval.

    Stages: EPOCH_END
    """

    _stages: List[str] = [EPOCH_END]

    def __init__(
        self,
        frequency: int = 1,
        show_target: bool = True,
        window_title: str = "Training Preview",
        H: Optional[int] = None,
        W: Optional[int] = None,
        max_batch: Optional[int] = None,
        low_vram: Optional[bool] = None,
        save_folder: Optional[Path] = None,
    ):
        """Initialize preview window callback.

        Args:
            frequency: Update window every N epochs (default: 1).
            show_target: If True, show target side-by-side with output.
            window_title: Title for the preview window.
        """
        super().__init__()
        self.frequency = max(frequency, 1)
        self.show_target = show_target
        self.window_title = window_title
        self.H = H
        self.W = W
        self.max_batch = max_batch
        self.low_vram = low_vram
        self.save_folder = save_folder
        if save_folder is not None:
            save_folder.mkdir(parents=True, exist_ok=True)

    def run(self, trainer: Trainer, stage: str):
        """Update preview window if epoch matches frequency.

        Args:
            trainer: Current trainer instance.
            stage: Current training stage.
        """
        if trainer.epoch % self.frequency != 0:
            return
        H = trainer.sampler.H if self.H is None else self.H
        W = trainer.sampler.W if self.W is None else self.W
        with torch.no_grad():
            img = trainer.last_image(
                H, W, max_batch=self.max_batch, low_vram=self.low_vram
            )
        if self.show_target:
            tgt_img = trainer.sampler.target_img
            t_H, t_W = tgt_img.shape[-2:]
            if t_H != H or t_W != W:
                tgt_img = ImgUtils.resize(tgt_img, H, W)
            img = torch.cat([img, tgt_img], dim=3)
        window = get_window(self.window_title)
        pil_img = ImgUtils.tensor2pil(img, normalized=False)
        if self.save_folder is not None:
            pil_img.save(self.save_folder / f"{trainer.epoch:07}.png")
        try:
            window.update_image(pil_img)
        except Exception as e:
            import tkinter

            if isinstance(e, tkinter.TclError):
                _logger.info("Preview closed, stopping training.")
                trainer.stop()
            else:
                raise


"""Lightweight tkinter-based image window utilities.

Notes:
    - tkinter is imported lazily so the package can be imported in environments
      without a display or tkinter build.
"""


_WINDOWS: Dict[str, "TkImageWindow"] = {}
_ROOT: Optional["tk.Tk"] = None


def _get_root():
    """Get or create the shared tkinter root."""
    global _ROOT
    import tkinter as tk

    if _ROOT is None:
        _ROOT = tk.Tk()
        _ROOT.withdraw()
    return _ROOT


class TkImageWindow:
    """A window that displays images using tkinter.

    Supports multiple windows via factory pattern. First call to
    update_image() creates the window; subsequent calls update it.

    Args:
        title: Window title (used as unique identifier).
        width: Initial window width in pixels.
        height: Initial window height in pixels.
    """

    def __init__(self, title: str = "Image", width: int = 640, height: int = 480):
        self.title = title
        self.width = width
        self.height = height
        self._toplevel = None
        self._canvas = None
        self._photo = None

    def update_image(self, img: Image.Image) -> None:
        """Update the displayed image.

        Args:
            img: PIL Image. Will be resized to fit within window bounds
                while preserving aspect ratio.
        """
        import tkinter as tk
        from PIL import ImageTk

        W, H = img.size
        if self._toplevel is None:
            root = _get_root()
            self._toplevel = tk.Toplevel(root)
            self._toplevel.title(self.title)
            self._canvas = tk.Canvas(
                self._toplevel, width=self.width, height=self.height
            )
            self._canvas.pack()

        scale = min(self.width / W, self.height / H, 1.0)
        if scale < 1.0:
            new_W, new_H = int(W * scale), int(H * scale)
            img = img.resize((new_W, new_H), Image.Resampling.LANCZOS)

        self._photo = ImageTk.PhotoImage(img)
        self._canvas.delete("all")
        offset_x = (self.width - img.width) // 2
        offset_y = (self.height - img.height) // 2
        self._canvas.create_image(offset_x, offset_y, anchor="nw", image=self._photo)
        self._toplevel.update()

    def close(self) -> None:
        """Close the window."""
        if self._toplevel is not None:
            self._toplevel.destroy()
            self._toplevel = None


def get_window(
    title: str = "Image", width: int = 640, height: int = 480
) -> "TkImageWindow":
    """Factory function to get or create a window by title.

    Windows are cached by title - calling this function with the same
    title returns the existing window instance.

    Args:
        title: Window title (unique identifier).
        width: Initial width if creating new window.
        height: Initial height if creating new window.

    Returns:
        TkImageWindow instance for the requested title.
    """
    if title not in _WINDOWS:
        _WINDOWS[title] = TkImageWindow(title, width, height)
    return _WINDOWS[title]


def run_mainloop() -> None:
    """Block until all windows are closed."""
    root = _get_root()
    root.mainloop()
