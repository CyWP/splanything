"""Tkinter preview window callback and image-window utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image

from ...rendering.sampler import Sampler
from ...utils.img import Splimage
from ..trainer import Trainer
from ..stages import EPOCH_END
from .base import Callback

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
        sampler: Optional[Sampler] = None,
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
            sampler: Optional Sampler used to rasterize the preview image.
                If None, the trainer's sampler is used.
            H: Optional height to resize the preview image to.
            W: Optional width to resize the preview image to.
            max_batch: Max batch size passed to the sampler's rasterize.
            low_vram: Low-VRAM flag passed to the sampler's rasterize.
            save_folder: Optional folder to save preview PNGs to.
        """
        super().__init__()
        self.frequency = max(frequency, 1)
        self.show_target = show_target
        self.window_title = window_title
        self.sampler = sampler
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
        with torch.no_grad():
            img = Splimage(
                trainer.last_image(
                    max_batch=self.max_batch,
                    low_vram=self.low_vram,
                    sampler=self.sampler,
                )
            )
        if self.H is not None and self.W is not None:
            cur_H, cur_W = img.shape[-2:]
            if cur_H != self.H or cur_W != self.W:
                img = img.resize(self.H, self.W)
        if self.show_target:
            tgt_img = trainer.sampler.target_img
            t_H, t_W = tgt_img.shape[-2:]
            i_H, i_W = img.shape[-2:]
            if t_H != i_H or t_W != i_W:
                tgt_img = tgt_img.resize(i_H, i_W)
            img = img.stack(tgt_img, dim="W")
        window = get_window(self.window_title)
        pil_img = img.to_pil()
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
    """A window that displays images using tkinter, resizing with the user.

    The displayed image is re-fit to the current canvas size on every
    ``update_image`` call and on every ``<Configure>`` window resize event,
    preserving aspect ratio and centering within the window. Both upscaling
    and downscaling are supported. The most recent PIL image is cached so
    resize events can re-render without a new ``update_image`` call.

    Attributes:
        title: Window title (unique identifier).
        width: Fallback width used only until the canvas is realised.
        height: Fallback height used only until the canvas is realised.

    Notes:
        - The canvas is packed with ``fill="both", expand=True`` so it
          tracks the toplevel's size.
        - ``<Configure>`` events are bound on the canvas; filtering by
          ``widget`` avoids spurious redraws from child widgets.
    """

    def __init__(self, title: str = "Image", width: int = 640, height: int = 480):
        """Configure the window (created lazily on first ``update_image``).

        Args:
            title: Window title (unique identifier).
            width: Fallback width used until the canvas is realised.
            height: Fallback height used until the canvas is realised.
        """
        self.title = title
        self.width = width
        self.height = height
        self._toplevel = None
        self._canvas = None
        self._photo = None
        self._img: Optional[Image.Image] = None

    def _canvas_size(self) -> tuple[int, int]:
        """Return the current canvas (visible) size in pixels.

        Falls back to ``self.width``/``self.height`` when tkinter reports
        an unrealised widget (``winfo_width`` <= 1).
        """
        w = self._canvas.winfo_width() if self._canvas is not None else 0
        h = self._canvas.winfo_height() if self._canvas is not None else 0
        if w <= 1:
            w = self.width
        if h <= 1:
            h = self.height
        return max(w, 1), max(h, 1)

    def _render(self) -> None:
        """Fit and draw the cached image to the current canvas size."""
        from PIL import ImageTk

        if self._img is None or self._canvas is None:
            return
        cw, ch = self._canvas_size()
        W, H = self._img.size
        if W == 0 or H == 0:
            return
        scale = min(cw / W, ch / H)
        new_W = max(1, int(W * scale))
        new_H = max(1, int(H * scale))
        resized = self._img.resize((new_W, new_H), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        self._canvas.delete("all")
        offset_x = (cw - new_W) // 2
        offset_y = (ch - new_H) // 2
        self._canvas.create_image(offset_x, offset_y, anchor="nw", image=self._photo)

    def update_image(self, img: Image.Image) -> None:
        """Update the displayed image, fitting it to the current window.

        Args:
            img: PIL image to display. Resized to fit within the current
                canvas bounds preserving aspect ratio, centered, and
                re-rendered automatically on subsequent resize events.
        """
        import tkinter as tk

        if self._toplevel is None:
            root = _get_root()
            self._toplevel = tk.Toplevel(root)
            self._toplevel.title(self.title)
            self._canvas = tk.Canvas(
                self._toplevel, width=self.width, height=self.height
            )
            self._canvas.pack(fill="both", expand=True)
            self._canvas.bind(
                "<Configure>",
                lambda e: self._on_configure(e),
            )
        self._img = img
        self._render()
        self._toplevel.update_idletasks()
        self._toplevel.update()

    def _on_configure(self, event) -> None:
        """Re-render on canvas resize events.

        Args:
            event: tkinter ``<Configure>`` event. Only handled when it
                comes from the canvas itself (not a child widget) to
                avoid redundant redraws.
        """
        if event.widget is not self._canvas:
            return
        self.width = max(event.width, 1)
        self.height = max(event.height, 1)
        self._render()

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
