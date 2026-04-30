"""Lightweight tkinter-based image window utilities."""

from __future__ import annotations
import tkinter as tk

from PIL import Image, ImageTk
from typing import Optional, Dict

import numpy as np


_WINDOWS: Dict[str, TkImageWindow] = {}
_ROOT: Optional[tk.Tk] = None


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
) -> TkImageWindow:
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
