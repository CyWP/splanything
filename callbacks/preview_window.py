import logging
import torch
import tkinter

from typing import List

from trainers import Trainer, EPOCH_END
from utils.img import ImgUtils
from utils.tkinter import get_window

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

    def run(self, trainer: Trainer, stage: str):
        """Update preview window if epoch matches frequency.

        Args:
            trainer: Current trainer instance.
            stage: Current training stage.
        """
        if trainer.epoch % self.frequency != 0:
            return

        img = trainer.last_output.clone().detach()
        if self.show_target:
            img = torch.cat([img, trainer.target], dim=3)
        window = get_window(self.window_title)
        try:
            window.update_image(ImgUtils.tensor2pil(img, normalized=False))
        except tkinter.TclError:
            _logger.info("Preview closed, stopping training.")
            trainer.stop()
