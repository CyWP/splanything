from typing import List, Optional

from trainers import Trainer, EPOCH_START, EPOCH_END

from .loop_control import LoopControl


class ComfyUIControl(LoopControl):
    """ComfyUI-specific loop control with interrupt handling.

    Extends LoopControl with ComfyUI progress bar and interrupt detection.

    Stages: EPOCH_START, EPOCH_END
    """

    _stages: List[str] = [EPOCH_START, EPOCH_END]

    def __init__(self, epochs: Optional[int] = None):
        """Initialize ComfyUI loop control.

        Args:
            epochs: Optional max epochs. If None, runs until interrupted.
        """
        super().__init__(epochs)
        self._comfy_pbar = None

    def _get_comfy_pbar(self):
        """Lazy initialization of ComfyUI progress bar."""
        from utils.comfy import ComfyUtils

        if self._comfy_pbar is None and self.epochs is not None and self.epochs > 0:
            self._comfy_pbar = ComfyUtils.make_progress(self.epochs)
        return self._comfy_pbar

    def run(self, trainer: Trainer, stage: str):
        """Handle epoch start/end with ComfyUI integration.

        Args:
            trainer: Current trainer instance.
            stage: Current training stage.
        """
        from utils.comfy import ComfyUtils

        if stage == EPOCH_START:
            if ComfyUtils.is_interrupted():
                trainer.stop()
        elif stage == EPOCH_END:
            ComfyUtils.update_progress(self._get_comfy_pbar(), trainer.epoch)
