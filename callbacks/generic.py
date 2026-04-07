import torch

from abc import ABC, abstractmethod
from typing import List, Optional, Any

from trainers import STAGES, Trainer


class Callback(ABC):
    """Base class for training callbacks.

    Callbacks hook into the training lifecycle to perform monitoring,
    logging, or control actions at specific stages.

    Attributes:
        stages: List of stages this callback responds to.
        node: Optional ComfyUI node reference for UI updates.

    Notes:
        - Subclasses must define `_stages` as a class attribute.
        - Subclasses must implement `run(trainer, stage)`.
    """

    _stages: List[str] = []

    def __init__(self, node: Optional[Any] = None):
        """Initialize callback.

        Args:
            node: Optional ComfyUI node for preview updates.
        """
        self.stages = self.__class__._stages.copy()
        assert all([s in STAGES for s in self.stages])
        self.node = node

    def __call__(self, trainer: Trainer, stage: str):
        """Invoke callback if stage matches.

        Args:
            trainer: Current trainer instance.
            stage: Current training stage.
        """
        if stage in self.stages:
            self.run(trainer, stage)

    @abstractmethod
    def run(self, trainer: Trainer, stage: str):
        """Execute callback logic.

        Args:
            trainer: Current trainer instance.
            stage: Current training stage.
        """
        raise NotImplementedError()
