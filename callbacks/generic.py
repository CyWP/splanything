import torch

from abc import ABC, abstractmethod
from typing import List

from trainers import STAGES, Trainer


class Callback(ABC):

    _stages: List[str] = []

    def __init__(self):
        self.stages = self.__class__._stages.copy()
        assert all([s in STAGES for s in self.stages])

    def __call__(self, trainer: Trainer, stage: str):
        if stage in self.stages:
            self.run(trainer, stage)

    @abstractmethod
    def run(self, trainer: Trainer, stage: str):
        raise NotImplementedError()
