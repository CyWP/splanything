import torch

from trainers import Trainer

from .generic import Loss

F = torch.FloatTensor


class L1Loss(Loss):

    _name = "L1"

    def compute(self, trainer: Trainer) -> F:
        return torch.abs(trainer.target - trainer.last_output).mean()
