import torch

from trainers import Trainer

from .generic import Loss

F = torch.FloatTensor


class L2Loss(Loss):

    _name = "L2"

    def compute(self, trainer: Trainer) -> F:
        return ((trainer.target - trainer.last_output) ** 2).mean()
