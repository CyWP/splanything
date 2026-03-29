import torch
import torch.n as nn

from trainers import Trainer

F = torch.FloatTensor


class Loss(nn.Module):

    def __init__(self, weight: float, **kwargs):
        self.weight = weight

    def compute(self, trainer: Trainer) -> F:
        raise NotImplementedError()

    def forward(self, trainer: Trainer) -> F:
        return self.compute * self.weight
