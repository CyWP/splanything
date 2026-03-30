import torch

from typing import Optional

from .generic import Primitive

F = torch.FloatTensor
L = torch.LongTensor
B = torch.BoolTensor


class CubicGrad(Primitive):

    def __init__(self, size: int):
        super().__init__()

    def sample(self, co: F, mask: Optional[B] = None) -> F:
        ax1 = self.R(mask) @ torch.tensor(
            [-1, 0], device=self.device, dtype=self.dtype
        ).reshape(1, 1, 2)
        ax2 = torch.stack([ax1[:, 1], -ax1[:, 0]], dim=0)

    def R(self, mask: Optional[B] = None) -> F:
        thetas = self.thetas if mask is None else self.thetas[mask]
        cos = torch.cos(thetas)
        sin = torch.sin(thetas)
        return torch.stack([cos, -sin, sin, cos], dim=0).reshape(-1, 2, 2)
