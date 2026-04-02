import torch

from typing import Optional

from .generic import Primitive

F = torch.FloatTensor
L = torch.LongTensor
B = torch.BoolTensor


class CubicGrad(Primitive):

    def __init__(self, size: int):
        super().__init__()

    def __len__(self) -> int:
        return self.thetas.shape[0]

    @torch.no_grad()
    def patch_mask(self, center: F, patch_size: int) -> B:
        pass

    def sample(self, co: F, mask: Optional[B] = None) -> F:
        ax1 = self.R(mask) @ torch.tensor(
            [-1, 0], device=self.device, dtype=self.dtype
        ).unsqueeze(0).expand(len(co), 2)
        ax2 = torch.stack([ax1[:, 1], -ax1[:, 0]], dim=0)
        deltas = self.centroids[mask] - co
        dot1 = (ax1 * deltas).sum(dim=1).abs()
        dot2 = (ax2 * deltas).sum(dim=1).abs()
        axmask = dot1 > dot2
        dots = dot2 / self.range2[mask]
        dots[axmask] = dot1[axmask] / self.range1[mask]
        alphas = torch.exp(-dots)
        colors = self.color2
        colors[axmask] = self.color1[axmask]
        return torch.cat([colors, alphas], dim=1)

    def R(self, mask: Optional[B] = None) -> F:
        thetas = self.thetas if mask is None else self.thetas[mask]
        cos = torch.cos(thetas)
        sin = torch.sin(thetas)
        return torch.stack([cos, -sin, sin, cos], dim=0).reshape(-1, 2, 2)
