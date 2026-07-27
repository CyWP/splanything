from __future__ import annotations
import torch
from torch import Tensor
from jaxtyping import Float, Bool
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ...primitives.base import Primitive


class Splitter:
    def split_vals(
        self, name: str, primitive: Primitive, split_param: Float[Tensor, "N_split ..."]
    ) -> Tuple[Float[Tensor, "N_split ..."], Float[Tensor, "N_split ..."]]:
        if "centroid" in name:
            disp = torch.rand_like(split_param) * 2 - 1
            disp *= (
                (
                    primitive.areas
                    if hasattr(primitive, "areas")
                    else torch.full_like(split_param[:, 0], (1 / len(primitive)))
                )
                ** 0.5
                / (4 * disp.norm())
            )[:, None]
            return split_param + disp, split_param - disp
        if any([n in name for n in ("scale", "range", "sigma")]):
            new = split_param / 2**0.5
            return new, new
        return split_param, split_param

    def __call__(
        self,
        primitive: Primitive,
        name: str,
        param: Float[Tensor, "N ..."],
        split_mask: Bool[Tensor, "N"],
    ) -> Float[Tensor, "N+N_split ..."]:
        with primitive.masked(split_mask):
            split_vals_1, split_vals_2 = self.split_vals(
                name, primitive, param[split_mask]
            )
        p = param.clone()
        p[split_mask] = split_vals_1
        return torch.cat([p, split_vals_2], dim=0)
