import torch
from torch import Tensor
from typing import Tuple
from jaxtyping import Float


class Initializer:
    def init_param(
        self, name: str, param_shape: Tuple[int], batched: bool
    ) -> Float[Tensor, "Size ..."]:
        area_factor = 1 / size**0.5
        if any([c in name for c in ("theta", "angle")]):
            return torch.rand(param_shape) * 2 * torch.pi
        if any([c in name for c in ("centroid",)]):
            return torch.rand(param_shape)
        if any([c in name for c in ("alpha", "transparency")]):
            return 1 - (torch.rand(param_shape) * 2 / 3) ** 2
        if any([c in name for c in ("scale", "range", "sigma")]):
            return (torch.randn(param_shape)) * area_factor
        if any([c in name for c in ("axis",)]):
            p = torch.rand(param_shape)
            return p / p.norm()
        if any([c in name for c in ("color", "red", "green", "blue")]):
            return torch.rand(param_shape)
        return torch.randn(param_shape)

    def __call__(
        self, name: str, size: int, channels: Optional[Tuple[int]] = None
    ) -> Float[Tensor, "Size ..."]:
        if channels is not None:
            if any([c < 0 for c in channels]):
                raise ValueError(
                    f"Value '{channels}' for channels is invalid. Must be >=0."
                )
        if size < 0:
            raise ValueError(
                f"Value '{size}' for size is invalid. Must be 0 for non batched params and >=1 for batched params."
            )
        if size == 0:
            param_shape = () if channels is None else channels
        else:
            param_shape = (size,) if channels is None else (size, *channels)
        return self.init_param(name, param_shape)
