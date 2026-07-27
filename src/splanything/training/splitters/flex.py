from __future__ import annotations
from torch import Tensor
from jaxtyping import Float
from typing import Callable, Tuple, TYPE_CHECKING

from .base import Splitter

if TYPE_CHECKING:
    from ...primitives.base import Primitive


class FlexibleSplitter(Splitter):
    def __init__(
        self,
        split_func: Callable[
            [str, Primitive, Float[Tensor, "N_split ..."]],
            Tuple[Float[Tensor, "N_split ..."], Float[Tensor, "N_split ..."]],
        ],
    ):
        self.split_func = split_func

    def split_vals(
        self, name: str, primitive: Primitive, split_param: Float[Tensor, "N_split ..."]
    ) -> Tuple[Float[Tensor, "N_split ..."], Float[Tensor, "N_split ..."]]:
        return self.split_func(name, primitive, split_param)
