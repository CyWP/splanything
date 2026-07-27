from torch import Tensor
from jaxtyping import Float
from typing import Callable, Tuple

from ...primitives.base import Primitive
from .base import Splitter


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
