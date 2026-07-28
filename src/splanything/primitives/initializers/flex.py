from typing import Callable, Tuple
from torch import Tensor
from jaxtyping import Float

from .base import Initializer


class FlexibleInitializer(Initializer):
    def __init__(self, func: Callable[[str, Tuple[int]], Float[Tensor, "Size ..."]]):
        self.func = func

    def init_param(
        self, name: str, param_shape: Tuple[int], batched: bool
    ) -> Float[Tensor, "Size ..."]:
        return func(name, param_shape)
