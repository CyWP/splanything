"""Callable-based parameter initializer."""

from typing import Callable, Tuple
from torch import Tensor
from jaxtyping import Float

from .base import Initializer


class FlexibleInitializer(Initializer):
    """Initializer delegating to a user-provided callable."""

    def __init__(self, func: Callable[[str, Tuple[int]], Float[Tensor, "Size ..."]]):
        """Initialize the initializer.

        Args:
            func: Callable (name, param_shape) -> initialized tensor.
        """
        self.func = func

    def init_param(
        self, name: str, param_shape: Tuple[int], batched: bool
    ) -> Float[Tensor, "Size ..."]:
        """Initialize a parameter tensor via the wrapped callable.

        Args:
            name: Parameter name.
            param_shape: Shape of the parameter tensor.
            batched: Whether the parameter has a batch dimension (unused).

        Returns:
            Initialized tensor.
        """
        return self.func(name, param_shape)
