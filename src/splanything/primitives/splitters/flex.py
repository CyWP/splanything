"""Callable-based splitter."""

from __future__ import annotations
from torch import Tensor
from jaxtyping import Float
from typing import Callable, Tuple, TYPE_CHECKING

from .base import Splitter

if TYPE_CHECKING:
    from ...primitives.base import Primitive


class FlexibleSplitter(Splitter):
    """Splitter delegating to a user-provided callable."""

    def __init__(
        self,
        split_func: Callable[
            [str, Primitive, Float[Tensor, "N_split ..."]],
            Tuple[Float[Tensor, "N_split ..."], Float[Tensor, "N_split ..."]],
        ],
    ):
        """Initialize the splitter.

        Args:
            split_func: Callable (name, primitive, split_param) ->
                (vals_1, vals_2).
        """
        self.split_func = split_func

    def split_vals(
        self, name: str, primitive: Primitive, split_param: Float[Tensor, "N_split ..."]
    ) -> Tuple[Float[Tensor, "N_split ..."], Float[Tensor, "N_split ..."]]:
        """Compute child parameter values via the wrapped callable.

        Args:
            name: Parameter name being split.
            primitive: Primitive being split, with the split instances
                masked in.
            split_param: Parameter values of the instances being split
                (N_split, ...).

        Returns:
            Tuple of two tensors (N_split, ...).
        """
        return self.split_func(name, primitive, split_param)
