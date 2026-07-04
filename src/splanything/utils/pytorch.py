import torch

from jaxtyping import Bool, Integer
from torch import Tensor
from typing import Union, Sequence

# Useful type
TensorIndex1D = Union[
    int,
    slice,
    type(...),  # Ellipsis
    None,
    Bool[Tensor, "N"],
    Integer[Tensor, "N"],
    Sequence[int],
]
