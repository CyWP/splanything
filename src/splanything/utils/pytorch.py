from typing import Sequence, Union

from jaxtyping import Bool, Integer
from torch import Tensor

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
