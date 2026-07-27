from typing import Sequence

from jaxtyping import Bool, Integer
from torch import Tensor

# Useful type
TensorIndex1D = (
    int
    | slice
    | type(...)
    | None
    | Bool[Tensor, "N"]
    | Integer[Tensor, "N"]
    | Sequence[int]
)
