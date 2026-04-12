from jaxtyping import Float
from torch import Tensor
from typing import Protocol, runtime_checkable, Optional

from utils.pytorch import TensorIndex


@runtime_checkable
class Splittable(Protocol):
    """Protocol for primitives that support splitting.

        Primitives implementing this protocol can be split during refinement
    to increase detail in high-gradient regions.

        Methods:
            split: Split primitives at given indices into smaller elements.
    """

    def split(self, idx: TensorIndex) -> None:
        """Split primitives at indices into smaller elements.

        Args:
            idx: Boolean or integer index selecting primitives to split.
            optimizer: optimizer to which new parameters can be added (optional).
        """
        pass


@runtime_checkable
class HasAreas(Protocol):
    """Protocol for primitives with area properties.

    Primitives implementing this protocol expose an areas property
    used by refinement rules to normalize gradient magnitudes.

    Properties:
        areas: Per-primitive area values.
    """

    @property
    def areas(self) -> Float[Tensor, "N"]:
        """Per-primitive areas.

        Returns:
            Tensor of shape (N,) with area values.
        """
        pass


@runtime_checkable
class HasAlphas(Protocol):
    """Protocol for primitives with alpha values.

    Primitives implementing this protocol expose alphas attribute
    used by refinement rules for transparency-based culling.

    Attributes:
        alphas: Per-primitive opacity values.
    """

    alphas: Float[Tensor, "N"]
    """Per-primitive alpha/opacity values of shape (N,)."""


@runtime_checkable
class HasScales(Protocol):
    """Protocol for primitives with scale values.

    Primitives implementing this protocol expose a scales property
    returning the two principal scales (e.g., sigma_1, sigma_2 or range_1, range_2).

    Properties:
        scales: Tuple of two per-primitive scale tensors (N,).
    """

    @property
    def scales(self) -> tuple[Float[Tensor, "N"], Float[Tensor, "N"]]:
        """Per-primitive scales.

        Returns:
            Tuple of two tensors (N,) representing the principal scales.
        """
        pass
