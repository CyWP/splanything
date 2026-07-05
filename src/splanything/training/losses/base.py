from typing import Optional

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from splanything.training import Trainer
from splanything.utils.img import ImgUtils


class Loss(nn.Module):
    """Base class for loss functions.

    A Loss computes a scalar value measuring the difference between
    the target image and the primitive's current output.

    Attributes:
        weight: Multiplier for the loss value in combined loss computation.
        weight_map: Spatial multiplier for the loss value in combined loss computation.

    Notes:
        - Subclasses must implement `compute(trainer) -> Float[Tensor, ""]`.
        - The `forward` method applies the weight multiplier.
    """

    def __init__(
        self,
        weight: float = 1,
        weight_map: Optional[Float[Tensor, "B 1 H W"]] = None,
        **kwargs,
    ):
        """Initialize loss.

        Args:
            weight: Weight multiplier for this loss term.
        """
        super().__init__()
        self.weight = weight
        self.weight_map = None if weight_map is None else weight * weight_map

    def compute(self, trainer: Trainer) -> Float[Tensor, ""]:
        """Compute unweighted loss value.

        Args:
            trainer: Current trainer state.

        Returns:
            Loss scalar tensor.
        """
        raise NotImplementedError()

    def forward(
        self, trainer: Trainer, co: Optional[Float[Tensor, "N 2"]] = None
    ) -> Float[Tensor, ""]:
        """Compute weighted loss.

        Args:
            trainer: Current trainer state.

        Returns:
            Weighted loss scalar.
        """
        if co is not None and self.weight_map is not None:
            weight = ImgUtils.uv_sample(self.weight_map, co)[0].squeeze(-1)
        else:
            weight = self.weight
        return self.compute(trainer) * weight
