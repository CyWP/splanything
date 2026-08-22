from typing import Optional

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from ..trainer import Trainer
from ...utils.img import Splimage


class Loss(nn.Module):
    """Base class for loss functions.

    A Loss computes a scalar value measuring the difference between
    the target image and the primitive's current output.

    Notes:
        - Subclasses must implement ``compute(trainer) -> Float[Tensor, ""]``.
        - Optional spatial weighting via ``weight_map``: when set and a
          coordinate tensor is supplied to ``forward``, the map is
          sampled at those coordinates and multiplied into the result.
        - Loss weighting (scalar) is the responsibility of the caller;
          subclasses carry no scalar ``weight`` argument or attribute.
    """

    def __init__(
        self,
        weight_map: Optional[Splimage] = None,
    ):
        """Initialize the loss.

        Args:
            weight_map: Optional spatial map (Splimage object) sampled at
                coordinates in ``forward`` to spatially weight the
                loss. Not premultiplied by any scalar weight.
        """
        super().__init__()
        self.weight_map = weight_map

    def compute(self, trainer: Trainer) -> Float[Tensor, ""]:
        """Compute unweighted loss value.

        Args:
            trainer: Current trainer state.

        Returns:
            Loss scalar tensor.
        """
        raise NotImplementedError()

    def forward(
        self,
        trainer: Trainer,
        co: Optional[Float[Tensor, "N 2"]] = None,
        **kwargs,
    ) -> Float[Tensor, ""]:
        """Compute the loss, optionally weighted by ``weight_map`` at ``co``.

        Args:
            trainer: Current trainer state.
            co: Optional coordinates used to sample ``weight_map``. When
                both ``co`` and ``weight_map`` are provided the sampled
                map multiplies the result.
            **kwargs: Accepted for call-site compatibility; unused by
                the base implementation.

        Returns:
            Loss value (scalar when ``weight_map`` is unused; otherwise
            broadcasted against the sampled weight tensor).
        """
        out = self.compute(trainer)
        if co is not None and self.weight_map is not None:
            return out * self.weight_map.mask_sample(co)[0].squeeze(-1)
        return out
