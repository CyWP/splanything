"""L1 (absolute error) loss."""

import torch
from typing import Literal, Optional
from jaxtyping import Float
from torch import Tensor

from ...utils.img import Splimage
from .base import Loss


class L1Loss(Loss):
    """L1 (Absolute Error) loss.

    Per-pixel absolute difference between the loss's target and the
    model output. Requires a ``target``;
    it is resized to the output resolution in ``compute``.
    """

    def __init__(
        self,
        target: Splimage,
        weight_map: Optional[Splimage] = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        """Initialize the loss.

        Args:
            target: Reference image the output is compared against.
            weight_map: Optional spatial weight map.
            reduction: Reduction mode — ``"mean"``, ``"sum"``, or ``"none"``.
        """
        super().__init__(weight_map=weight_map, reduction=reduction)
        self.target = target

    def compute(
        self,
        x: Float[Tensor, "B C H W"],
    ) -> Float[Tensor, "B C H W"]:
        """Compute L1 loss map between target and output.

        The target is resized to the output resolution first.

        Args:
            x: Model output (B, C, H, W).

        Returns:
            Per-pixel absolute error (B, C, H, W).
        """
        tgt = self.target.to(x.device).resize(*x.shape[-2:]).image()
        return torch.abs(tgt - x)
