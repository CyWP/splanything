"""Base class for image-level loss functions."""

from typing import Literal, Optional

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from ...utils.img import Splimage


class Loss(nn.Module):
    """Base class for image-level loss functions operating on BCHW tensors.

    Subclasses implement ``compute`` to return a per-pixel loss map.
    The base ``forward`` applies optional spatial weighting via a
    resized ``weight_map`` and then reduces the result to a scalar.

    Subclasses own their own reference targets when needed; the base
    class makes no assumptions about or provisions for a target.

    Attributes:
        weight_map (Optional[Splimage]): Spatial weight map resized to
            match the loss map before multiplication.
        reduction (str): Reduction applied after weighting —
            ``"mean"``, ``"sum"``, or ``"none"``.

    Notes:
        - Subclasses must implement ``compute(x) -> Float[Tensor, "B C H W"]``.
        - Resizing or sampling internal targets to the rendered
          resolution is the responsibility of each subclass; the base
          only resizes ``weight_map``.
        - Loss weighting (scalar) is the responsibility of the caller;
          subclasses carry no scalar ``weight`` argument or attribute.
    """

    def __init__(
        self,
        weight_map: Optional[Splimage] = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        """Initialize the loss.

        Args:
            weight_map: Optional spatial map (Splimage) resized to the
                loss map's spatial dimensions before element-wise
                multiplication. Not premultiplied by any scalar weight.
            reduction: How to reduce the weighted loss map —
                ``"mean"`` (default), ``"sum"``, or ``"none"``.
        """
        super().__init__()
        self.weight_map = weight_map
        self.reduction = reduction

    def compute(
        self,
        x: Float[Tensor, "B C H W"],
    ) -> Float[Tensor, "B C H W"]:
        """Compute per-pixel loss map.

        Args:
            x: Model output (B, C, H, W).

        Returns:
            Unreduced loss map (B, C, H, W).
        """
        raise NotImplementedError()

    def _reduce(self, x: Tensor) -> Tensor:
        """Apply the configured reduction.

        Args:
            x: Tensor to reduce.

        Returns:
            Reduced tensor (scalar for ``mean``/``sum``, same shape for ``none``).
        """
        if self.reduction == "mean":
            return x.mean()
        if self.reduction == "sum":
            return x.sum()
        return x

    def forward(
        self,
        x: Float[Tensor, "B C H W"],
    ) -> Tensor:
        """Compute the loss, optionally weighted and reduced.

        Args:
            x: Model output (B, C, H, W).

        Returns:
            Loss value (scalar for ``mean``/``sum`` reductions,
            (B, C, H, W) for ``none``).
        """
        out = self.compute(x)
        if self.weight_map is not None:
            _, _, H, W = out.shape
            w = self.weight_map.resize(H, W).image()
            out = out * w
        return self._reduce(out)
