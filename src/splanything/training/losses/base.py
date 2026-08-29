from typing import Literal, Optional

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from ...utils.img import Splimage


class Loss(nn.Module):
    """Base class for per-sample loss functions.

    A Loss computes a scalar value measuring the difference between
    a target and the model's output.

    Attributes:
        weight_map (Optional[Splimage]): Spatial weight map sampled at
            coordinates in ``forward`` to spatially weight the loss.

    Notes:
        - Subclasses must implement ``compute(x, target) -> Float[Tensor, ""]``.
        - Loss weighting (scalar) is the responsibility of the caller;
          subclasses carry no scalar ``weight`` argument or attribute.
    """

    def __init__(
        self,
        weight_map: Optional[Splimage] = None,
    ):
        """Initialize the loss.

        Args:
            weight_map: Optional spatial map (Splimage) sampled at
                coordinates in ``forward`` to spatially weight the
                loss. Not premultiplied by any scalar weight.
        """
        super().__init__()
        self.weight_map = weight_map

    def compute(
        self,
        x: Float[Tensor, "..."],
        target: Float[Tensor, "..."],
    ) -> Float[Tensor, ""]:
        """Compute unweighted loss value.

        Args:
            x: Model output.
            target: Ground truth target.

        Returns:
            Loss scalar tensor.
        """
        raise NotImplementedError()

    def forward(
        self,
        x: Float[Tensor, "..."],
        target: Float[Tensor, "..."],
        co: Optional[Float[Tensor, "N 2"]] = None,
        **kwargs,
    ) -> Float[Tensor, ""]:
        """Compute the loss, optionally weighted by ``weight_map`` at ``co``.

        Args:
            x: Model output.
            target: Ground truth target.
            co: Optional coordinates used to sample ``weight_map``. When
                both ``co`` and ``weight_map`` are provided the sampled
                map multiplies the result.
            **kwargs: Accepted for call-site compatibility; unused.

        Returns:
            Loss value (scalar when ``weight_map`` is unused; otherwise
            broadcasted against the sampled weight tensor).
        """
        out = self.compute(x, target)
        if co is not None and self.weight_map is not None:
            return out * self.weight_map.mask_sample(co)[0].squeeze(-1)
        return out


class ImageLoss(nn.Module):
    """Base class for image-level loss functions operating on BCHW tensors.

    Subclasses implement ``compute`` to return a per-pixel loss map.
    The base ``forward`` applies optional spatial weighting via a
    resized ``weight_map`` and then reduces the result to a scalar.

    Attributes:
        weight_map (Optional[Splimage]): Spatial weight map resized to
            match the loss map before multiplication.
        reduction (str): Reduction applied after weighting —
            ``"mean"``, ``"sum"``, or ``"none"``.

    Notes:
        - Subclasses must implement
          ``compute(x, target) -> Float[Tensor, "B C H W"]``.
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
        target: Float[Tensor, "B C H W"],
    ) -> Float[Tensor, "B C H W"]:
        """Compute per-pixel loss map.

        Args:
            x: Model output (B, C, H, W).
            target: Ground truth target (B, C, H, W).

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
        target: Float[Tensor, "B C H W"],
        co: Optional[Float[Tensor, "N 2"]] = None,
        **kwargs,
    ) -> Tensor:
        """Compute the loss, optionally weighted and reduced.

        Args:
            x: Model output (B, C, H, W).
            target: Ground truth target (B, C, H, W).
            co: Accepted for call-site compatibility; unused by
                image-level losses.
            **kwargs: Accepted for call-site compatibility; unused.

        Returns:
            Loss value (scalar for ``mean``/``sum`` reductions,
            (B, C, H, W) for ``none``).
        """
        out = self.compute(x, target)
        if self.weight_map is not None:
            _, _, H, W = out.shape
            w = self.weight_map.resize(H, W).image()
            out = out * w
        return self._reduce(out)
