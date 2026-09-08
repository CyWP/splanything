"""SSIM loss."""

from typing import Literal, Optional
from jaxtyping import Float
from torch import Tensor

from ...utils.img import ImgUtils, Splimage
from .base import Loss


class SSIMLoss(Loss):
    """Structural Similarity Index (SSIM) loss.

    Computes ``1 - SSIM`` per-pixel, where SSIM ∈ [-1, 1] against the
    loss's own target. Loss ∈ [0, 2] (0 = identical). The target is
    resized to the output resolution in ``compute``.

    Attributes:
        kernel_size (int): Gaussian kernel side length.
        sigma (float): Gaussian kernel standard deviation.

    Notes:
        - The Gaussian kernel is registered as a buffer so it moves
          with the module's device automatically.
    """

    def __init__(
        self,
        target: Splimage,
        weight_map: Optional[Splimage] = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
        kernel_size: int = 11,
        sigma: float = 1.5,
    ):
        """Initialize the loss.

        Args:
            target: Reference image the output is compared against.
            weight_map: Optional spatial weight map.
            reduction: Reduction mode — ``"mean"``, ``"sum"``, or ``"none"``.
            kernel_size: Gaussian kernel side length.
            sigma: Gaussian kernel standard deviation.
        """
        super().__init__(weight_map=weight_map, reduction=reduction)
        self.target = target
        self.kernel_size = kernel_size
        self.sigma = sigma
        kernel = ImgUtils.gaussian_kernel(kernel_size, [sigma, sigma])
        self.register_buffer("kernel", kernel)

    def compute(
        self,
        x: Float[Tensor, "B C H W"],
    ) -> Float[Tensor, "B C H W"]:
        """Compute SSIM loss map between target and output.

        The target is resized to the output resolution first.

        Args:
            x: Model output (B, C, H, W).

        Returns:
            Per-pixel ``1 - SSIM`` (B, C, H, W).
        """
        tgt = self.target.to(x.device).resize(*x.shape[-2:]).image()
        ssim_map = ImgUtils.SSIM(x, tgt, self.kernel)
        return 1.0 - ssim_map
