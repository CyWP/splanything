from typing import Literal, Optional

from jaxtyping import Float
from torch import Tensor

from ...utils.img import ImgUtils, Splimage
from .base import ImageLoss


class SSIMImageLoss(ImageLoss):
    """Structural Similarity Index (SSIM) image loss.

    Computes ``1 - SSIM`` per-pixel, where SSIM ∈ [-1, 1].
    Loss ∈ [0, 2] (0 = identical).

    Attributes:
        kernel_size (int): Gaussian kernel side length.
        sigma (float): Gaussian kernel standard deviation.

    Notes:
        - The Gaussian kernel is registered as a buffer so it moves
          with the module's device automatically.
    """

    def __init__(
        self,
        weight_map: Optional[Splimage] = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
        kernel_size: int = 11,
        sigma: float = 1.5,
    ):
        """Initialize the loss.

        Args:
            weight_map: Optional spatial weight map.
            reduction: Reduction mode — ``"mean"``, ``"sum"``, or ``"none"``.
            kernel_size: Gaussian kernel side length.
            sigma: Gaussian kernel standard deviation.
        """
        super().__init__(weight_map=weight_map, reduction=reduction)
        self.kernel_size = kernel_size
        self.sigma = sigma
        kernel = ImgUtils.gaussian_kernel(kernel_size, [sigma, sigma])
        self.register_buffer("kernel", kernel)

    def compute(
        self,
        x: Float[Tensor, "B C H W"],
        target: Float[Tensor, "B C H W"],
    ) -> Float[Tensor, "B C H W"]:
        """Compute SSIM loss map between output and target.

        Args:
            x: Model output (B, C, H, W).
            target: Ground truth target (B, C, H, W).

        Returns:
            Per-pixel ``1 - SSIM`` (B, C, H, W).
        """
        ssim_map = ImgUtils.SSIM(x, target, self.kernel)
        return 1.0 - ssim_map
