import torch

from typing import Optional, Iterator, Union, Dict
from jaxtyping import Float
from torch import Tensor

from PIL.Image import Image
from primitives import Primitive
from rasterizers import Rasterizer, WeightedRasterizer
from utils.img import ImgUtils

GEN_START = "gen_start"
GEN_END = "gen_end"
GEN_STAGES = [GEN_START, GEN_END]


class Generator:
    """Image generator from pretrained primitive.

    Generates images at specified resolution using a pretrained primitive.
    Supports yielding generation state for step-by-step execution.

    Attributes:
        primitive: The pretrained primitive for generation.
        H: Output image height.
        W: Output image width.
        patch_size: Patch size for rasterization.
        device: Device to run generation on.
        rasterizer: Rasterizer for sample to image output.
    """

    def __init__(
        self,
        H: int,
        W: int,
        patch_size: int,
        rasterizer: Optional[Rasterizer] = None,
    ):
        """Initialize generator.

        Args:
            H: Desired output height.
            W: Desired output width.
            patch_size: Patch size for rasterization.
            rasterizer: Optional rasterizer override.
        """
        self.H = H
        self.W = W
        self.patch_size = patch_size
        self.rasterizer = rasterizer or WeightedRasterizer

    def __call__(self, primitive: Primitive) -> Float[Tensor, "B C H W"]:
        """Generate image from primitive at specified resolution.

        Returns:
            Generated image tensor (B, C, H, W).
        """
        patches, centers = ImgUtils.get_patches(
            self.H, self.W, primitive.device, patch_size=self.patch_size
        )
        return primitive(self.H, self.W, patches, centers, self.rasterizer)
