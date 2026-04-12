import torch

from typing import Optional, Iterator, Union
from jaxtyping import Float
from torch import Tensor

from primitives import Primitive

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
    """

    def __init__(
        self,
        primitive: Primitive,
        H: int,
        W: int,
        patch_size: int,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Initialize generator.

        Args:
            primitive: Pretrained primitive for generation.
            H: Desired output height.
            W: Desired output width.
            patch_size: Patch size for rasterization.
            device: Optional device override. Uses primitive device if None.
        """
        self.primitive = primitive
        self.H = H
        self.W = W
        self.patch_size = patch_size
        self.device = device or primitive.device
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

    def generate(self) -> Float[Tensor, "B C H W"]:
        """Generate image from primitive at specified resolution.

        Returns:
            Generated image tensor (B, C, H, W).
        """
        return self.primitive.rasterize(self.H, self.W, patch_size=self.patch_size)

    def to_image(self) -> Float[Tensor, "B H W C"]:
        """Generate and convert to displayable image.

        Returns:
            Image tensor (B, H, W, C) in [0, 1] range.
        """
        return self.primitive.image(self.H, self.W, patch_size=self.patch_size)

    def gen(self) -> Iterator[dict]:
        """Generator for step-by-step generation.

        Yields state dicts at each stage (GEN_START, GEN_END).

        Yields:
            Dict with keys:
                - image: Generated image tensor
                - H: Output height
                - W: Output width
        """
        self.primitive.eval()
        yield {"stage": GEN_START, "H": self.H, "W": self.W}
        img = self.generate()
        yield {"stage": GEN_END, "image": img, "H": self.H, "W": self.W}
