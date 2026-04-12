"""Test CubicGrad optim_step."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np

from PIL import Image
from primitives import CubicGrad
from utils.img import ImgUtils


def test_cubicgrad_optim_step():
    """Initialize CubicGrad, prepare for optimization, run one optim_step."""
    torch.manual_seed(42)
    H, W = 1024, 1024
    patch_size = 64
    device = torch.device("cuda")

    primitive = CubicGrad(size=20)
    primitive.to(device)

    target = torch.rand(1, 3, H, W, device=device)
    primitive.prepare_for_optimization(target, patch_size=patch_size)
    primitive.eval()
    output = primitive.optim_step()

    print(f"Primitive parameters: {primitive.named_parameters()}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

    assert output.shape == (
        1,
        4,
        H,
        W,
    ), f"Expected (1, 4, {H}, {W}), got {output.shape}"
    print("Test passed!")
    rgbimg = (
        (ImgUtils.tensor2img(output.detach()) * 255)
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )
    Image.fromarray(rgbimg).show()


if __name__ == "__main__":
    test_cubicgrad_optim_step()
