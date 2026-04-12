"""Tests for patch extraction and assembly."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import torch
from utils.img import ImgUtils


def test_assemble_preserves_patch_layout():
    """Each patch gets a unique uniform color. After assembly, verify each region has the right color."""
    H, W = 32, 32
    patch_size = 8
    device = torch.device("cuda")
    C = 3

    patches, centers = ImgUtils.get_patches(H, W, device, patch_size)
    P, S, _ = patches.shape

    colors = torch.arange(1, P + 1, dtype=torch.float32).unsqueeze(1).unsqueeze(2)
    sampled = colors.expand(P, S, C)

    assembled = ImgUtils.assemble_patches(sampled, H, W)
    assert assembled.shape == (1, C, H, W)

    patches_H = H // patch_size
    patches_W = W // patch_size
    for py in range(patches_H):
        for px in range(patches_W):
            pidx = py * patches_W + px
            expected_color = float(pidx + 1)
            y0, y1 = py * patch_size, (py + 1) * patch_size
            x0, x1 = px * patch_size, (px + 1) * patch_size
            region = assembled[0, :, y0:y1, x0:x1]
            assert torch.allclose(
                region, torch.tensor(expected_color)
            ), f"Patch ({py},{px}) expected {expected_color}, got {region[0,0,0].item()}"


def test_assemble_preserves_patch_layout_non_divisible():
    """Test patch layout with non-divisible dimensions (padding trimmed)."""
    H, W = 30, 30
    patch_size = 8
    device = torch.device("cuda")
    C = 3

    patches, centers = ImgUtils.get_patches(H, W, device, patch_size)
    P, S, _ = patches.shape

    colors = torch.arange(1, P + 1, dtype=torch.float32).unsqueeze(1).unsqueeze(2)
    sampled = colors.expand(P, S, C)

    assembled = ImgUtils.assemble_patches(sampled, H, W)
    assert assembled.shape == (1, C, H, W)

    patches_H = math.ceil(H / patch_size)
    patches_W = math.ceil(W / patch_size)

    for py in range(patches_H):
        for px in range(patches_W):
            pidx = py * patches_W + px
            expected_color = float(pidx + 1)
            y0 = py * patch_size
            y1 = min((py + 1) * patch_size, H)
            x0 = px * patch_size
            x1 = min((px + 1) * patch_size, W)
            region = assembled[0, :, y0:y1, x0:x1]
            assert torch.allclose(
                region, torch.tensor(expected_color)
            ), f"Patch ({py},{px}) expected {expected_color}, got {region[0,0,0].item()}"


def test_coordinate_spatial_order():
    """Test that patch coordinates maintain correct spatial ordering."""
    H, W = 8, 8
    patch_size = 4
    device = torch.device("cuda")

    co = ImgUtils.gen_px_coords(H, W, device)
    patches, centers = ImgUtils.extract_patches(co, patch_size)

    p = patches[0].reshape(patch_size, patch_size, 2)

    y_vals = p[:, :, 0]
    for row in range(patch_size):
        assert y_vals[row, :].unique().numel() == 1, f"Row {row} y values not uniform"
        if row > 0:
            assert y_vals[row, 0] > y_vals[row - 1, 0], "y values not increasing"

    x_vals = p[:, :, 1]
    for col in range(patch_size):
        assert x_vals[:, col].unique().numel() == 1, f"Col {col} x values not uniform"
        if col > 0:
            assert x_vals[0, col] > x_vals[0, col - 1], "x values not increasing"


def visualize_assembled_patches():
    """Show assembled patch grid in a tkinter window for visual verification."""
    import numpy as np
    from utils.tkinter import get_window

    H, W = 32, 32
    patch_size = 8
    device = torch.device("cuda")
    C = 3

    patches, centers = ImgUtils.get_patches(H, W, device, patch_size)
    P, S, _ = patches.shape

    colors = torch.arange(1, P + 1, dtype=torch.float32).unsqueeze(1).unsqueeze(2)
    sampled = colors.expand(P, S, C)

    assembled = ImgUtils.assemble_patches(sampled, H, W)
    img_np = assembled[0].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np / img_np.max() * 255).astype(np.uint8)

    window = get_window("Patch Assembly Test", W * 16, H * 16)
    window.update_image(img_np)


def visualize_image_sampling_roundtrip():
    """Create random image, sample at patch coordinates, reassemble, visualize vs original."""
    import numpy as np
    import torch.nn.functional as F
    from utils.tkinter import get_window

    torch.manual_seed(42)
    H, W = 64, 64
    patch_size = 8
    device = torch.device("cuda")
    C = 3

    img = torch.rand(1, C, H, W, device=device)

    patches_coords, centers = ImgUtils.get_patches(H, W, device, patch_size)
    P, S, _ = patches_coords.shape

    coords = patches_coords.reshape(1, P * S, 2)
    grid = coords.unsqueeze(0) * 2 - 1

    sampled = F.grid_sample(img, grid, align_corners=False, mode="bilinear")
    sampled = sampled.squeeze(0).squeeze(1).t().reshape(P, S, C)

    assembled = ImgUtils.assemble_patches(sampled, H, W)

    orig_np = (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    recon_np = (assembled[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    equals = orig_np == recon_np
    if not np.all(equals):
        breakpoint

    side_by_side = np.concatenate([orig_np, recon_np], axis=1)

    window = get_window("Original (left) vs Reconstructed (right)", W * 2 * 4, H * 4)
    window.update_image(side_by_side)


if __name__ == "__main__":
    test_assemble_preserves_patch_layout()
    test_assemble_preserves_patch_layout_non_divisible()
    test_coordinate_spatial_order()
    print("All tests passed!")
    visualize_assembled_patches()
    visualize_image_sampling_roundtrip()

    from utils.tkinter import run_mainloop

    run_mainloop()
