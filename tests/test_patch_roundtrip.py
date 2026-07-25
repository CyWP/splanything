"""Round-trip tests for patch extraction and assembly.

These tests verify the contract that ``ImgUtils.extract_image_patches``
(pixels) and ``ImgUtils.extract_patches`` / ``get_patches`` (coordinates)
share an identical patch layout, and that ``ImgUtils.assemble_patches``
inverts ``extract_image_patches``.

The image-side extractor pads on the bottom/right and the coordinate-side
extractor must pad on the same side, otherwise the original content lands
in a different corner of the padded grid and row-major patch indexing
covers mismatched spatial regions.
"""

from __future__ import annotations

from pathlib import Path

import torch

from splanything.utils.img import ImgUtils

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def _load_asset(name: str) -> torch.Tensor:
    """Load an asset PNG as a (1, C, H, W) float tensor in [0, 1]."""
    return ImgUtils.load_image(str(ASSETS / name), mode="RGBA")


def _toy_image(H: int, W: int, C: int = 4) -> torch.Tensor:
    """Build an image whose pixel value encodes its (h, w, c) position.

    Each channel carries a distinct linear ramp so that any reshape that
    confuses the channel axis with a spatial axis is detectable.
    """
    img = torch.zeros(1, C, H, W)
    for c in range(C):
        for h in range(H):
            for w in range(W):
                img[0, c, h, w] = (c + 1) * 1000 + h * 100 + w
    return img


# --------------------------------------------------------------------------- #
# Shape contract
# --------------------------------------------------------------------------- #


def test_extract_image_patches_patch_count_matches_coords():
    """Number of image patches must equal number of coordinate patches.

    A classic reshape bug folds the channel axis into the patch axis,
    inflating the patch count by a factor of C (here 4) and producing
    horizontal banding with one band per channel.
    """
    H, W, ps = 64, 96, 16
    img = _toy_image(H, W, C=4)
    device = img.device

    img_patches = ImgUtils.extract_image_patches(img, patch_size=ps)  # (B, P, S, C)
    co_patches, co_centers = ImgUtils.get_patches(H, W, device, patch_size=ps)

    assert img_patches.shape[0] == 1
    P_img = img_patches.shape[1]
    P_co = co_patches.shape[0]
    assert P_img == P_co, (
        f"Patch count mismatch: image patches={P_img}, coord patches={P_co}. "
        f"Expected equal; ratio={P_img / P_co:.3f} (a ratio of C suggests the "
        f"channel axis was folded into the patch axis by a bad reshape)."
    )


def test_extract_image_patches_S_is_patch_size_squared():
    H, W, ps = 64, 96, 16
    img = _toy_image(H, W, C=4)
    img_patches = ImgUtils.extract_image_patches(img, patch_size=ps)
    assert img_patches.shape[2] == ps * ps
    assert img_patches.shape[3] == 4


# --------------------------------------------------------------------------- #
# Round trip: extract -> assemble == identity
# --------------------------------------------------------------------------- #


def test_roundtrip_divisible_size():
    """assemble(extract(img)) must reproduce img when H,W divisible by ps."""
    H, W, ps = 64, 96, 16
    img = _toy_image(H, W, C=4)
    patches = ImgUtils.extract_image_patches(img, patch_size=ps)  # (1, P, S, C)
    recon = ImgUtils.assemble_patches(patches[0], H, W)  # (1, C, H, W)
    assert recon.shape == img.shape
    assert torch.equal(recon, img)


def test_roundtrip_nondivisible_size_replicate():
    """Non-divisible size: padded edge pixels are replicated then cropped."""
    H, W, ps = 70, 90, 16  # not divisible -> pad bottom/right
    img = _toy_image(H, W, C=4)
    patches = ImgUtils.extract_image_patches(
        img, patch_size=ps, padding_mode="replicate"
    )
    recon = ImgUtils.assemble_patches(patches[0], H, W)
    assert recon.shape == img.shape
    assert torch.equal(recon, img)


def test_roundtrip_real_asset_divisible():
    img = _load_asset("bra_nor_offside.png")
    H, W = img.shape[-2:]
    ps = 16
    if H % ps or W % ps:
        H, W = (H // ps) * ps, (W // ps) * ps
        img = img[:, :, :H, :W]
    patches = ImgUtils.extract_image_patches(img, patch_size=ps)
    recon = ImgUtils.assemble_patches(patches[0], H, W)
    assert recon.shape == img.shape
    assert torch.allclose(recon, img, atol=1e-6)


def test_roundtrip_fallback_single_patch():
    """patch_size larger than both H and W -> single patch, full image."""
    H, W, C = 30, 40, 4
    img = _toy_image(H, W, C=C)
    patches = ImgUtils.extract_image_patches(img, patch_size=max(H, W) + 5)
    assert patches.shape == (1, 1, H * W, C)
    recon = ImgUtils.assemble_patches(patches[0], H, W)
    assert recon.shape == img.shape
    assert torch.equal(recon, img)


# --------------------------------------------------------------------------- #
# Coordinate / pixel alignment
# --------------------------------------------------------------------------- #


def _coord_to_pixel_idx(coord: torch.Tensor, dim: int) -> torch.Tensor:
    """Map a normalized pixel-center coordinate back to an integer index.

    ``gen_px_coords`` centers pixels: coord = (idx + 0.5) / dim, so
    idx = round(coord * dim - 0.5).
    """
    return (coord * dim - 0.5).round().long().clamp(0, dim - 1)


def test_coord_patch_aligns_with_image_patch_divisible():
    """For every patch, the coordinates from get_patches must index the same
    spatial pixels that extract_image_patches places in that patch."""
    H, W, ps = 64, 96, 16
    img = _toy_image(H, W, C=4)
    device = img.device

    ImgUtils.extract_image_patches(img, patch_size=ps)  # asserts no-throw + layout
    co_patches, _ = ImgUtils.get_patches(H, W, device, patch_size=ps)  # (P, S, 2)
    P, S, _ = co_patches.shape

    PW = W // ps
    for p in range(P):
        pi, pj = p // PW, p % PW
        for s in range(S):
            yi = _coord_to_pixel_idx(co_patches[p, s, 0], H)
            xi = _coord_to_pixel_idx(co_patches[p, s, 1], W)
            local_h = pi * ps + (s // ps)
            local_w = pj * ps + (s % ps)
            for c in range(4):
                expected = (c + 1) * 1000 + local_h * 100 + local_w
                got = img[0, c, yi, xi].item()
                assert got == expected, (
                    f"mismatch p={p} s={s} patch=({pi},{pj}) "
                    f"coord_idx=({yi},{xi}) local=({local_h},{local_w}) "
                    f"c={c} got={got} expected={expected}"
                )


def test_coord_patch_aligns_with_image_patch_nondivisible():
    """Alignment must also hold for non-divisible sizes (padding side matters)."""
    H, W, ps = 70, 90, 16
    img = _toy_image(H, W, C=4)
    device = img.device

    img_patches = ImgUtils.extract_image_patches(
        img, patch_size=ps, padding_mode="replicate"
    )[0]
    co_patches, _ = ImgUtils.get_patches(H, W, device, patch_size=ps)
    P, S, _ = co_patches.shape
    assert img_patches.shape[0] == P

    import math

    PW = math.ceil(W / ps)
    mismatches = 0
    for p in range(P):
        pi, pj = p // PW, p % PW
        for s in range(S):
            yi = _coord_to_pixel_idx(co_patches[p, s, 0], H)
            xi = _coord_to_pixel_idx(co_patches[p, s, 1], W)
            local_h = min(pi * ps + (s // ps), H - 1)
            local_w = min(pj * ps + (s % ps), W - 1)
            for c in range(4):
                expected = (c + 1) * 1000 + local_h * 100 + local_w
                got = img[0, c, yi, xi].item()
                if got != expected:
                    mismatches += 1
    assert mismatches == 0, (
        f"{mismatches} coord/pixel mismatches for non-divisible size"
    )
