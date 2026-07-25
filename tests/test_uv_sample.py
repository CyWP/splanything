"""Tests for ``ImgUtils.uv_sample`` pixel-center alignment.

``uv_sample`` must sample at the same normalized locations produced by
``ImgUtils.gen_px_coords`` and ``primitive.centroids`` (both use the
pixel-center convention: coord ``i = (i + 0.5) / dim``). This is the
contract relied on by the refinement rules (``MapFilter``, ``MapSplit``,
``MapCriterionProcessor``) and by the per-pixel loss weighting in
``splanything.training.losses.Loss``.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from splanything.utils.img import ImgUtils


@pytest.fixture
def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _identity_map(H: int, W: int, C: int = 1, device=torch.device("cpu")) -> Tensor:
    """Build a (1, C, H, W) map where pixel (c, i, j) carries value ``j + i*W``.

    Lets us identify which pixel each sample landed at by inspecting the
    sampled value rather than the index math.
    """
    base = torch.arange(H * W, device=device, dtype=torch.float32).view(H, W)
    img = base.unsqueeze(0).unsqueeze(0).expand(1, C, H, W).contiguous()
    return img


# ---------------------------------------------------------------------------
# Pixel-center convention: matches gen_px_coords and primitive centroids
# ---------------------------------------------------------------------------


def test_pixel_center_sampled_exactly(device):
    """UV at pixel ``i``'s center must return pixel ``i``'s value."""
    H = W = 16
    img = _identity_map(H, W, device=device)
    for i in range(H):
        for j in range(W):
            uv = torch.tensor(
                [[(i + 0.5) / H, (j + 0.5) / W]], device=device, dtype=torch.float32
            )
            sampled = ImgUtils.uv_sample(img, uv)
            expected = float(i * W + j)
            assert sampled[0, 0, 0].item() == pytest.approx(expected), (
                f"mismatch at pixel ({i},{j}): "
                f"uv={uv[0].tolist()} got={sampled[0, 0, 0].item()} "
                f"expected={expected}"
            )


def test_pixel_centers_align_with_gen_px_coords(device):
    """``uv_sample`` and ``gen_px_coords`` must sample the same pixel."""
    H = W = 32
    img = _identity_map(H, W, device=device)
    co = ImgUtils.gen_px_coords(H, W, device=device).reshape(2, -1).T
    sampled = ImgUtils.uv_sample(img, co)  # (1, H*W, 1)

    for idx, (cy, cx) in enumerate(co):
        i = int(cy.item() * H - 0.5)
        j = int(cx.item() * W - 0.5)
        expected = float(i * W + j)
        got = sampled[0, idx, 0].item()
        assert got == pytest.approx(expected), (
            f"gen_px_coords coord ({cy.item():.4f}, {cx.item():.4f}) -> "
            f"pixel ({i},{j}) but uv_sample returned {got} (expected {expected})"
        )


def test_uv_half_samples_midpoint_of_two_pixels(device):
    """``uv = 0.5`` (geometric centre of an 8x8 image) is between pixels 3 and 4
    and must bilinear-average them 50/50."""
    H = W = 8
    img = _identity_map(H, W, device=device)
    uv = torch.tensor([[0.5, 0.5]], device=device)
    sampled = ImgUtils.uv_sample(img, uv)
    expected = (3 * W + 3) * 0.5 + (4 * W + 4) * 0.5
    assert sampled[0, 0, 0].item() == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Boundary clamping: out-of-frame UVs return the nearest edge pixel
# ---------------------------------------------------------------------------


def test_uv_zero_returns_leftmost_topmost_pixel(device):
    """``uv = 0`` is outside the leftmost/topmost pixel; it must clamp to that
    pixel rather than extrapolating."""
    H = W = 16
    img = _identity_map(H, W, device=device)
    uv = torch.tensor([[0.0, 0.0]], device=device)
    sampled = ImgUtils.uv_sample(img, uv)
    assert sampled[0, 0, 0].item() == pytest.approx(0.0)


def test_uv_one_returns_rightmost_bottommost_pixel(device):
    H = W = 16
    img = _identity_map(H, W, device=device)
    uv = torch.tensor([[1.0, 1.0]], device=device)
    sampled = ImgUtils.uv_sample(img, uv)
    assert sampled[0, 0, 0].item() == pytest.approx(float((H - 1) * W + (W - 1)))


def test_uv_negative_clamps_to_edge(device):
    """Centroids that drift below zero (possible after splits) must clamp to
    pixel 0 instead of extrapolating with negative weights."""
    H = W = 16
    img = _identity_map(H, W, device=device)
    uv = torch.tensor([[-0.5, -0.5]], device=device)
    sampled = ImgUtils.uv_sample(img, uv)
    assert sampled[0, 0, 0].item() == pytest.approx(0.0)
    assert torch.isfinite(sampled).all()


def test_uv_above_one_clamps_to_edge(device):
    H = W = 16
    img = _identity_map(H, W, device=device)
    uv = torch.tensor([[1.5, 1.5]], device=device)
    sampled = ImgUtils.uv_sample(img, uv)
    assert sampled[0, 0, 0].item() == pytest.approx(float((H - 1) * W + (W - 1)))
    assert torch.isfinite(sampled).all()


# ---------------------------------------------------------------------------
# Output shape and batching
# ---------------------------------------------------------------------------


def test_output_shape(device):
    B, C, H, W = 2, 3, 16, 16
    N = 5
    img = torch.rand(B, C, H, W, device=device)
    uv = torch.rand(N, 2, device=device)
    out = ImgUtils.uv_sample(img, uv)
    assert out.shape == (B, N, C)


def test_single_pixel_returns_full_value(device):
    """At pixel ``i``'s centre, uv_sample must return pixel ``i``'s exact
    value (no blending with neighbours)."""
    H = W = 4
    img = torch.rand(1, 1, H, W, device=device)
    uv = torch.tensor([[0.625, 0.625]], device=device)  # pixel (2, 2) center
    sampled = ImgUtils.uv_sample(img, uv)
    assert sampled[0, 0, 0].item() == pytest.approx(img[0, 0, 2, 2].item())


# ---------------------------------------------------------------------------
# End-to-end with primitive centroids
# ---------------------------------------------------------------------------


def test_uv_sample_matches_primitive_centroid_location(device):
    """A primitive whose centroid is exactly pixel ``i``'s centre must sample
    pixel ``i``'s value. This is the contract MapFilter / MapSplit rely on."""
    from splanything.primitives import RadialFreqPrimitive

    prim = RadialFreqPrimitive(size=4).to(device)
    H = W = 16
    img = _identity_map(H, W, device=device)

    # Force centroid 0 to pixel (3, 5) center, others to image corners.
    new_centroids = prim.centroids.clone()
    new_centroids[0, 0] = (3 + 0.5) / H  # y
    new_centroids[0, 1] = (5 + 0.5) / W  # x
    with torch.no_grad():
        prim.centroids.copy_(new_centroids)

    sampled = ImgUtils.uv_sample(img, prim.centroids)
    assert sampled[0, 0, 0].item() == pytest.approx(float(3 * W + 5))
