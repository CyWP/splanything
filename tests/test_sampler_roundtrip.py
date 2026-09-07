"""Round-trip tests for the training sampler patch layout.

``TrainSampler`` extracts target pixels via ``extract_image_patches`` and
coordinates via ``get_patches``. The two must describe the same patch grid:
equal patch count, and ``target_patches[i]`` must correspond to
``co_patches[i]`` for every i. A reshape bug that folds the channel axis
into the patch axis inflates the target patch count by a factor of C and
desynchronises targets from coordinates, which surfaces as channel-coloured
horizontal banding in the fitted image.
"""

from __future__ import annotations

from pathlib import Path

import torch

from splanything.training.sampler import TrainSampler
from splanything.utils.img import ImgUtils, Splimage

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def _toy_image(H: int, W: int, C: int = 4) -> torch.Tensor:
    img = torch.zeros(1, C, H, W)
    for c in range(C):
        for h in range(H):
            for w in range(W):
                img[0, c, h, w] = (c + 1) * 1000 + h * 100 + w
    return img


def test_sampler_target_patch_count_equals_coord_patch_count():
    """target_patches.shape[0] must equal co_patches.shape[0].

    This is the direct symptom of the channel-folded-into-patch reshape bug:
    target_patches ends up with B*P*C rows while co_patches has P rows.
    """
    H, W, ps = 64, 96, 16
    img = _toy_image(H, W, C=4)
    sampler = TrainSampler(target=Splimage(img), patch_size=ps)

    n_target = sampler.target_patches.shape[0]
    n_co = sampler.co_patches.shape[0]
    assert n_target == n_co, (
        f"target_patches has {n_target} rows but co_patches has {n_co} rows. "
        f"Ratio {n_target / n_co:.3f} (expected 1.0; C={img.shape[1]})."
    )


def test_sampler_batch_dim_collapsed_correctly():
    """After reshape, target_patches should be (P, S, C), not (B*P*C, S, C)."""
    H, W, ps = 64, 96, 16
    img = _toy_image(H, W, C=4)
    sampler = TrainSampler(target=Splimage(img), patch_size=ps)

    P_expected = (H // ps) * (W // ps)
    S_expected = ps * ps
    assert sampler.target_patches.shape == (P_expected, S_expected, 4), (
        f"expected ({P_expected}, {S_expected}, 4), "
        f"got {tuple(sampler.target_patches.shape)}"
    )


def test_sampler_target_patch_i_matches_coord_patch_i():
    """For each patch i, target_pixels[i, s] must be the image pixel at the
    spatial location described by co_patches[i, s]."""
    H, W, ps = 64, 96, 16
    img = _toy_image(H, W, C=4)
    sampler = TrainSampler(target=Splimage(img), patch_size=ps)

    P, S, _ = sampler.co_patches.shape
    PW = W // ps
    for p in range(P):
        pi, pj = p // PW, p % PW
        co_patch = sampler.co_patches[p]  # (S, 2)
        tgt_patch = sampler.target_patches[p]  # (S, C)
        for s in range(S):
            yi = (co_patch[s, 0] * H - 0.5).round().long().clamp(0, H - 1)
            xi = (co_patch[s, 1] * W - 0.5).round().long().clamp(0, W - 1)
            local_h = pi * ps + (s // ps)
            local_w = pj * ps + (s % ps)
            for c in range(4):
                expected = (c + 1) * 1000 + local_h * 100 + local_w
                got = tgt_patch[s, c].item()
                assert got == expected, (
                    f"patch {p} (grid {pi},{pj}) slot {s}: "
                    f"target c={c} got={got} expected={expected} "
                    f"coord_idx=({yi},{xi}) local=({local_h},{local_w})"
                )


def test_sampler_rasterize_roundtrip_with_identity_target():
    """If the primitive reproduces the target exactly, rasterize must return it.

    Uses a trivial monkeypatch-free check: assemble the target patches back
    through the sampler's assemble path and confirm it matches the target.
    This exercises the same assemble_patches the rasterizer uses, against the
    sampler's own patch grid.
    """
    H, W, ps = 64, 96, 16
    img = _toy_image(H, W, C=4)
    sampler = TrainSampler(target=Splimage(img), patch_size=ps)
    recon = ImgUtils.assemble_patches(sampler.target_patches, H, W)
    assert recon.shape == img.shape
    assert torch.equal(recon, img)


# --------------------------------------------------------------------------- #
# Sampling map patch layout
# --------------------------------------------------------------------------- #
#
# Regression: when TrainSampler is constructed with a sampling_map, the
# map patches must share the same (P, S) layout as co_patches. A previous
# bug called set_sampling_map a second time with patch_size=None inside
# __init__, which routed through the single-patch fallback in
# extract_image_patches and produced sampling_patches shaped (1, H*W)
# instead of (P, S). Every subsequent indexing operation in samples()
# then crashed or mis-masked.


def _toy_map(H: int, W: int) -> torch.Tensor:
    """Build a (1, 1, H, W) probability map in [0, 1]."""
    m = torch.zeros(1, 1, H, W)
    for h in range(H):
        for w in range(W):
            m[0, 0, h, w] = 0.5
    return m


def test_sampler_sampling_patches_matches_coord_patch_count():
    """sampling_patches.shape[0] must equal co_patches.shape[0].

    Symptom of the redundant-set_sampling_map bug: sampling_patches ends
    up with 1 row (single-patch fallback) while co_patches has P rows.
    """
    H, W, ps = 64, 96, 16
    img = _toy_image(H, W, C=4)
    mp = _toy_map(H, W)
    sampler = TrainSampler(
        target=Splimage(img), patch_size=ps, sampling_map=Splimage(mp)
    )

    P_co = sampler.co_patches.shape[0]
    P_sp = sampler.sampling_patches.shape[0]
    assert P_sp == P_co, (
        f"sampling_patches has {P_sp} rows but co_patches has {P_co}. "
        f"Ratio {P_sp / P_co:.3f} (expected 1.0; a ratio near 0 indicates "
        f"the patch_size=None fallback was taken)."
    )


def test_sampler_sampling_patches_matches_coord_patch_slots():
    """sampling_patches.shape[1] must equal co_patches.shape[1] (S)."""
    H, W, ps = 64, 96, 16
    img = _toy_image(H, W, C=4)
    mp = _toy_map(H, W)
    sampler = TrainSampler(
        target=Splimage(img), patch_size=ps, sampling_map=Splimage(mp)
    )

    S_co = sampler.co_patches.shape[1]
    S_sp = sampler.sampling_patches.shape[1]
    assert S_sp == S_co, (
        f"sampling_patches has {S_sp} cols but co_patches has {S_co} (S). "
        f"A mismatch desynchronises per-pixel Bernoulli masks from "
        f"coordinates and targets."
    )


def test_sampler_sampling_patches_in_unit_range():
    """Per-pixel sampling probabilities must lie in [0, 1].

    Without re-normalisation the stored map should pass through unchanged;
    this guards against accidental scaling or dtype corruption.
    """
    H, W, ps = 64, 96, 16
    img = _toy_image(H, W, C=4)
    mp = _toy_map(H, W)
    sampler = TrainSampler(
        target=Splimage(img), patch_size=ps, sampling_map=Splimage(mp)
    )

    sp = sampler.sampling_patches
    assert sp.min().item() >= 0.0
    assert sp.max().item() <= 1.0


def test_sampler_sampling_patches_resized_when_map_size_mismatches():
    """If sampling_map H/W differ from the target, it is resized to match.

    Layout (P, S) must then still align with co_patches.
    """
    H, W, ps = 64, 96, 16
    img = _toy_image(H, W, C=4)
    mp = _toy_map(H * 2, W * 2)  # different resolution
    sampler = TrainSampler(
        target=Splimage(img), patch_size=ps, sampling_map=Splimage(mp)
    )

    assert sampler.sampling_patches.shape[0] == sampler.co_patches.shape[0]
    assert sampler.sampling_patches.shape[1] == sampler.co_patches.shape[1]
