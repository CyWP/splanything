"""Regression tests for dormant code paths that previously crashed.

These paths were untested (and unreachable through the standard training
loop), so several NameError/shape bugs survived unnoticed:

* ``GaussianSplitter.split_vals`` used undefined names (``p``, ``param_split``).
* ``FlexibleInitializer.init_param`` referenced ``func`` instead of ``self.func``.
* ``MappedSampleProcessor.process_map`` passed undefined ``distances`` to a
  custom ``proc_fn``; ``process`` also never dropped the batch dim of
  ``mask_sample`` output.
* ``TrainSampler.jitter_target`` had an undefined ``co_patches``, a
  normalized-vs-pixel unit mismatch, and an inverted y-sign between the
  coordinate shift and the image shift.
* ``MetaPrimitive.forward`` skipped local-frame culling (all coordinates
  were sampled regardless of the ``inside`` test).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from splanything.primitives import GaussianPrimitive, RadialFreqPrimitive
from splanything.primitives.initializers.flex import FlexibleInitializer
from splanything.primitives.meta import MetaPrimitive
from splanything.rendering.processors.flex import FlexibleSampleProcessor
from splanything.rendering.processors.mapped import MappedSampleProcessor
from splanything.rendering.sample_output import SampleOutput
from splanything.training.sampler import TrainSampler
from splanything.utils.img import Splimage


@pytest.fixture
def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# GaussianSplitter (split path)
# --------------------------------------------------------------------------- #


def test_gaussian_split_expands_and_halves_dominant_sigma(device):
    """``GaussianPrimitive.split`` must run without NameError and halve the
    dominant sigma of every split instance (both children)."""
    torch.manual_seed(0)
    p = GaussianPrimitive(size=4).to(device)
    s1, s2 = p.sigma_1.detach().clone(), p.sigma_2.detach().clone()
    c = p.centroids.detach().clone()
    dom1 = s1 > s2
    idx = torch.tensor([True, False, True, False], device=device)
    p.split(idx)
    assert len(p) == 6

    e1 = torch.where(dom1, s1 / 2, s1)
    e2 = torch.where(dom1, s2, s2 / 2)
    split_rows = torch.tensor([0, 2], device=device)  # child 1 at original slots
    new_rows = torch.tensor([4, 5], device=device)  # child 2 appended
    for rows in (split_rows, new_rows):
        assert torch.allclose(p.sigma_1[rows], e1[idx])
        assert torch.allclose(p.sigma_2[rows], e2[idx])
    # Child centroids are symmetric about the original centroid.
    assert torch.allclose((p.centroids[split_rows] + p.centroids[new_rows]) / 2, c[idx])


# --------------------------------------------------------------------------- #
# FlexibleInitializer
# --------------------------------------------------------------------------- #


def test_flexible_initializer_uses_wrapped_callable():
    """``FlexibleInitializer`` must delegate to its callable (not crash on
    an undefined ``func``)."""
    calls = []

    def make(name, shape):
        calls.append((name, tuple(shape)))
        return torch.full(shape, 0.5)

    p = RadialFreqPrimitive(size=3, initializers=FlexibleInitializer(make))
    assert ("alphas", (3,)) in calls
    assert torch.allclose(p.alphas, torch.full((3,), 0.5))
    assert torch.allclose(p.centroids, torch.full((3, 2), 0.5))


# --------------------------------------------------------------------------- #
# MappedSampleProcessor
# --------------------------------------------------------------------------- #


def test_mapped_sample_processor_proc_fn_receives_sampled_vals():
    """``process_map`` must pass the sampled map values to ``proc_fn``."""
    mp = Splimage(torch.full((1, 1, 8, 8), 0.5))
    prim = RadialFreqPrimitive(size=3)
    seen = {}

    def proc_fn(sample, primitive, sampled_vals):
        seen["vals"] = sampled_vals
        return SampleOutput(
            rgb=sample.rgb, weights=sample.weights * 2.0, co=sample.co
        )

    proc = MappedSampleProcessor(
        FlexibleSampleProcessor(lambda s, p: s), mp, proc_fn
    )
    sample = SampleOutput(
        rgb=torch.rand(4, 3, 3), weights=torch.rand(4, 3), co=torch.rand(4, 2)
    )
    out = proc(sample, prim)
    # Batch dim dropped: per-primitive values (Np,), all 0.5 here.
    assert seen["vals"].shape == (3,)
    assert torch.allclose(seen["vals"], torch.full((3,), 0.5))
    assert torch.allclose(out.weights, sample.weights * 2.0)


# --------------------------------------------------------------------------- #
# TrainSampler.jitter_target
# --------------------------------------------------------------------------- #


def test_jitter_target_matches_image_at_jittered_coords():
    """The jittered target patches must equal a direct bilinear sample of
    the image at the jittered coordinates (pairing of co/image shifts)."""
    H, W, ps = 32, 32, 16
    img = torch.arange(H * W, dtype=torch.float32).reshape(1, 1, H, W)
    img = img.repeat(1, 4, 1, 1)  # (1, 4, H, W)
    sampler = TrainSampler(target=Splimage(img), patch_size=ps, jitter_coords=True)
    co_before = sampler.co_patches.clone()

    co_j, tgt = sampler.jitter_target()

    # co_patches must not be mutated (clone, no drift across epochs).
    assert torch.equal(sampler.co_patches, co_before)
    assert not torch.equal(co_j, co_before)
    dy = co_j[..., 0] - co_before[..., 0]
    dx = co_j[..., 1] - co_before[..., 1]
    assert dy.abs().max() <= 1 / (2 * H) + 1e-6
    assert dx.abs().max() <= 1 / (2 * W) + 1e-6

    # Independent check: bilinear sample the image at the jittered coords.
    # Sample per coordinate (patch order is patch-major, not image row-major),
    # with border padding to match extract_image_patches' jitter sampling.
    flat = co_j.reshape(H * W, 2)
    grid = torch.stack([flat[:, 1] * 2 - 1, flat[:, 0] * 2 - 1], dim=-1)
    grid = grid.view(1, -1, 1, 2)
    expected = F.grid_sample(
        img, grid, align_corners=False, padding_mode="border"
    )  # (1, C, H*W, 1)
    expected = expected.squeeze(-1).permute(0, 2, 1).reshape(H * W, 4)
    assert torch.allclose(tgt.reshape(H * W, 4), expected, atol=1e-3)


# --------------------------------------------------------------------------- #
# MetaPrimitive.forward local-frame culling
# --------------------------------------------------------------------------- #


def test_meta_forward_zeroes_outside_local_frame(device):
    """Coordinates outside a meta splat's local frame must contribute zero.

    The child is configured with sigma large enough that its weight is
    non-negligible everywhere — so a zero weight at the far coordinate can
    only come from the culling itself."""
    child = RadialFreqPrimitive(size=1).to(device)
    meta = MetaPrimitive(child, size=1, modify_rotation=False).to(device)
    with torch.no_grad():
        meta.centroids.copy_(torch.tensor([[0.5, 0.5]], device=device))
        meta.scales_1.copy_(torch.tensor([0.2], device=device))
        meta.scales_2.copy_(torch.tensor([0.2], device=device))
        child.centroids.copy_(torch.tensor([[0.5, 0.5]], device=device))
        child.sigma.copy_(torch.tensor([10.0], device=device))
        child.floor.copy_(torch.tensor([4.0], device=device))
        child.alphas.copy_(torch.tensor([0.9], device=device))

    co = torch.tensor([[0.5, 0.5], [0.05, 0.05]], device=device)
    out = meta(co)
    assert out.weights[0].abs().sum() > 0  # inside frame: child sampled
    assert out.weights[1].abs().sum() == 0  # outside frame: culled
    assert out.rgb[1].abs().sum() == 0
