"""Tests for ``len(primitive)`` under an active ``masked`` context.

``Primitive.__len__`` reads a batched parameter through the same
``_batched_param`` path used by ``__getattribute__``, so it honours an
active ``masked`` context. ``len(self)`` therefore returns the masked
length inside a ``masked`` block, consistent with batched-parameter
accesses (``self.alphas`` etc.), and any ``cached_property`` deriving a
shape from ``len(self)`` stays
aligned with tensor parameters (``self.thetas`` -> ``R``).
"""

from __future__ import annotations

import pytest
import torch

from splanything.primitives import (
    CubicFanPrimitive,
    MultiPrimitive,
    RadialFreqPrimitive,
)


@pytest.fixture
def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _keep_odd(n: int, device) -> torch.Tensor:
    m = torch.zeros(n, dtype=torch.bool, device=device)
    m[0::2] = True
    return m


# --------------------------------------------------------------------------- #
# Primitive.__len__ under masked
# --------------------------------------------------------------------------- #


def test_len_adapts_to_single_primitive_mask(device):
    p = RadialFreqPrimitive(size=10).to(device)
    m = _keep_odd(10, device)
    with p.masked(m):
        assert len(p) == 5


def test_len_consistent_with_batched_param_inside_mask(device):
    p = RadialFreqPrimitive(size=10).to(device)
    m = _keep_odd(10, device)
    with p.masked(m):
        assert len(p) == p.alphas.shape[0]
        assert len(p) == p.centroids.shape[0]


def test_len_restored_on_mask_exit(device):
    p = RadialFreqPrimitive(size=10).to(device)
    m = _keep_odd(10, device)
    with p.masked(m):
        assert len(p) == 5
    assert len(p) == 10


def test_len_adapts_under_nested_masks(device):
    p = RadialFreqPrimitive(size=10).to(device)
    outer = torch.tensor(
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.bool, device=device
    )
    inner = torch.tensor(
        [1, 0, 1, 0], dtype=torch.bool, device=device
    )  # over outer's 4
    with p.masked(outer):
        assert len(p) == 4
        with p.masked(inner):
            assert len(p) == 2
        assert len(p) == 4
    assert len(p) == 10


def test_len_adapts_under_all_false_mask(device):
    p = RadialFreqPrimitive(size=4).to(device)
    none = torch.zeros(4, dtype=torch.bool, device=device)
    with p.masked(none):
        assert len(p) == 0
    assert len(p) == 4


# --------------------------------------------------------------------------- #
# MultiPrimitive.__len__ under masked (sums per-primitive masked lens)
# --------------------------------------------------------------------------- #


def test_multi_len_adapts_to_mask(device):
    radial = RadialFreqPrimitive(size=5).to(device)
    cubic = CubicFanPrimitive(size=3).to(device)
    multi = MultiPrimitive({"radial": radial, "cubic": cubic})
    m = torch.tensor([1, 0, 1, 0, 1, 0, 1, 1], dtype=torch.bool, device=device)
    with multi.masked(m):
        assert len(multi) == 5  # 3 radial + 2 cubic
    assert len(multi) == 8


def test_multi_masked_slices_use_masked_len(device):
    """``MultiPrimitive.masked`` slices the mask by each prim's len().
    With mask-aware ``len()``, a nested ``MultiPrimitive.masked`` slices
    against the masked per-primitive length, so an inner mask of the
    masked total composes correctly."""
    radial = RadialFreqPrimitive(size=5).to(device)
    cubic = CubicFanPrimitive(size=3).to(device)
    multi = MultiPrimitive({"radial": radial, "cubic": cubic})
    outer = torch.tensor([1, 1, 1, 0, 0, 1, 1, 0], dtype=torch.bool, device=device)
    with multi.masked(outer):
        assert len(multi) == 5
        inner = torch.tensor([1, 0, 1, 0, 1], dtype=torch.bool, device=device)
        with multi.masked(inner):
            assert len(multi) == 3
        assert len(multi) == 5
    assert len(multi) == 8


# --------------------------------------------------------------------------- #
# cached_property / len(self) consistency — the actual error surface
# --------------------------------------------------------------------------- #


def test_cubic_axes_computable_under_mask(device):
    """``axes = R @ ref.unsqueeze(-1)`` must not raise under a mask."""
    p = CubicFanPrimitive(size=80).to(device)
    m = _keep_odd(80, device)
    with p.masked(m):
        ax_1, ax_2 = p.axes
        assert ax_1.shape[0] == 40
        assert ax_2.shape[0] == 40


def test_cubic_sample_rgb_under_mask(device):
    """End-to-end: ``sample_rgb`` exercises ``axes`` and must not raise."""

    p = CubicFanPrimitive(size=80).to(device)
    m = _keep_odd(80, device)
    co = torch.rand(100, 2, device=device)
    with p.masked(m):
        rgb = p.sample_rgb(co)  # (Nc, Np, 3)
        assert rgb.shape[1] == 40


def test_multi_sample_rgb_under_mask(device):
    """The exact traceback scenario: ``MultiPrimitive.sample_rgb`` under a
    mask. Pre-fix this raised ``RuntimeError: size of tensor a (43) must
    match size of tensor b (40)"""
    radial = RadialFreqPrimitive(size=40).to(device)
    cubic = CubicFanPrimitive(size=40).to(device)
    multi = MultiPrimitive({"radial": radial, "cubic": cubic})
    m = torch.zeros(80, dtype=torch.bool, device=device)
    m[0::2] = True  # 40 True
    co = torch.rand(50, 2, device=device)
    with multi.masked(m):
        rgb = multi.sample_rgb(co)
        assert rgb.shape[1] == int(m.sum().item())


# --------------------------------------------------------------------------- #
# Forward path under mask (MetaPrimitive uses len(self) in forward)
# --------------------------------------------------------------------------- #


def test_primitive_forward_under_mask(device):
    """``Primitive.forward`` early-returns zeros if ``len(self) == 0``.
    Under an all-False mask, ``len`` must read 0 so this fast path triggers
    instead of sampling an empty primitive."""
    from splanything.rendering.rasterizers.weighted import WeightedRasterizer

    p = RadialFreqPrimitive(size=4).to(device)
    none = torch.zeros(4, dtype=torch.bool, device=device)
    co = torch.rand(10, 2, device=device)
    with p.masked(none):
        out = p(co, WeightedRasterizer())
        assert out.shape == (10, 4)
        assert out.abs().sum() == 0


# --------------------------------------------------------------------------- #
# Refinement-rule masks (check_filter / check_split allocate via len(self))
# --------------------------------------------------------------------------- #


def test_check_filter_mask_size_matches_under_outer_mask(device):
    """``check_filter`` does ``torch.ones(len(self), ...)``; under an outer
    mask, the combined-filter tensor must match the masked batched params."""
    p = RadialFreqPrimitive(size=8).to(device)
    outer = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.bool, device=device)
    from splanything.training.refinement.rules import AlphaFilter

    p.add_filter_rule(AlphaFilter(threshold=10.0, interval=1))
    with p.masked(outer):
        keep = p.check_filter()
        # Under the outer mask, the combined mask is sized to the masked
        # primitive length (4), not the full (8).
        if keep is not None:
            assert keep.shape[0] == len(p)
            assert keep.shape[0] == 4
    assert len(p) == 8
