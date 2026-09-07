"""Tests for ``MultiPrimitive.masked`` context manager.

``MultiPrimitive.masked`` accepts a single boolean mask spanning every
contained primitive in the order of ``self.primitives.values()``. Inside
the context, accessing batched parameters of any contained primitive
returns the masked slice; the slice boundaries are determined by each
primitive's ``len()``. On exit, the previous mask (or none) is restored.

These tests pin down that contract by exercising:

* per-primitive slicing boundaries;
* consistent masking across multiple batched parameters;
* restoration of full-size access on context exit;
* no leakage from an aborted context (exception path);
* nested ``MultiPrimitive.masked`` composition with the inner mask
  applied on top of the outer mask;
* primitive-level ``masked`` invoked directly inside a multi context.
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


@pytest.fixture
def multi(device):
    radial = RadialFreqPrimitive(size=5).to(device)
    cubic = CubicFanPrimitive(size=3).to(device)
    return MultiPrimitive({"radial": radial, "cubic": cubic})


def _mask_keep_odd(n: int, device) -> torch.Tensor:
    out = torch.zeros(n, dtype=torch.bool, device=device)
    out[0::2] = True
    return out


def test_masked_slices_per_primitive_in_concatenation_order(multi, device):
    """Mask of length len(multi) is sliced by each prim's len() in order."""
    radial = multi["radial"]
    cubic = multi["cubic"]
    n = len(multi)
    assert n == len(radial) + len(cubic)
    mask = torch.tensor(
        [True, False, True, False, True, False, True, True], device=device
    )
    with multi.masked(mask):
        assert radial.alphas.shape[0] == 3
        assert cubic.alphas.shape[0] == 2
    assert radial.alphas.shape[0] == 5
    assert cubic.alphas.shape[0] == 3


def test_masked_returns_correct_subsets(multi, device):
    """Inside the context, batched params return the kept rows."""
    radial = multi["radial"]
    full_alphas = radial.alphas.detach().clone()
    mask = torch.tensor(
        [True, False, True, False, True, False, True, True], device=device
    )
    n_radial = len(radial)  # capture full length before masking
    with multi.masked(mask):
        expected = full_alphas[mask[:n_radial]]
        assert torch.equal(radial.alphas, expected)


def test_masked_applies_to_all_batched_params(multi, device):
    """Every batched parameter of a contained primitive is masked consistently."""
    radial = multi["radial"]
    cubic = multi["cubic"]
    mask = _mask_keep_odd(len(multi), device)
    n_radial = len(radial)  # full length, captured before masking
    with multi.masked(mask):
        r_keep = mask[:n_radial]
        c_keep = mask[n_radial:]
        assert radial.alphas.shape[0] == int(r_keep.sum())
        assert radial.centroids.shape[0] == int(r_keep.sum())
        assert radial.color_1.shape[0] == int(r_keep.sum())
        assert radial.color_2.shape[0] == int(r_keep.sum())
        assert cubic.alphas.shape[0] == int(c_keep.sum())
        assert cubic.centroids.shape[0] == int(c_keep.sum())
        assert cubic.color_1.shape[0] == int(c_keep.sum())
        assert cubic.color_2.shape[0] == int(c_keep.sum())


def test_mask_state_restored_on_exit(multi, device):
    """After the context, batched params see the unmasked tensor again."""
    radial = multi["radial"]
    cubic = multi["cubic"]
    mask = torch.ones(len(multi), dtype=torch.bool, device=device)
    mask[1::2] = False
    with multi.masked(mask):
        pass
    assert radial.alphas.shape[0] == 5
    assert cubic.alphas.shape[0] == 3


def test_mask_state_restored_on_exception(multi, device):
    """Even if the body raises, the mask is unwound on exit."""
    radial = multi["radial"]
    mask = torch.ones(len(multi), dtype=torch.bool, device=device)
    mask[:] = True
    mask[0] = False
    with pytest.raises(RuntimeError, match="boom"):
        with multi.masked(mask):
            assert radial.alphas.shape[0] == 4
            raise RuntimeError("boom")
    assert radial.alphas.shape[0] == 5


def test_outer_then_inner_multi_mask_compose(multi, device):
    """A second ``MultiPrimitive.masked`` stacks on top of the first.

    With mask-aware ``len()``, the inner mask must be sized to the outer
    masked total (not the full ``len(multi)``). The inner mask slices the
    outer-masked sub-tensor per primitive, so a slot is kept iff its outer
    AND inner bits are both True.
    """
    radial = multi["radial"]
    cubic = multi["cubic"]
    outer = torch.tensor(
        [True, False, True, False, True, False, True, True], device=device
    )
    with multi.masked(outer):
        assert radial.alphas.shape[0] == 3
        assert cubic.alphas.shape[0] == 2
        assert len(multi) == 5  # outer masked total
        # Inner mask of length 5 (the outer masked total), keeping 2 slots.
        inner = torch.tensor([True, False, True, False, False], device=device)
        with multi.masked(inner):
            assert radial.alphas.shape[0] == 2
            assert cubic.alphas.shape[0] == 0
        # Inner exited; outer mask still active.
        assert radial.alphas.shape[0] == 3
        assert cubic.alphas.shape[0] == 2
    # Outer exited; full tensors visible.
    assert radial.alphas.shape[0] == 5
    assert cubic.alphas.shape[0] == 3


def test_primitive_masked_directly_inside_multi_masked(multi, device):
    """A child primitive can re-mask itself directly inside a multi context."""
    radial = multi["radial"]
    outer = torch.ones(len(multi), dtype=torch.bool, device=device)
    outer[1::2] = False  # radial indices 0,2,4 kept, cubic indices 0,2 kept
    with multi.masked(outer):
        assert radial.alphas.shape[0] == 3
        inner = torch.tensor([True, False, True], device=device)
        with radial.masked(inner):
            assert radial.alphas.shape[0] == 2
        assert radial.alphas.shape[0] == 3
    assert radial.alphas.shape[0] == 5


def test_masked_preserves_grad_history(multi, device):
    """Masked slicing keeps tensors attached to the autograd graph."""
    radial = multi["radial"]
    mask = torch.ones(len(multi), dtype=torch.bool, device=device)
    mask[1::2] = False
    with multi.masked(mask):
        a = radial.alphas
        assert a.requires_grad
        a.sum().backward()
    # Gradients should be set only on kept rows in the unmasked param.
    g = radial.alphas.grad
    assert g is not None
    expected = torch.zeros(5, device=device)
    expected[0::2] = 1.0
    assert torch.allclose(g, expected)


def test_masked_full_true_is_identity(multi, device):
    """An all-True mask of length len(multi) preserves all batched params."""
    radial = multi["radial"]
    cubic = multi["cubic"]
    full = torch.ones(len(multi), dtype=torch.bool, device=device)
    pre_r = radial.alphas.detach().clone()
    pre_c = cubic.alphas.detach().clone()
    with multi.masked(full):
        assert torch.equal(radial.alphas, pre_r)
        assert torch.equal(cubic.alphas, pre_c)


def test_masked_with_all_false_yields_zero_length(multi, device):
    """An all-False mask leaves each primitive with zero rows."""
    radial = multi["radial"]
    cubic = multi["cubic"]
    none = torch.zeros(len(multi), dtype=torch.bool, device=device)
    with multi.masked(none):
        assert radial.alphas.shape[0] == 0
        assert cubic.alphas.shape[0] == 0
    assert radial.alphas.shape[0] == 5
    assert cubic.alphas.shape[0] == 3


def test_masked_ignores_extra_length(multi, device):
    """Extra trailing elements beyond len(multi) are silently truncated.

    ``MultiPrimitive.masked`` slices the mask per primitive via
    ``mask[si:ei]``; surplus elements past the last primitive are never
    read and do not raise. (Under-length masks would produce a shorter
    slice for the last primitive and surface as a shape mismatch when
    batched parameters are accessed inside the context.)
    """
    radial = multi["radial"]
    cubic = multi["cubic"]
    over = torch.ones(len(multi) + 3, dtype=torch.bool, device=device)
    over[1::2] = False  # pattern: [T,F,T,F,T,F,T,F,T,F,T]
    with multi.masked(over):
        assert radial.alphas.shape[0] == 3  # mask[:5]=[T,F,T,F,T]
        assert cubic.alphas.shape[0] == 1  # mask[5:8]=[F,T,F]
    assert radial.alphas.shape[0] == 5
    assert cubic.alphas.shape[0] == 3
