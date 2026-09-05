"""Tests for ``Primitive.filter`` / ``Primitive.split`` ↔ ``OptimizerWrapper``
state tracking.

After ``Primitive.filter`` or ``Primitive.split`` the primitive replaces
its ``nn.Parameter`` tensors with new ones (via ``update_parameters``).
``OptimizerWrapper.filter`` and ``OptimizerWrapper.split`` must update
the optimizer's ``param_groups`` to point at those new tensors; otherwise
``opt.step()`` writes into orphan tensors that no primitive owns and the
actual primitive params never move.

Two failure modes are exercised:

  1. Group-name mismatch between ``MultiPrimitive.param_groups`` (whose
     names look like ``radial$$centroids``) and ``MultiPrimitive.check_filter``
     (whose mask dict is keyed by the bare sub-primitive name ``radial``).
     Previously the optimizer's name-vs-key check used an exact match, so
     no group was matched and every group kept its orphan OLD tensor.
     Fixed by prefix matching via ``OptimizerWrapper._resolve_subparam_key``.

  2. ``OptimizerWrapper.filter`` and ``OptimizerWrapper.split`` replace
     ``self._optimizer.state`` with a plain ``dict``, clobbering PyTorch's
     ``defaultdict(dict)``. AdamW then raises ``KeyError`` on the next
     ``step()`` when a freshly-rebuilt group references a param with no
     state entry. Not fixed yet.
"""

from __future__ import annotations

from collections import defaultdict

import pytest
import torch
from torch.optim import AdamW

from splanything.primitives import (
    CubicFanPrimitive,
    MetaPrimitive,
    MultiPrimitive,
    RadialFreqPrimitive,
)
from splanything.training import OptimizerWrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _seed_grads(prim, scale=1.0):
    for _, p in prim.batched_parameters():
        if p.requires_grad:
            p.grad = torch.full_like(p, scale)


def _group_param_ids(opt):
    """Return the set of param tensor ids that appear in any optimizer group."""
    ids = set()
    for g in opt._optimizer.param_groups:
        ps = g["params"]
        if isinstance(ps, torch.Tensor):
            ps = [ps]
        ids.update(id(p) for p in ps)
    return ids


# ---------------------------------------------------------------------------
# Bug 1: Primitive.param_groups omits "name"; filter/split collapse groups.
# ---------------------------------------------------------------------------


def test_primitive_param_groups_lacks_name_key(device):
    """``Primitive.param_groups`` should label each group by parameter name
    so that ``OptimizerWrapper.filter`` / ``split`` can map old groups to
    new ones. Today they are nameless, which is what triggers the bug.
    """
    prim = RadialFreqPrimitive(size=4).to(device)
    pg = prim.param_groups()
    assert all("name" in g for g in pg), (
        f"Every param_groups dict should have a 'name' key; got "
        f"{[list(g.keys()) for g in pg]}"
    )


def test_filter_keeps_all_batched_params_in_optimizer(device):
    """After ``Primitive.filter`` every surviving batched parameter must
    appear in some optimizer group (otherwise AdamW will not touch it)."""
    prim = RadialFreqPrimitive(size=5).to(device)
    opt = OptimizerWrapper(prim, AdamW, lr=0.01)

    keep = torch.tensor([True, False, True, False, True], device=device)
    prim.filter(keep)
    opt.filter(prim.param_groups(), keep)

    expected_ids = {id(p) for _, p in prim.batched_parameters()}
    actual_ids = _group_param_ids(opt)
    missing = expected_ids - actual_ids
    assert not missing, (
        f"After filter: {len(missing)}/{len(expected_ids)} batched params "
        f"are missing from the optimizer's param_groups. They will not be "
        f"updated by AdamW."
    )


def test_split_keeps_all_batched_params_in_optimizer(device):
    prim = RadialFreqPrimitive(size=4).to(device)
    opt = OptimizerWrapper(prim, AdamW, lr=0.01)

    split_idx = torch.tensor([True, False, True, False], device=device)
    prim.split(split_idx)
    opt.split(prim.param_groups(), split_idx)

    expected_ids = {id(p) for _, p in prim.batched_parameters()}
    actual_ids = _group_param_ids(opt)
    missing = expected_ids - actual_ids
    assert not missing, (
        f"After split: {len(missing)}/{len(expected_ids)} batched params "
        f"are missing from the optimizer's param_groups."
    )


def test_filter_does_not_collapse_groups_to_single_tensor(device):
    """The smoking-gun assertion: after filter, no two groups should
    reference the same ``nn.Parameter`` object."""
    prim = RadialFreqPrimitive(size=5).to(device)
    opt = OptimizerWrapper(prim, AdamW, lr=0.01)

    keep = torch.tensor([True, False, True, False, True], device=device)
    prim.filter(keep)
    opt.filter(prim.param_groups(), keep)

    seen_ids = []
    for g in opt._optimizer.param_groups:
        ps = g["params"]
        if isinstance(ps, torch.Tensor):
            ps = [ps]
        seen_ids.extend(id(p) for p in ps)

    unique_ids = set(seen_ids)
    assert len(unique_ids) == len(seen_ids), (
        f"Filter collapsed {len(seen_ids)} group slots onto "
        f"{len(unique_ids)} unique tensors. Every group is pointing at "
        f"the same parameter, so AdamW updates it repeatedly and ignores "
        f"all the others."
    )


def test_split_does_not_collapse_groups_to_single_tensor(device):
    prim = RadialFreqPrimitive(size=4).to(device)
    opt = OptimizerWrapper(prim, AdamW, lr=0.01)

    split_idx = torch.tensor([True, False, True, False], device=device)
    prim.split(split_idx)
    opt.split(prim.param_groups(), split_idx)

    seen_ids = []
    for g in opt._optimizer.param_groups:
        ps = g["params"]
        if isinstance(ps, torch.Tensor):
            ps = [ps]
        seen_ids.extend(id(p) for p in ps)

    unique_ids = set(seen_ids)
    assert len(unique_ids) == len(seen_ids), (
        f"Split collapsed {len(seen_ids)} group slots onto "
        f"{len(unique_ids)} unique tensors."
    )


# ---------------------------------------------------------------------------
# Bug 1 in the actual example setup: MetaPrimitive over MultiPrimitive
# ---------------------------------------------------------------------------


def test_meta_filter_does_not_collapse_groups(device):
    """Reproduces the user's exact setup: MetaPrimitive wrapping a
    MultiPrimitive (radial + cubic)."""
    prim = MetaPrimitive(
        primitive=MultiPrimitive(
            {
                "radial": RadialFreqPrimitive(size=5),
                "cubic": CubicFanPrimitive(size=5),
            }
        ),
        size=20,
        primitive_trainable=False,
    ).to(device)
    opt = OptimizerWrapper(prim, AdamW, lr=0.01)

    keep = torch.zeros(len(prim), dtype=torch.bool, device=device)
    keep[:10] = True
    prim.filter(keep)
    opt.filter(prim.param_groups(), keep)

    expected_ids = {id(p) for _, p in prim.batched_parameters()}
    actual_ids = _group_param_ids(opt)
    missing = expected_ids - actual_ids
    assert not missing, (
        f"After filter on the example MetaPrimitive: "
        f"{len(missing)}/{len(expected_ids)} batched params are missing "
        f"from optimizer.param_groups."
    )


def test_multiprimitive_param_groups_have_names(device):
    """Sanity check: ``MultiPrimitive.param_groups`` already adds a name
    (this is why the inner radial/cubic groups are unaffected)."""
    multi = MultiPrimitive(
        {
            "radial": RadialFreqPrimitive(size=5),
            "cubic": CubicFanPrimitive(size=5),
        }
    ).to(device)
    pg = multi.param_groups()
    assert all("name" in g for g in pg), (
        f"MultiPrimitive param_groups should set 'name'; got "
        f"{[g.get('name') for g in pg]}"
    )


def _all_batched_params(multi):
    """Flatten ``MultiPrimitive.batched_parameters`` over its sub-primitives.

    ``MultiPrimitive`` itself raises ``NotImplementedError``; this helper
    iterates each sub-primitive's batched params and yields them.
    """
    for prim in multi.primitives.values():
        for name, p in prim.batched_parameters():
            yield name, p


def _optimizer_param_ids(opt):
    ids = set()
    for g in opt._optimizer.param_groups:
        ps = g["params"]
        if isinstance(ps, torch.Tensor):
            ps = [ps]
        ids.update(id(p) for p in ps)
    return ids


def test_multiprimitive_filter_also_collapses_groups(device):
    """After filtering a sub-primitive of a ``MultiPrimitive``, every
    post-filter batched parameter must be tracked by the optimizer.

    Before the fix, ``OptimizerWrapper.filter`` used exact-name matching
    between group names (``radial$$centroids``) and ``keep_mask`` keys
    (``radial``), so no group was matched and the optimizer kept
    references to the OLD pre-filter tensors. Subsequent ``opt.step()``
    would update orphans that no primitive owned.
    """
    multi = MultiPrimitive(
        {
            "radial": RadialFreqPrimitive(size=5),
            "cubic": CubicFanPrimitive(size=5),
        }
    ).to(device)
    opt = OptimizerWrapper(multi, AdamW, lr=0.01)

    keep = {
        "radial": torch.tensor([True, False, True, False, True], device=device),
    }
    multi.filter(keep)
    opt.filter(multi.param_groups(), keep)

    primitive_ids = {id(p) for _, p in _all_batched_params(multi)}
    opt_ids = _optimizer_param_ids(opt)

    missing = primitive_ids - opt_ids
    assert not missing, (
        f"After MultiPrimitive.filter: {len(missing)} of "
        f"{len(primitive_ids)} post-filter batched params are missing "
        f"from optimizer.param_groups. opt.step() will not update them."
    )


def test_multiprimitive_split_also_collapses_groups(device):
    """Same as the filter case but for ``MultiPrimitive.split``: the
    optimizer must track the post-split tensors, not the pre-split ones.
    """
    multi = MultiPrimitive(
        {
            "radial": RadialFreqPrimitive(size=4),
            "cubic": CubicFanPrimitive(size=4),
        }
    ).to(device)
    opt = OptimizerWrapper(multi, AdamW, lr=0.01)

    split_mask = {
        "radial": torch.tensor([True, False, True, False], device=device),
    }
    multi.split(split_mask)
    opt.split(multi.param_groups(), split_mask)

    primitive_ids = {id(p) for _, p in _all_batched_params(multi)}
    opt_ids = _optimizer_param_ids(opt)

    missing = primitive_ids - opt_ids
    assert not missing, (
        f"After MultiPrimitive.split: {len(missing)} of "
        f"{len(primitive_ids)} post-split batched params are missing "
        f"from optimizer.param_groups."
    )


def test_optimizer_refs_track_primitive_refs_after_filter(device):
    """Sanity check: every optimizer group param must point to a tensor
    the primitive actually owns (i.e. is in ``multi.batched_parameters``
    or its sub-primitives'). If the optimizer holds an orphan, opt.step()
    silently writes to a detached tensor.
    """
    multi = MultiPrimitive(
        {
            "radial": RadialFreqPrimitive(size=5),
            "cubic": CubicFanPrimitive(size=5),
        }
    ).to(device)
    opt = OptimizerWrapper(multi, AdamW, lr=0.01)

    keep = {
        "radial": torch.tensor([True, False, True, False, True], device=device),
    }
    multi.filter(keep)
    opt.filter(multi.param_groups(), keep)

    primitive_ids = {id(p) for _, p in _all_batched_params(multi)}
    orphan_ids = _optimizer_param_ids(opt) - primitive_ids
    assert not orphan_ids, (
        f"After filter: optimizer holds {len(orphan_ids)} param tensors "
        f"that no primitive owns."
    )


# ---------------------------------------------------------------------------
# Bug 2: filter/split clobber defaultdict state.
# ---------------------------------------------------------------------------


def test_filter_preserves_defaultdict_state(device):
    """``OptimizerWrapper.filter`` (and ``split``) replace
    ``self._optimizer.state`` with a plain ``dict``, breaking lazy state
    initialization. PyTorch optimizers rely on ``state[p]`` returning
    ``{}`` for unseen params; with a plain dict this raises ``KeyError``."""
    prim = RadialFreqPrimitive(size=5).to(device)
    opt = OptimizerWrapper(prim, AdamW, lr=0.01)
    assert isinstance(opt._optimizer.state, defaultdict), (
        "Before filter the optimizer state should be a defaultdict so "
        "AdamW can lazily init state for new params."
    )

    keep = torch.tensor([True, False, True, False, True], device=device)
    prim.filter(keep)
    opt.filter(prim.param_groups(), keep)

    assert isinstance(opt._optimizer.state, defaultdict), (
        "After filter the optimizer state should STILL be a defaultdict "
        "(or otherwise allow lazy init). Replacing it with a plain dict "
        "breaks AdamW.step()."
    )

    keep = torch.tensor([True, False, True], device=device)  # 3 rows remain
    prim.filter(keep)
    opt.filter(prim.param_groups(), keep)

    assert isinstance(
        opt._optimizer.state, type(__import__("collections").defaultdict())
    ), (
        "After filter the optimizer state should STILL be a defaultdict "
        "(or otherwise allow lazy init). Replacing it with a plain dict "
        "breaks AdamW.step()."
    )
