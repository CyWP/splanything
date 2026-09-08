"""Tests for the ``Sampler.train_sampler`` factory and training render path.

The previous ``TrainSampler`` class (patch-target extraction, per-pixel
Bernoulli subsampling, coordinate jittering) was removed; target images
are now owned by individual losses and the trainer renders the full image
each epoch via a plain ``Sampler`` built by the ``train_sampler`` factory.
"""

from __future__ import annotations

import torch
import pytest

from splanything.primitives import RadialFreqPrimitive
from splanything.rendering import Sampler
from splanything.rendering.rasterizers.weighted import WeightedRasterizer
from splanything.training import Trainer, OptimizerWrapper
from splanything.training.losses import L1Loss, L2Loss
from splanything.utils.img import Splimage


@pytest.fixture
def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_train_sampler_builds_sampler_with_cleared_flags():
    """The factory returns a Sampler with training flags cleared and
    parameters threaded through."""
    sampler = Sampler.train_sampler(64, 96, patch_size=16, max_batch=1000)
    assert isinstance(sampler, Sampler)
    assert sampler.H == 64
    assert sampler.W == 96
    assert sampler.patch_size == 16
    assert sampler.max_batch == 1000
    assert sampler.low_vram is False
    assert isinstance(sampler.rasterizer, WeightedRasterizer)


def test_train_sampler_rasterize_covers_full_canvas():
    """Rasterizing through the factory-built sampler yields a full
    (B, C, H, W) canvas with padding included."""
    H, W = 64, 96
    sampler = Sampler.train_sampler(
        H,
        W,
        patch_size=16,
        max_batch=10000,
        padding=(8, 8, 8, 8),
    )
    p = RadialFreqPrimitive(size=2)
    out = sampler.rasterize(p)
    assert out.shape == (1, 4, H + 16, W + 16)


def test_train_sampler_render_is_outside_autograd():
    """``render`` produces a Splimage without a grad graph regardless of
    grad mode (low_vram/verbose semantics of the training factory)."""
    sampler = Sampler.train_sampler(32, 32, patch_size=16, max_batch=10000)
    p = RadialFreqPrimitive(size=2).requires_grad_(True)
    with torch.no_grad():
        img = sampler.render(p)
    assert not img.image().requires_grad


def test_samples_masks_are_never_mutated_in_place():
    """Boolean masks handed to ``Primitive.masked`` during ``samples()``
    must never be modified in place after their batch is recorded.

    Regression: ``Sampler.samples`` reused one mask tensor per patch
    group and zeroed it in place for the next group, mutating the bool
    tensor saved by autograd (``IndexBackward0`` of the primitive's
    masked indexing) and breaking a single end-of-epoch backward with a
    version-counter error. Fixed by allocating a fresh mask per group.
    """
    import contextlib

    torch.manual_seed(0)
    p = RadialFreqPrimitive(size=20)
    sampler = Sampler.train_sampler(64, 64, patch_size=16, max_batch=40000)
    recorded = []
    orig_masked = p.masked

    @contextlib.contextmanager
    def tracked_masked(mask):
        recorded.append((mask, mask._version))
        with orig_masked(mask):
            yield

    p.masked = tracked_masked
    n_groups = 0
    for _sample, _co in sampler.samples(p):
        n_groups += 1
    assert n_groups > 1, "test setup must produce several mask groups"
    mutated = [i for i, (m, ver) in enumerate(recorded) if m._version != ver]
    assert mutated == [], (
        f"sample-group masks {mutated} were modified in place after being "
        f"handed to Primitive.masked; a single end-of-epoch backward would "
        f"fail with an inplace version-counter error."
    )


def test_backward_across_masked_batches_survives_single_backward(device):
    """One backward at epoch end must work even though several patch
    groups were yielded from the same ``samples()`` generator.

    Regression: ``Sampler.samples`` used to zero its primitive mask in
    place between groups, mutating the bool tensor saved by autograd
    (``IndexBackward0`` of the primitive's masked indexing) and crashing
    the end-of-epoch backward with a version-counter error.
    """
    torch.manual_seed(0)
    prim = RadialFreqPrimitive(size=20).to(device).requires_grad_(True)
    tgt = Splimage(torch.rand(1, 4, 64, 64, device=device))
    loss = L1Loss(tgt)
    # Multiple small patches force several mask accumulations per epoch,
    # while max_batch forces several groups (and pixel chunking) per epoch.
    sampler = Sampler.train_sampler(
        64, 64, patch_size=16, max_batch=1000, device=device
    )
    trainer = Trainer(
        name="masked_backward",
        primitive=prim,
        sampler=sampler,
        optimizer=OptimizerWrapper(prim, torch.optim.AdamW, lr=0.01),
        losses={"L1": (loss, 1.0)},
        callbacks=[],
        base_folder="/tmp/splanything_masked_backward",
    )
    trainer.epoch = 1
    try:
        # Passing means the end-of-epoch backward over several masked
        # batch groups did not hit the inplace-mask version error.
        trainer.exec_epoch()
        assert all(
            torch.isfinite(torch.tensor(v)) for v in trainer.last_losses.values()
        )
    finally:
        trainer._log_handler.close()
        trainer._pkg_logger.removeHandler(trainer._log_handler)


def test_trainer_backward_through_loss_owned_target(device):
    """The full-image training path renders with the sampler and backprops
    through the loss's own (resized) target into primitive parameters."""
    prim = RadialFreqPrimitive(size=2).to(device)
    tgt = Splimage(torch.rand(1, 4, 32, 32, device=device))
    # Deliberately mismatched target resolution: loss handles the resize.
    loss = L1Loss(Splimage(torch.rand(1, 4, 48, 64, device=device)))
    sampler = Sampler.train_sampler(
        32, 32, patch_size=16, max_batch=10000, device=device
    )
    trainer = Trainer(
        name="loss_target",
        primitive=prim,
        sampler=sampler,
        optimizer=OptimizerWrapper(prim, torch.optim.AdamW, lr=0.01),
        losses={"L1": (loss, 1.0), "L2": (L2Loss(tgt), 0.5)},
        callbacks=[],
        base_folder="/tmp/splanything_loss_target",
    )
    trainer.epoch = 1
    prim.requires_grad_(True)
    try:
        before = {n: pp.detach().clone() for n, pp in prim.named_parameters()}
        trainer.exec_epoch()
        after = {n: pp for n, pp in prim.named_parameters()}
        moved = [n for n in before if not torch.equal(before[n], after[n])]
        assert len(moved) > 0, "no primitive parameter moved after one epoch"
        assert all(
            torch.isfinite(torch.tensor(v)) for v in trainer.last_losses.values()
        )
    finally:
        trainer._log_handler.close()
        trainer._pkg_logger.removeHandler(trainer._log_handler)
