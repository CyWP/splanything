"""Tests for ``TrainerLogHandler`` and the trainer's logger wiring.

The handler attaches to the top-level ``splanything`` package logger so it
catches records emitted by any submodule whose dotted name begins with
``splanything.`` (``splanything.training.trainer``, ``splanything.training.optimizer``,
``splanything.training.sampler``, ``splanything.primitives.*`` ...) via the
standard logger-hierarchy propagation.
"""

from __future__ import annotations

import logging

import pytest
import torch
from torch.optim import AdamW

import splanything.training.trainer as trainer_mod
from splanything.primitives import RadialFreqPrimitive
from splanything.training import Trainer, TrainSampler, OptimizerWrapper
from splanything.training.losses import L2Loss
from splanything.training.trainer import TrainerLogHandler
from splanything.utils.img import Splimage


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _make_minimal_trainer(device, name="probe"):
    prim = RadialFreqPrimitive(size=2).to(device)
    target = torch.rand(1, 4, 8, 8, device=device)
    sampling_map = torch.full((1, 1, 8, 8), 1.0, device=device)
    sampler = TrainSampler(
        target=Splimage(target),
        patch_size=8,
        max_batch=10000,
        sampling_map=Splimage(sampling_map),
        low_vram=False,
    )
    optimizer = OptimizerWrapper(prim, AdamW, lr=0.01)
    return Trainer(
        name=name,
        primitive=prim,
        sampler=sampler,
        optimizer=optimizer,
        losses={"L2": (L2Loss(), 1.0)},
        callbacks=[],
        base_folder="/tmp/splanything_log_handler",
    )


# ---------------------------------------------------------------------------
# Handler wiring
# ---------------------------------------------------------------------------


def test_trainer_attaches_handler_to_package_logger():
    """``Trainer.__init__`` must attach the handler to the top-level
    ``splanything`` package logger so logs from ANY submodule
    (``splanything.training.trainer``, ``splanything.training.optimizer``,
    ``splanything.training.sampler``, ``splanything.primitives.*`` ...)
    that propagate up the dotted-name hierarchy are captured."""
    trainer = _make_minimal_trainer(torch.device("cpu"), name="wiring")
    try:
        assert isinstance(trainer._log_handler, TrainerLogHandler)
        assert trainer._log_handler.trainer is trainer
        assert trainer.logger is trainer_mod._logger
        assert trainer.logger.name == "splanything.training.trainer"
        assert trainer._pkg_logger.name == "splanything"
        assert trainer._log_handler in trainer._pkg_logger.handlers
    finally:
        trainer._log_handler.close()
        trainer._pkg_logger.removeHandler(trainer._log_handler)


def test_trainer_logger_is_shared_across_instances():
    """Two trainers share the package logger AND the module logger, but
    each contributes its own handler (so each trainer's logs land in its
    own ``self.logs``)."""
    a = _make_minimal_trainer(torch.device("cpu"), name="alpha")
    b = _make_minimal_trainer(torch.device("cpu"), name="beta")
    try:
        assert a.logger is b.logger
        assert a._pkg_logger is b._pkg_logger
        assert a._log_handler is not b._log_handler
        assert a._log_handler in a._pkg_logger.handlers
        assert b._log_handler in b._pkg_logger.handlers
    finally:
        for t in (a, b):
            t._log_handler.close()
            t._pkg_logger.removeHandler(t._log_handler)


# ---------------------------------------------------------------------------
# Formatting and forwarding
# ---------------------------------------------------------------------------


def test_emit_formats_and_appends_to_self_logs():
    trainer = _make_minimal_trainer(torch.device("cpu"), name="emit")
    trainer.epoch = 7
    try:
        trainer.logger.info("hello world")
        trainer.logger.warning("a warning")

        msgs = trainer.logs[7]["msg"]
        assert "[INFO] [epoch 7] hello world" in msgs
        assert "[WARNING] [epoch 7] a warning" in msgs
    finally:
        trainer._log_handler.close()
        trainer._pkg_logger.removeHandler(trainer._log_handler)


def test_emit_respects_handler_level():
    trainer = _make_minimal_trainer(torch.device("cpu"), name="level")
    trainer.epoch = 1
    trainer._pkg_logger.removeHandler(trainer._log_handler)
    trainer._log_handler.close()
    trainer._log_handler = TrainerLogHandler(
        trainer,
        level=logging.WARNING,
        fmt="[%(levelname)s] [epoch %(epoch)s] %(message)s",
    )
    trainer._pkg_logger.addHandler(trainer._log_handler)
    try:
        trainer.logger.info("info: should be dropped")
        trainer.logger.warning("warn: kept")
        trainer.logger.error("err: kept")

        msgs = trainer.logs[1]["msg"]
        assert not any("info:" in m for m in msgs)
        assert "[WARNING] [epoch 1] warn: kept" in msgs
        assert "[ERROR] [epoch 1] err: kept" in msgs
    finally:
        trainer._log_handler.close()
        trainer._pkg_logger.removeHandler(trainer._log_handler)


def test_emit_with_no_formatter_uses_raw_message():
    trainer = _make_minimal_trainer(torch.device("cpu"), name="raw")
    trainer.epoch = 2
    trainer._pkg_logger.removeHandler(trainer._log_handler)
    trainer._log_handler.close()
    trainer._log_handler = TrainerLogHandler(trainer, level=logging.INFO)
    trainer._pkg_logger.addHandler(trainer._log_handler)
    try:
        trainer.logger.info("plain text")
        assert trainer.logs[2]["msg"][-1] == "plain text"
    finally:
        trainer._log_handler.close()
        trainer._pkg_logger.removeHandler(trainer._log_handler)


def test_emit_with_custom_format():
    trainer = _make_minimal_trainer(torch.device("cpu"), name="custom")
    trainer.epoch = 3
    trainer._pkg_logger.removeHandler(trainer._log_handler)
    trainer._log_handler.close()
    trainer._log_handler = TrainerLogHandler(
        trainer, level=logging.INFO, fmt="%(levelname)s|%(name)s|%(epoch)s|%(message)s"
    )
    trainer._pkg_logger.addHandler(trainer._log_handler)
    try:
        trainer.logger.info("hello")
        assert any(
            "INFO|splanything.training.trainer|3|hello" in m
            for m in trainer.logs[3]["msg"]
        )
    finally:
        trainer._log_handler.close()
        trainer._pkg_logger.removeHandler(trainer._log_handler)


# ---------------------------------------------------------------------------
# Filter integration
# ---------------------------------------------------------------------------


class _NameExactFilter(logging.Filter):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        return record.name == self.name


def test_emit_applies_attached_filter():
    trainer = _make_minimal_trainer(torch.device("cpu"), name="filt")
    trainer.epoch = 1
    trainer._pkg_logger.removeHandler(trainer._log_handler)
    trainer._log_handler.close()
    trainer._log_handler = TrainerLogHandler(
        trainer,
        level=logging.INFO,
        fmt="[%(levelname)s] %(message)s",
        filter=_NameExactFilter("splanything.training.trainer"),
    )
    trainer._pkg_logger.addHandler(trainer._log_handler)
    try:
        trainer.logger.info("kept")
        # A record from a different logger must be dropped.
        other = logging.getLogger("splanything.unrelated")
        other.info("dropped")

        msgs = trainer.logs[1]["msg"]
        assert any("kept" in m for m in msgs)
        assert not any("dropped" in m for m in msgs)
    finally:
        trainer._log_handler.close()
        trainer._pkg_logger.removeHandler(trainer._log_handler)


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_emit_does_not_crash_when_log_method_fails():
    class BrokenTrainer:
        logs = {}
        epoch = 1

        def log(self, msg):  # noqa: D401
            raise RuntimeError("boom")

    trainer = _make_minimal_trainer(torch.device("cpu"), name="robust")
    trainer._pkg_logger.removeHandler(trainer._log_handler)
    trainer._log_handler.close()
    handler = TrainerLogHandler(
        BrokenTrainer(), level=logging.INFO, fmt="[%(levelname)s] %(message)s"
    )
    trainer._pkg_logger.addHandler(handler)

    logging.raiseExceptions = False
    try:
        trainer.logger.info("should not crash")
    finally:
        logging.raiseExceptions = True
        trainer._pkg_logger.removeHandler(handler)
        handler.close()


def test_close_drops_trainer_reference():
    trainer = _make_minimal_trainer(torch.device("cpu"), name="close")
    handler = trainer._log_handler
    handler.close()
    assert handler.trainer is None
    trainer._pkg_logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# Integration: training loop generates real log records
# ---------------------------------------------------------------------------


def test_train_loop_emits_records_into_self_logs(device):
    """Run a short training loop and verify that ``_logger.info(...)``
    calls inside ``train`` (and any other trainer submodule reached via
    propagation) actually land in ``trainer.logs``."""
    trainer = _make_minimal_trainer(device, name="e2e")
    try:
        gen = trainer.train()
        next(gen)  # finishes epoch 1; self.epoch becomes 2 before yielding
        trainer.stop()
        try:
            next(gen)
        except StopIteration:
            pass

        msgs_final = sum((trainer.logs[e].get("msg", []) for e in trainer.logs), [])
        assert any("Checkpoint saved" in m for m in msgs_final), msgs_final
        assert any("Training ended" in m for m in msgs_final), msgs_final
        assert any("Logs saved" in m for m in msgs_final), msgs_final

        # Each message must be prefixed with both the severity and the
        # epoch that was current at format time.
        assert any("[INFO]" in m and "[epoch " in m for m in msgs_final), msgs_final
    finally:
        trainer._log_handler.close()
        trainer._pkg_logger.removeHandler(trainer._log_handler)


def test_logs_are_picked_up_from_module_logger_calls(device):
    """``_logger.info(...)`` calls inside trainer.py (e.g. from
    ``save_checkpoint``) must also reach the handler."""
    trainer = _make_minimal_trainer(device, name="modlog")
    try:
        trainer_mod._logger.info("direct module logger call")
        assert (
            trainer.logs[0]["msg"][-1] == "[INFO] [epoch 0] direct module logger call"
        )
    finally:
        trainer._log_handler.close()
        trainer._pkg_logger.removeHandler(trainer._log_handler)


def test_multiple_trainers_each_receive_their_own_logs(device):
    """Even though they share the package logger, each trainer's handler
    calls its OWN ``log`` method, so log records end up in the right
    trainer's ``self.logs``."""
    a = _make_minimal_trainer(device, name="multi_a")
    b = _make_minimal_trainer(device, name="multi_b")
    try:
        a.epoch = 1
        b.epoch = 2
        trainer_mod._logger.info("shared event")

        pkg = logging.getLogger("splanything")
        assert pkg.handlers.count(a._log_handler) == 1
        assert pkg.handlers.count(b._log_handler) == 1

        # The shared event lands in BOTH trainers' logs.
        assert any("shared event" in m for m in a.logs[1].get("msg", [])), a.logs
        assert any("shared event" in m for m in b.logs[2].get("msg", [])), b.logs
    finally:
        for t in (a, b):
            t._log_handler.close()
            t._pkg_logger.removeHandler(t._log_handler)


def test_logs_from_other_submodules_are_caught(device):
    """Records emitted on a sibling submodule logger (e.g.
    ``splanything.training.optimizer``) must propagate up to the package
    logger and reach the handler."""
    trainer = _make_minimal_trainer(device, name="othermod")
    trainer.epoch = 4
    try:
        # A record from a sibling submodule propagates up to ``splanything``.
        sibling = logging.getLogger("splanything.training.optimizer")
        sibling.info("from sibling module")
        sibling.warning("sibling warn")
        msgs = trainer.logs[4]["msg"]
        assert any("from sibling module" in m for m in msgs), msgs
        assert any("sibling warn" in m for m in msgs), msgs
    finally:
        trainer._log_handler.close()
        trainer._pkg_logger.removeHandler(trainer._log_handler)


def test_logs_from_primitives_subpackage_are_caught(device):
    """Records emitted on ``splanything.primitives.*`` propagate up to
    ``splanything`` and reach the handler."""
    trainer = _make_minimal_trainer(device, name="primmod")
    trainer.epoch = 5
    try:
        prim_logger = logging.getLogger("splanything.primitives.gaussian")
        prim_logger.info("from primitives subpackage")
        msgs = trainer.logs[5]["msg"]
        assert any("from primitives subpackage" in m for m in msgs), msgs
    finally:
        trainer._log_handler.close()
        trainer._pkg_logger.removeHandler(trainer._log_handler)


def test_unrelated_logger_outside_package_is_ignored(device):
    """Records from a logger outside ``splanything`` (e.g.
    ``some.other.lib``) do NOT propagate to the package logger and
    therefore must not be picked up by the handler."""
    trainer = _make_minimal_trainer(device, name="outside")
    trainer.epoch = 6
    try:
        outside = logging.getLogger("some.other.lib")
        outside.info("must not be caught")
        assert 6 not in trainer.logs or "msg" not in trainer.logs.get(6, {})
    finally:
        trainer._log_handler.close()
        trainer._pkg_logger.removeHandler(trainer._log_handler)
