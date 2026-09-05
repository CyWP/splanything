"""Training loop orchestration with losses, callbacks, and refinement."""

import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor
from jaxtyping import Float

from ..primitives.base import Primitive
from ..rendering.sampler import Sampler
from ..utils.img import Splimage
from .losses.base import ImageLoss
from .optimizer import OptimizerWrapper
from .sampler import TrainSampler
from .stages import (
    PRE_STEP,
    TRAIN_END,
    TRAIN_START,
    EPOCH_END,
    EPOCH_START,
    BATCH_START,
    BATCH_END,
)

_logger = logging.getLogger(__name__)


class TrainerLogFormatter(logging.Formatter):
    """``logging.Formatter`` that injects the trainer's current epoch.

    The epoch is read at format time (not record-creation time), so it
    always reflects the trainer's current ``self.epoch`` when the
    handler processes the record. This means ``_logger.info(...)`` calls
    scattered across the training loop do not need to thread the epoch
    through ``extra={...}`` themselves — the formatter picks it up.

    Args:
        trainer: Trainer whose ``self.epoch`` is injected into each record.
        fmt: Optional format string. Use ``%(epoch)s`` to reference the
            injected value (e.g. ``"[%(levelname)s] [epoch %(epoch)s] %(message)s"``).
    """

    def __init__(self, trainer: "Trainer", fmt: Optional[str] = None):
        """Store the trainer reference and initialise the formatter.

        Args:
            trainer: Trainer whose ``self.epoch`` is injected into records.
            fmt: Optional format string (see class docstring).
        """
        super().__init__(fmt)
        self.trainer = trainer

    def format(self, record: logging.LogRecord) -> str:
        """Inject the trainer's current epoch and format the record.

        Args:
            record: Log record to format.

        Returns:
            Formatted message string.
        """
        if not hasattr(record, "epoch"):
            record.epoch = self.trainer.epoch
        return super().format(record)


class TrainerLogHandler(logging.Handler):
    """Logging handler that formats records and forwards them to ``Trainer.log``.

    Each emitted ``LogRecord`` is run through the handler's ``Formatter``
    (a :class:`TrainerLogFormatter` if a format string is provided, which
    automatically injects the trainer's current epoch) and then appended
    to the owning trainer's per-epoch log list via ``Trainer.log``.
    Exceptions raised inside ``Trainer.log`` are caught and routed through
    ``Handler.handleError`` so logging never crashes training.

    Attributes:
        trainer: Trainer whose ``log`` method receives each formatted record.

    Args:
        trainer: Trainer instance whose ``log`` method will receive messages.
        level: Minimum level for records to be forwarded (default ``INFO``).
        fmt: Optional format string passed to ``logging.Formatter``. When
            ``None`` the raw message is forwarded. The token ``%(epoch)s``
            is always populated by the handler's formatter with the
            trainer's current ``self.epoch``.
        filter: Optional ``logging.Filter`` (or callable) attached to the
            handler so callers can drop records by logger name, level, or
            any custom rule.
    """

    def __init__(
        self,
        trainer: "Trainer",
        level: int = logging.INFO,
        fmt: Optional[str] = None,
        filter: Optional[logging.Filter] = None,
    ):
        """Configure level, optional formatter, and optional filter."""
        super().__init__(level=level)
        self.trainer = trainer
        if fmt is not None:
            self.setFormatter(TrainerLogFormatter(trainer, fmt))
        if filter is not None:
            self.addFilter(filter)

    def emit(self, record: logging.LogRecord) -> None:
        """Format the record and forward it to ``Trainer.log``.

        Args:
            record: Log record to forward.
        """
        try:
            msg = self.format(record)
            self.trainer.log(msg)
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        """Detach from the trainer so ``Trainer.__dict__`` can be GC'd.

        The handler holds a strong reference to the trainer, which in turn
        holds the handler via ``self._log_handler``. Calling ``close``
        breaks the cycle explicitly.
        """
        self.trainer = None
        super().close()


class Trainer:
    """Training orchestrator for primitive-based image reconstruction.

    Manages the optimization loop, computing losses, executing callbacks,
    and saving state for inspection. Uses a generator pattern to allow
    step-by-step execution.

    Attributes:
        target: Ground truth image tensor (B, C, H, W).
        primitive: Trainable primitive to optimize.
        optimizer: PyTorch optimizer updating primitive parameters.
        scheduler: Optional learning rate scheduler.
        losses: Dict of loss functions to combine.
        callbacks: List of callbacks for monitoring and control.
        logs: Dict mapping epoch to logged metrics.
        run_folder: Path to folder for this training run.

    Notes:
        - Callbacks are triggered at TRAIN_START, TRAIN_END, EPOCH_START,
          EPOCH_END, BATCH_START, BATCH_END, PRE_STEP.
        - Use `stop()` to halt training early (e.g., from interrupt callback).
    """

    def __init__(
        self,
        name: str,
        primitive: Primitive,
        sampler: TrainSampler,
        optimizer: OptimizerWrapper,
        losses: Dict[str, Tuple[Callable, float]],
        callbacks: Sequence[Callable],
        base_folder: Optional[Path] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        low_vram: bool = False,
        adjust_prim: bool = True,
    ):
        """Initialise the trainer and create its run folder.

        Args:
            name: Run name; the run folder is ``base_folder / name``.
            primitive: Trainable primitive to optimize.
            sampler: TrainSampler providing target patches and coordinates.
            optimizer: OptimizerWrapper updating the primitive's parameters.
            losses: Dict mapping loss name to ``(loss_fn, weight)``.
            callbacks: Callbacks invoked at each training stage.
            base_folder: Base directory for checkpoints and logs.
            scheduler: Optional LR scheduler stepped once per epoch.
            low_vram: If True, rendering intermediates are moved to CPU.
            adjust_prim: If True, resize the primitive to the sampler
                canvas via ``adjust_to_canvas`` on init.

        Notes:
            - Warns when ``losses`` mixes per-sample ``Loss`` and
              full-image ``ImageLoss`` entries.
        """

        self.name = name
        self.base_folder = Path(base_folder) or Path(".")
        self.run_folder = self.base_folder / name
        self.run_folder.mkdir(parents=True, exist_ok=True)
        self.primitive = primitive
        self.sampler = sampler
        if adjust_prim:
            self.primitive.adjust_to_canvas(self.sampler.H, self.sampler.W)
        self.target = sampler.target_img
        self.optimizer = optimizer
        self.losses = losses
        self.callbacks = callbacks

        has_image = any(isinstance(fn, ImageLoss) for fn, _ in losses.values())
        has_sample = any(not isinstance(fn, ImageLoss) for fn, _ in losses.values())
        if has_image and has_sample:
            _logger.warning(
                "Mixed loss types: losses contain both Loss and ImageLoss. "
                "ImageLoss expects full BCHW images while Loss expects "
                "per-sample patches. This may produce incorrect results."
            )
        self._use_image_losses = has_image
        self.scheduler = scheduler
        self.logs: Dict[int, Dict[str, Any]] = dict()
        self.epoch: int = 0
        self.low_vram = low_vram

        self.trainer_path = self.run_folder / "trainer.pt"
        self.primitive_path = self.run_folder / "primitive.pt"
        self.logs_path = self.run_folder / "logs.json"

        self.logger = _logger
        self._pkg_logger = logging.getLogger("splanything")
        self._pkg_logger.setLevel(logging.INFO)
        self._log_handler = TrainerLogHandler(
            self,
            level=logging.INFO,
            fmt="[%(levelname)s] [epoch %(epoch)s] %(message)s",
        )
        self._pkg_logger.addHandler(self._log_handler)

    def call_back(self, stage: str):
        """Trigger all callbacks for a given stage.

        Args:
            stage: One of STAGES (TRAIN_START, TRAIN_END, etc.).
        """
        for c in self.callbacks:
            c(self, stage)

    def stop(self):
        """Halt training after current epoch completes."""
        self.should_continue = False

    def save_checkpoint(self, epoch: Optional[int] = None):
        """Save trainer and primitive state.

        Args:
            epoch: Optional epoch number for checkpoint naming.
        """
        primitive = self.primitive
        if epoch is not None:
            torch.save(
                self.state_dict(), self.trainer_path.replace(".pt", f"_e{epoch}.pt")
            )
            torch.save(
                primitive.state_dict(),
                self.primitive_path.replace(".pt", f"_e{epoch}.pt"),
            )
        else:
            torch.save(self.state_dict(), self.trainer_path)
            torch.save(primitive.state_dict(), self.primitive_path)
        _logger.info(f"Checkpoint saved to {self.run_folder}")

    def _save_logs(self):
        """Save logs to JSON file."""
        if self.logs_path is None:
            return
        try:
            logs_serializable = {
                str(epoch): {
                    k: v if not isinstance(v, torch.Tensor) else v.item()
                    for k, v in data.items()
                }
                for epoch, data in self.logs.items()
            }
            with open(self.logs_path, "w") as f:
                json.dump(logs_serializable, f, indent=2)
            _logger.info(f"Logs saved to {self.logs_path}")
        except Exception as e:
            _logger.error(f"Failed to save logs: {e}")

    def _save_all(self):
        """Save trainer state, primitive state, and logs."""
        self.save_checkpoint()
        self._save_logs()

    def train(self):
        """Run training loop as generator.

        Yields:
            None after each completed epoch; inspect state via
            ``state_dict()`` and ``last_image()``.
        """
        self.primitive.requires_grad_(True)
        self.should_continue = True
        self.call_back(TRAIN_START)
        self.epoch = 1
        try:
            while self.should_continue:
                self.exec_epoch()
                self.epoch += 1
                yield
        except KeyboardInterrupt:
            _logger.info("Training interrupted by user")
        finally:
            self.stop()
            self._save_all()
            _logger.info(f"Training ended: name={self.name}, epochs={self.epoch - 1}")
            self.call_back(TRAIN_END)

    def exec_epoch(self) -> Dict[str, Any]:
        """Execute one training epoch.

        Runs: zero_grad -> forward -> compute losses -> backward -> step.
        Triggers callbacks at EPOCH_START, PRE_STEP, EPOCH_END.

        Dispatches to per-sample or full-image path depending on whether
        any loss is an :class:`ImageLoss`.
        """
        self.last_epoch_image = None
        self.last_losses = {name: 0.0 for name in self.losses.keys()}
        self.last_regularizers = {}
        self.epoch_backward_passes = 0
        self.call_back(EPOCH_START)
        if self._use_image_losses:
            self._exec_epoch_image()
        else:
            self._exec_epoch_sample()
        self.call_back(PRE_STEP)
        with self._apply_refinements():
            self._compute_regularizers()
            self.optimizer.step()
            self.optimizer.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step()
        with torch.no_grad():
            self.call_back(EPOCH_END)

    def _exec_epoch_sample(self):
        """Per-sample training loop.

        Iterates over sampler patches, computing losses on each batch.
        """
        for gen, target, batch_co in self.sampler.samples(self.primitive):
            self.call_back(BATCH_START)
            self.last_output = gen
            self.last_target = target
            self._compute_losses(co=batch_co)
            self.call_back(BATCH_END)

    def _exec_epoch_image(self):
        """Full-image training loop.

        Renders the complete image, sets ``last_output`` and
        ``last_target`` as BCHW tensors, and computes losses once.
        """
        gen, target = self.sampler.rasterize(self.primitive)
        self.last_output = gen
        self.last_target = target.image()
        self.call_back(BATCH_START)
        self._compute_losses()
        self.call_back(BATCH_END)

    def _compute_losses(self, co: Optional[Float[Tensor, "B 2"]] = None):
        last_losses = {
            name: weight * loss_fn(self.last_output, self.last_target, co=co)
            for name, (loss_fn, weight) in self.losses.items()
        }
        last_loss = sum(last_losses.values())
        last_loss.backward()
        for name, loss in last_losses.items():
            self.last_losses[name] += loss.item()
        self.epoch_backward_passes += 1

    def _compute_regularizers(self):
        regs = self.primitive.compute_regularization()
        reg = sum(regs.values()) * float(self.epoch_backward_passes)
        if isinstance(reg, Tensor):
            reg.backward()
        for name, r in regs.items():
            self.last_regularizers[name] = r.item()

    def last_image(
        self,
        max_batch: Optional[int] = None,
        low_vram: Optional[bool] = None,
        sampler: Optional[Sampler] = None,
        force_grad: bool = False,
    ) -> Splimage:
        """Get the last rendered image, optionally recomputing it.

        Args:
            max_batch: Max batch size for rasterization.
            low_vram: Low-VRAM flag for rasterization.
            sampler: Optional sampler to rasterize with. If None, the
                trainer's sampler is used.
            force_grad: If True and a cached image exists but is detached
                (requires_grad=False), recompute it under the autograd
                graph.

        Returns:
            Rendered image Splimage object.
        """
        lv = self.low_vram if low_vram is None else low_vram
        if (
            sampler is None
            and self.last_epoch_image is not None
            and (not force_grad or self.last_epoch_image.requires_grad)
        ):
            return self.last_epoch_image
        if sampler is not None:
            return sampler.rasterize(self.primitive, max_batch=max_batch, low_vram=lv)
        img = self.sampler.rasterize(self.primitive, max_batch=max_batch, low_vram=lv)
        self.last_epoch_image = img
        return img

    @contextmanager
    def _apply_refinements(self):
        """Filter rules are applied first, then split rules are computed on the
        filtered primitive and applied after.
        Returned masks are used to update optimizer.
        """
        try:
            p = self.primitive
            keep = p.check_filter()
            split = p.check_split()
            yield
        finally:
            if keep is not None:
                p.filter(keep)
                self.optimizer.filter(p.param_groups(), keep)
                if split is not None:
                    split = split[keep]
            if split is not None:
                p.split(split)
                self.optimizer.split(p.param_groups(), split)

    def log_stat(self, key: str, val: Any):
        """Log a scalar value for current epoch.

        Args:
            key: Metric name.
            val: Metric value.
        """
        if self.epoch not in self.logs:
            self.logs[self.epoch] = {}
        self.logs[self.epoch][key] = val

    def log(self, msg: str):
        """Append message to current epoch's log.

        Args:
            msg: Message string to log.
        """
        if self.epoch not in self.logs:
            self.logs[self.epoch] = {}
        if self.logs[self.epoch].get("msg", None) is None:
            self.logs[self.epoch]["msg"] = []
        self.logs[self.epoch]["msg"].append(msg)

    def state_dict(self) -> Dict[str, Any]:
        """Current training state for inspection.

        Returns:
            Dict with epoch, primitive count, learning rate, per-loss and
            per-regularizer values, merged with the current epoch's
            logged metrics.
        """
        stats = {
            "Epoch": self.epoch,
            "Primitives": len(self.primitive),
            "Learning rate": self.optimizer.lr,
            **{f"Loss({k})": v for k, v in self.last_losses.items()},
            **{f"Reg({k})": v for k, v in self.last_regularizers.items()},
        }
        if self.epoch in self.logs:
            stats = {**stats, **self.logs[self.epoch]}
        return stats
