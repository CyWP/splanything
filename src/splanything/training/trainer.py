import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import torch
from jaxtyping import Float, Tensor

from ..primitives import Primitive
from ..rendering import Sampler
from .optimizer import OptimizerWrapper
from .sampler import TrainSampler

_logger = logging.getLogger(__name__)

TRAIN_START = "train_start"
TRAIN_END = "train_end"
EPOCH_START = "epoch_start"
EPOCH_END = "epoch_end"
PRE_STEP = "pre_step"
STAGES = [TRAIN_START, TRAIN_END, EPOCH_START, EPOCH_END, PRE_STEP]


class Trainer:
    """Training orchestrator for primitive-based image reconstruction.

    Manages the optimization loop, computing losses, executing callbacks,
    and yielding state for real-time inspection. Uses a generator pattern
    to allow step-by-step execution with ComfyUI integration.

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
        - Callbacks are triggered at TRAIN_START, TRAIN_END, EPOCH_START, EPOCH_END, PRE_STEP.
        - Use `stop()` to halt training early (e.g., from interrupt callback).
    """

    def __init__(
        self,
        name: str,
        primitive: Primitive,
        sampler: TrainSampler,
        optimizer: OptimizerWrapper,
        losses: Dict[str, Callable],
        callbacks: Sequence[Callable],
        batch_size: Optional[int] = None,
        base_folder: Optional[Path] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        low_vram: bool = False,
    ):

        self.name = name
        self.base_folder = base_folder or Path(".")
        self.run_folder = self.base_folder / name
        self.run_folder.mkdir(parents=True, exist_ok=True)

        self.primitive = primitive
        self.sampler = sampler
        self.target = sampler.target_img
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.losses = losses
        self.callbacks = callbacks
        self.scheduler = scheduler
        self.logs: Dict[int, Dict[str, Any]] = dict()
        self.low_vram = low_vram

        self.trainer_path = self.run_folder / "trainer.pt"
        self.primitive_path = self.run_folder / "primitive.pt"
        self.logs_path = self.run_folder / "logs.json"

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
            State dict with current epoch, loss, and output for inspection.
        """
        self.primitive.requires_grad_(True)
        self.should_continue = True
        self.call_back(TRAIN_START)
        self.epoch = 1
        try:
            while self.should_continue:
                self.exec_epoch()
                self.epoch += 1
                yield self.state_dict()
        except KeyboardInterrupt:
            _logger.info("Training interrupted by user")
            self.stop()
        self.call_back(TRAIN_END)
        self._save_all()

    def exec_epoch(self) -> Dict[str, Any]:
        """Execute one training epoch.

        Runs: zero_grad -> forward -> compute losses -> backward -> step.
        Triggers callbacks at EPOCH_START, PRE_STEP, EPOCH_END.
        """
        self.last_epoch_image = None
        self.call_back(EPOCH_START)
        self._zero_grad()
        patch_count = 0
        for gen, target, batch_co in self.sampler.samples(self.primitive):
            self.last_output = gen
            self.last_target = target
            self._update_losses(co=batch_co)
            self.last_loss.backward()
            patch_count += 1
            step = self.batch_size is None or patch_count % self.batch_size == 0
            if step:
                self.call_back(PRE_STEP)
                self.optimizer.step()
                self._zero_grad()
        if self.scheduler is not None:
            self.scheduler.step()
        with torch.no_grad():
            self.call_back(EPOCH_END)
            self._apply_refinements()

    def _zero_grad(self):
        self.optimizer.zero_grad()
        self.last_losses = {name: 0.0 for name in self.losses.keys()}
        self.last_loss = torch.tensor(0.0)

    def last_image(
        self,
        max_batch: Optional[int] = None,
        low_vram: Optional[bool] = None,
        sampler: Optional[Sampler] = None,
        force_grad: bool = False,
    ) -> Float[Tensor, "B C H W"]:
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
            Rendered image tensor (B, C, H, W).
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

    def _update_losses(self, co: Float[Tensor, "B 2"]):
        last_losses = {
            name: loss_fn(self, co=co) for name, loss_fn in self.losses.items()
        }
        last_loss = sum(last_losses.values())
        for name, loss in last_losses.items():
            self.last_losses[name] = self.last_losses[name] + loss
        self.last_loss = self.last_loss + last_loss

    @torch.no_grad()
    def _apply_refinements(self):
        """Filter rules are applied first, then split rules are computed on the
        filtered primitive and applied after.
        Returned masks are sued to update optimizer.
        """
        p = self.primitive

        keep = p.check_filter()
        if keep is not None:
            self.optimizer.filter(p.param_groups(), keep)

        # Compute and apply split on the filtered primitive
        split = p.check_split()

        finetuned = p.check_finetune()

        if finetuned or split is not None:
            self.optimizer.reinit(p.param_groups())  # ToDo: split op for optimizer

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
            self.logs[self.epoch]["msg"] = ""
        self.logs[self.epoch]["msg"] += f"\n{msg}"

    def state_dict(self) -> Dict[str, Any]:
        """Current training state for inspection.

        Returns:
            Dict with epoch, loss, output tensor, and logs.
        """
        return {
            "epoch": self.epoch,
            "loss": self.last_loss,
            "output": self.last_output,
            "logs": self.logs,
        }
