import json
import logging
import os
import torch

from typing import Sequence, Callable, Dict, Any, Optional, Union
from jaxtyping import Float
from torch import Tensor

from primitives import Primitive
from utils.lazy import clear_all_caches

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
        target: Float[Tensor, "B C H W"],
        primitive: Primitive,
        optimizer: torch.optim.Optimizer,
        losses: Dict[str, Callable],
        callbacks: Sequence[Callable],
        base_folder: Optional[str] = None,
        patch_size: Optional[int] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        refinements: Sequence[Callable] = (),
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Initialize trainer.

        Args:
            name: Name for this training run (used to create run folder).
            target: Target image tensor (B, C, H, W).
            primitive: Primitive to optimize.
            optimizer: Optimizer for primitive parameters.
            losses: Dict of loss functions {"name": Loss}.
            callbacks: List of Callback instances.
            base_folder: Base folder for saving runs (default: current directory).
            patch_size: Optional patch size for patch-based rendering.
            scheduler: Optional learning rate scheduler.
            refinements: List of FilterRule and SplitRule instances for epoch-end refinement.
            device: Optional device override. Uses get_device() fallback if None.
        """
        from utils.pytorch import get_device

        self.name = name
        self.base_folder = base_folder or "."
        self.run_folder = os.path.join(self.base_folder, name)
        os.makedirs(self.run_folder, exist_ok=True)

        device = device or get_device()
        if isinstance(device, str):
            device = torch.device(device)

        self.target = target.to(device)
        self.primitive = primitive.to(device)
        self.patch_size = patch_size
        self.optimizer = optimizer
        self.losses = losses
        self.callbacks = callbacks
        self.refinements = list(refinements)
        self.scheduler = scheduler
        self.logs: Dict[int, Dict[str, Any]] = dict()

        self.trainer_path = os.path.join(self.run_folder, "trainer.pt")
        self.primitive_path = os.path.join(self.run_folder, "primitive.pt")
        self.logs_path = os.path.join(self.run_folder, "logs.json")

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
        if epoch is not None:
            torch.save(
                self.state_dict(), self.trainer_path.replace(".pt", f"_e{epoch}.pt")
            )
            torch.save(
                self.primitive.state_dict(),
                self.primitive_path.replace(".pt", f"_e{epoch}.pt"),
            )
        else:
            torch.save(self.state_dict(), self.trainer_path)
            torch.save(self.primitive.state_dict(), self.primitive_path)
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
        self.primitive.prepare_for_optimization(
            target=self.target, patch_size=self.patch_size
        )
        self.should_continue = True
        self.call_back(TRAIN_START)
        self.epoch = 1
        try:
            while self.should_continue:
                clear_all_caches()
                self.exec_epoch()
                self.epoch += 1
                yield self.state_dict
        except KeyboardInterrupt:
            _logger.info("Training interrupted by user")
            self.stop()
        self.call_back(TRAIN_END)
        self.primitive.end_optimization()
        self._save_all()

    def exec_epoch(self) -> Dict[str, Any]:
        """Execute one training epoch.

        Runs: zero_grad -> forward -> compute losses -> backward -> step.
        Triggers callbacks at EPOCH_START, PRE_STEP, EPOCH_END.
        """
        self.call_back(EPOCH_START)
        self.optimizer.zero_grad()
        self.last_output = self.primitive.optim_step()
        self.last_losses = {name: l(self) for name, l in self.losses.items()}
        self.last_loss = sum(self.last_losses.values())
        self.last_loss.backward()
        self.call_back(PRE_STEP)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        with torch.no_grad():
            self.call_back(EPOCH_END)
            self._apply_refinements()

    def _apply_refinements(self):
        """Aggregate and apply filter and split rules.

        Collects all FilterRule and SplitRule from self.refinements,
        aggregates their masks, and applies the combined filter/split to the primitive.

        Filter rules are applied first, then split rules are computed on the
        filtered primitive and applied after.
        """
        filter_rules = []
        split_rules = []
        for c in self.refinements:
            if hasattr(c, "_filter_rule"):
                filter_rules.append(c)
            elif hasattr(c, "_split_rule"):
                split_rules.append(c)

        p = self.primitive

        # Apply filter/cull first
        if filter_rules:
            combined_keep = torch.ones(len(p), dtype=torch.bool, device=p.thetas.device)
            for rule in filter_rules:
                mask = rule.apply(self)
                if mask is not None:
                    combined_keep &= mask
            if (~combined_keep).any():
                print(f"FilterRule: {(~combined_keep).sum()} primitives culled")
                p.filter(combined_keep)
                self.optimizer.filter(p.parameters())

        # Compute and apply split on the filtered primitive
        if split_rules:
            combined_split = torch.zeros(
                len(p), dtype=torch.bool, device=p.thetas.device
            )
            for rule in split_rules:
                mask = rule.apply(self)
                if mask is not None:
                    combined_split |= mask
            if combined_split.any():
                print(f"SplitRule: {combined_split.sum()} primitives split")
                p.split(combined_split)
                # self.optimizer.split(p.parameters(), combined_split)
                self.optimizer.reinit(p.parameters())

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
