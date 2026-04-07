import torch

from typing import Sequence, Callable, Dict, Any, Optional
from jaxtyping import Float
from torch import Tensor

from primitives import Primitive
from utils.lazy import clear_all_caches

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

    Notes:
        - Callbacks are triggered at TRAIN_START, TRAIN_END, EPOCH_START, EPOCH_END, PRE_STEP.
        - Use `stop()` to halt training early (e.g., from interrupt callback).
    """

    def __init__(
        self,
        target: Float[Tensor, "B C H W"],
        primitive: Primitive,
        optimizer: torch.optim.Optimizer,
        losses: Dict[str, Callable],
        callbacks: Sequence[Callable],
        patch_size: Optional[int] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        """Initialize trainer.

        Args:
            target: Target image tensor (B, C, H, W).
            primitive: Primitive to optimize.
            optimizer: Optimizer for primitive parameters.
            losses: Dict of loss functions {"name": Loss}.
            callbacks: List of Callback instances.
            patch_size: Optional patch size for patch-based rendering.
            scheduler: Optional learning rate scheduler.
        """
        self.target = target
        self.primitive = primitive
        self.primitive.prepare_for_optimization(self.target, patch_size=patch_size)
        self.optimizer = optimizer
        self.losses = losses
        self.callbacks = callbacks
        self.scheduler = scheduler
        self.logs: Dict[int, Dict[str, Any]] = {}

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

    def train(self):
        """Run training loop as generator.

        Yields:
            State dict with current epoch, loss, and output for inspection.
        """
        self.should_continue = True
        self.call_back(TRAIN_START)
        self.epoch = 0
        while self.should_continue:
            self.epoch()
            self.epoch += 1
            yield self.state_dict
        self.call_back(TRAIN_END)

    def epoch(self) -> Dict[str, Any]:
        """Execute one training epoch.

        Runs: zero_grad -> forward -> compute losses -> backward -> step.
        Triggers callbacks at EPOCH_START, PRE_STEP, EPOCH_END.
        """
        clear_all_caches()
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
        self.call_back(EPOCH_END)

    def log_stat(self, key: str, val: Any):
        """Log a scalar value for current epoch.

        Args:
            key: Metric name.
            val: Metric value.
        """
        self.logs[self.epoch][key] = val

    def log(self, msg: str):
        """Append message to current epoch's log.

        Args:
            msg: Message string to log.
        """
        if self.logs[self.epoch].get("msg", None) is None:
            self.logs[self.epoch]["msg"] = ""
        self.logs[self.epoch]["msg"] += f"\n{msg}"

    @property
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
