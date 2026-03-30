import torch

from typing import Sequence, Callable, Dict, Any, Optional

from primitives import Primitive

TRAIN_START = "train_start"
TRAIN_END = "train_end"
EPOCH_START = "epoch_start"
EPOCH_END = "epoch_end"
PRE_STEP = "pre_step"
STAGES = [TRAIN_START, TRAIN_END, EPOCH_START, EPOCH_END, PRE_STEP]

F = torch.FloatTensor


class Trainer:

    def __init__(
        self,
        target: F,
        primitive: Primitive,
        optimizer: torch.optim.Optimizer,
        losses: Dict[str, Callable],
        callbacks: Sequence[Callable],
        patch_size: Optional[int] = None,
    ):
        self.target = target
        self.primitive = primitive
        self.primitive.prepare_for_optimization(self.target, patch_size=patch_size)
        self.optimizer = optimizer
        self.losses = losses
        self.callbacks = callbacks
        self.logs: Dict[int, Dict[str, Any]] = {}

    def call_back(self, stage: str):
        for c in self.callbacks:
            c(self, stage)

    def stop(self):
        self.should_continue = False

    def train(self):
        self.should_continue = True
        self.call_back(TRAIN_START)
        self.epoch = 0
        while self.should_continue:
            self.epoch()
            self.epoch += 1
            yield self.state_dict
        self.call_back(TRAIN_END)

    def epoch(self) -> Dict[str, Any]:
        self.call_back(EPOCH_START)
        self.optimizer.zero_grad()
        self.last_output = self.primitive.optim_step()
        self.last_losses = {name: l(self) for name, l in self.losses.items()}
        self.last_loss = sum(self.last_losses.values())
        self.last_loss.backward()
        self.call_back(PRE_STEP)
        self.optimizer.step()
        self.call_back(EPOCH_END)

    def log_stat(self, key: str, val: Any):
        self.logs[self.epoch][key] = val

    def log(self, msg: str):
        if self.logs[self.epoch].get("msg", None) is None:
            self.logs[self.epoch]["msg"] = ""
        self.logs[self.epoch]["msg"] += f"\n{msg}"
