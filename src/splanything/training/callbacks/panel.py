import time
import torch
from collections import deque
from typing import List, Dict, Any, Optional

from ..trainer import Trainer
from ..stages import EPOCH_END, TRAIN_END

from .base import Callback


class StatsPanel(Callback):
    """
    Stages: EPOCH_END, TRAIN_END
    """

    def __init__(self, excl_kw: Optional[List[str]] = None, max_logs: int = 10):
        super().__init__()
        from rich.live import Live

        self.excl_kw = set() if excl_kw is None else set(excl_kw)
        self.start_time = time.time()
        self.logs = deque(maxlen=max_logs)
        self.live = Live(auto_refresh=False)
        self.live.start()

    _stages: List[str] = [EPOCH_END, TRAIN_END]

    def run(self, trainer: Trainer, stage: str):
        if stage == TRAIN_END:
            self.live.stop()
            return
        self.live.update(self.render(trainer.state_dict()), refresh=True)

    def render(self, stats: Dict[str, Any]) -> "Columns":
        from rich.columns import Columns
        from rich.panel import Panel
        from rich.table import Table

        table = Table(show_header=False)
        table.add_column()
        table.add_column(justify="right")

        logs = stats.pop("msg") if "msg" in stats else []
        self.logs.extend(logs)

        stats["time"] = f"{(time.time() - self.start_time):.2f} seconds"

        for k, v in stats.items():
            if k in self.excl_kw:
                continue
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                v = v.clone().item()
            if isinstance(v, float):
                v = format(v, ".8f")
            else:
                v = str(v)
            table.add_row(k, v)

        return Columns(
            [
                Panel(table, title="Training"),
                Panel("\n".join(self.logs), title="Logs"),
            ],
            expand=True,
        )
