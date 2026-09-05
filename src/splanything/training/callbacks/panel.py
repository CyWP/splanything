"""Live rich-based statistics panel callback."""

import time
import torch
from collections import deque
from typing import List, Dict, Any, Optional

from ..trainer import Trainer
from ..stages import EPOCH_END, TRAIN_END

from .base import Callback


class StatsPanel(Callback):
    """Live terminal panel of training stats and recent logs (rich-based).

    Renders ``trainer.state_dict()`` into a table next to a scrolling log
    pane, refreshed at every EPOCH_END; the live display is stopped at
    TRAIN_END.

    Attributes:
        excl_kw: Stat keys hidden from the table.
        logs: Recent log messages (bounded deque).

    Stages: EPOCH_END, TRAIN_END
    """

    def __init__(self, excl_kw: Optional[List[str]] = None, max_logs: int = 10):
        """Start the rich live display.

        Args:
            excl_kw: Stat keys to exclude from the table.
            max_logs: Maximum number of recent log lines kept.
        """
        super().__init__()
        from rich.live import Live

        self.excl_kw = set() if excl_kw is None else set(excl_kw)
        self.start_time = time.time()
        self.logs = deque(maxlen=max_logs)
        self.live = Live(auto_refresh=False)
        self.live.start()

    _stages: List[str] = [EPOCH_END, TRAIN_END]

    def run(self, trainer: Trainer, stage: str):
        """Refresh the panel on EPOCH_END; stop the display on TRAIN_END.

        Args:
            trainer: Current trainer instance.
            stage: Current training stage.
        """
        if stage == TRAIN_END:
            self.live.stop()
            return
        self.live.update(self.render(trainer.state_dict()), refresh=True)

    def render(self, stats: Dict[str, Any]) -> "Columns":
        """Format trainer stats into a two-panel rich display.

        Args:
            stats: Stat dict from ``trainer.state_dict()``; ``"msg"`` is
                popped into the log pane and ``"time"`` is added.

        Returns:
            Rich ``Columns`` with a stats table panel and a logs panel.
        """
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
