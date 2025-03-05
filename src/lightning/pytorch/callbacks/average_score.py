import logging
from typing import Callable

import torch
from torch import Tensor
from typing_extensions import override

import lightning.pytorch as pl
from lightning.pytorch.callbacks.callback import Callback

log = logging.getLogger(__name__)


class AverageScore(Callback):
    def __init__(
        self,
        monitor: str,
        score_name: str,
        window: int = 5,
        average_fn: Callable[[list[Tensor]], Tensor] = torch.mean,
        log_rank_zero_only: bool = False,
    ):
        super().__init__()
        self.monitor = monitor
        self.window = window
        self.average_fn = average_fn
        self.log_rank_zero_only = log_rank_zero_only
        self.scores: list[Tensor] = []
        self.score_name = score_name or f"average_{monitor}"

    @override
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logs = trainer.callback_metrics
        current = logs.get(self.monitor)

        if current is None:
            log.warning(f"Metric `{self.monitor}` not found in logs.")
            return

        self.scores.append(current)
        if len(self.scores) > self.window:
            self.scores.pop(0)

        average_score = self.average_fn(self.scores)
        self._log_info(trainer, f"{self.score_name} over last {self.window} logs: {average_score:.3f}")

    def _log_info(self, trainer: "pl.Trainer", message: str) -> None:
        rank = trainer.global_rank if trainer.world_size > 1 else None
        message = f"[Rank {rank}] {message}" if rank is not None else message
        if rank is None or not self.log_rank_zero_only or rank == 0:
            log.info(message)
