# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
import time
import warnings
from typing import List, Optional, Sequence

from lightning.pytorch.callbacks.callback import Callback

from .plugins import BaseEventPlugin, SupportsOnEvent
from .types import EventRecord

log = logging.getLogger(__name__)


class EventLogger(Callback):
    """Dispatcher that converts Trainer lifecycle hooks into EventRecord emissions for plugins.

    Args:
        plugins: A sequence (list or tuple) of plugin instances to receive events in the exact given order.
        dry_run: If True, events are dropped and plugins are not invoked.
    """

    def __init__(self, *, plugins: Optional[Sequence[SupportsOnEvent]] = None, dry_run: bool = False) -> None:
        super().__init__()
        # normalize to list and preserve order
        self._plugins: List[SupportsOnEvent] = list(plugins) if plugins is not None else []
        self._dry_run = bool(dry_run)
        self._quarantined: set[int] = set()  # indices of plugins removed due to failures

    # -------------- internal helpers --------------
    def _emit(self, type_: str, metadata: Optional[dict] = None, duration: Optional[float] = None) -> None:
        if self._dry_run or not self._plugins:
            return
        event = EventRecord(type=type_, timestamp=time.time(), metadata=metadata or {}, duration=duration)
        # dispatch in the provided order, quarantining faulty plugins
        for idx, plugin in enumerate(self._plugins):
            if idx in self._quarantined:
                continue
            try:
                plugin.on_event(event)
            except Exception as ex:  # noqa: BLE001 - isolate faults
                # quarantine this plugin for the rest of the run and warn
                self._quarantined.add(idx)
                msg = f"EventLogger: quarantining plugin {type(plugin).__name__} after exception: {ex}"
                try:
                    log.warning(msg)
                finally:
                    warnings.warn(msg)

    # -------------- training hooks --------------
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx: int) -> None:  # type: ignore[override]
        # "forward" lifecycle marker
        self._emit("forward", {"stage": "train", "batch_idx": batch_idx})

    def on_after_backward(self, trainer, pl_module) -> None:  # type: ignore[override]
        self._emit("backward", {"stage": "train", "global_step": getattr(trainer, "global_step", None)})

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:  # type: ignore[override]
        self._emit(
            "optimizer_step",
            {
                "stage": "train",
                "optimizer": type(optimizer).__name__,
                "global_step": getattr(trainer, "global_step", None),
            },
        )

    # -------------- validation hooks (metrics proxy) --------------
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:  # type: ignore[override]
        # Use validation batch end as a proxy for metric emission.
        # Emit only when validation was explicitly requested (e.g., via limit_val_batches) or checkpointing is enabled.
        limit_val_batches = getattr(trainer, "limit_val_batches", 1.0)
        explicit_val = isinstance(limit_val_batches, int) or limit_val_batches != 1.0
        has_ckpt = bool(getattr(trainer, "checkpoint_callbacks", ()))
        if explicit_val or has_ckpt:
            self._emit(
                "metric",
                {"stage": "validation", "batch_idx": batch_idx, "dataloader_idx": dataloader_idx},
            )

    # -------------- checkpoint hooks --------------
    def on_save_checkpoint(self, trainer, pl_module, checkpoint: dict) -> None:  # type: ignore[override]
        self._emit(
            "checkpoint",
            {"epoch": getattr(trainer, "current_epoch", None), "global_step": getattr(trainer, "global_step", None)},
        )
