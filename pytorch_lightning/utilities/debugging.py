# Copyright The PyTorch Lightning team.
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

import os
import time
from collections import Counter
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl


def enabled_only(fn: Callable) -> Optional[Callable]:
    """Decorate a logger method to run it only on the process with rank 0.

    Args:
        fn: Function to decorate
    """

    @wraps(fn)
    def wrapped_fn(self: Callable, *args: Any, **kwargs: Any) -> Optional[Any]:
        if self.enabled:
            fn(self, *args, **kwargs)
        return None

    return wrapped_fn


class InternalDebugger:
    def __init__(self, trainer: 'pl.Trainer') -> None:
        self.enabled = os.environ.get("PL_DEV_DEBUG", "0") == "1"
        self.trainer = trainer
        self.saved_train_losses: List[Dict[str, Union[float, torch.Tensor, object]]] = []
        self.saved_val_losses: List[Dict[str, Union[float, torch.Tensor, object]]] = []
        self.saved_test_losses: List[Dict[str, Union[float, torch.Tensor, object]]] = []
        self.early_stopping_history: List[Dict[str, Union[float, torch.Tensor, object]]] = []
        self.checkpoint_callback_history: List[Dict[str, Union[float, torch.Tensor, object]]] = []
        self.events: List[Dict[str, Any]] = []
        self.saved_lr_scheduler_updates: List[Dict[str, Union[int, float, str, torch.Tensor, None]]] = []
        self.train_dataloader_calls: List[Dict[str, Union[int, str, object]]] = []
        self.val_dataloader_calls: List[Dict[str, Union[int, str, object]]] = []
        self.test_dataloader_calls: List[Dict[str, Union[int, str, object]]] = []
        self.dataloader_sequence_calls: List[Dict[str, Union[int, str, object]]] = []

    @enabled_only
    def track_event(
        self,
        evt_type: str,
        evt_value: Any = None,
        global_rank: Optional[int] = None,
        local_rank: Optional[int] = None,
        comment: str = "",
    ) -> None:
        self.events.append(
            {
                "timestamp": time.time(),
                "event": evt_type,
                "value": evt_value,
                "global_rank": global_rank,
                "local_rank": local_rank,
                "comment": comment,
            }
        )

    @enabled_only
    def track_load_dataloader_call(self, name: str, dataloaders: List[DataLoader]) -> None:
        loader_counts = len(dataloaders)

        lengths = []
        for dl in dataloaders:
            try:
                length = len(dl)
            # todo: specify the possible exception
            except Exception:
                length = -1
            lengths.append(length)

        values = {
            "global_step": self.trainer.global_step,
            "epoch": self.trainer.current_epoch,
            "num_loaders": loader_counts,
            "lengths": lengths,
            "name": name,
        }

        # track the sequence in case we need to verify the sequence
        self.dataloader_sequence_calls.append(values)

        if "train" in name:
            self.train_dataloader_calls.append(values)
        elif "val" in name:
            self.val_dataloader_calls.append(values)
        elif "test" in name:
            self.test_dataloader_calls.append(values)

    @enabled_only
    def track_train_loss_history(self, batch_idx: int, loss: torch.Tensor) -> None:
        loss_dict = {"batch_idx": batch_idx, "epoch": self.trainer.current_epoch, "loss": loss.detach()}
        self.saved_train_losses.append(loss_dict)

    @enabled_only
    def track_lr_schedulers_update(
        self,
        batch_idx: int,
        interval: int,
        scheduler_idx: int,
        old_lr: float,
        new_lr: float,
        monitor_key: Optional[str] = None,
        monitor_val: Optional[torch.Tensor] = None,
    ) -> None:
        loss_dict = {
            "batch_idx": batch_idx,
            "interval": interval,
            "scheduler_idx": scheduler_idx,
            "epoch": self.trainer.current_epoch,
            "monitor_key": monitor_key,
            "monitor_val": monitor_val,
            "old_lr": old_lr,
            "new_lr": new_lr,
        }
        self.saved_lr_scheduler_updates.append(loss_dict)

    @enabled_only
    def track_eval_loss_history(self, batch_idx: int, dataloader_idx: int, output: torch.Tensor) -> None:
        loss_dict = {
            "sanity_check": self.trainer.sanity_checking,
            "dataloader_idx": dataloader_idx,
            "batch_idx": batch_idx,
            "epoch": self.trainer.current_epoch,
            "output": output,
        }

        if self.trainer.testing:
            self.saved_test_losses.append(loss_dict)
        else:
            self.saved_val_losses.append(loss_dict)

    @enabled_only
    def track_early_stopping_history(
        self,
        callback: 'pl.callbacks.early_stopping.EarlyStopping',
        current: torch.Tensor,
    ) -> None:
        debug_dict = {
            "epoch": self.trainer.current_epoch,
            "global_step": self.trainer.global_step,
            "rank": self.trainer.global_rank,
            "current": current,
            "best": callback.best_score,
            "patience": callback.wait_count,
        }
        self.early_stopping_history.append(debug_dict)

    @enabled_only
    def track_checkpointing_history(self, filepath: str) -> None:
        cb = self.trainer.checkpoint_callback
        debug_dict = {
            "epoch": self.trainer.current_epoch,
            "global_step": self.trainer.global_step,
            "monitor": cb.monitor,
            "rank": self.trainer.global_rank,
            "filepath": filepath,
        }
        self.checkpoint_callback_history.append(debug_dict)

    @property
    def num_seen_sanity_check_batches(self) -> int:
        count = sum(1 for x in self.saved_val_losses if x["sanity_check"])
        return count

    @property
    def num_seen_val_check_batches(self) -> Counter:
        counts: Counter = Counter()
        for x in self.saved_val_losses:
            if not x["sanity_check"]:
                counts.update({x["dataloader_idx"]: 1})
        return counts

    @property
    def num_seen_test_check_batches(self) -> Counter:
        counts: Counter = Counter()
        for x in self.saved_test_losses:
            if not x["sanity_check"]:
                counts.update({x["dataloader_idx"]: 1})
        return counts
