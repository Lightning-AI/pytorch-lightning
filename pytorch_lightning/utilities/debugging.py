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
from typing import Any, Callable, Optional


def enabled_only(fn: Callable):
    """Decorate a logger method to run it only on the process with rank 0.

    Args:
        fn: Function to decorate
    """

    @wraps(fn)
    def wrapped_fn(self, *args, **kwargs):
        if self.enabled:
            fn(self, *args, **kwargs)

    return wrapped_fn


class InternalDebugger(object):

    def __init__(self, trainer):
        self.enabled = os.environ.get('PL_DEV_DEBUG', '0') == '1'
        self.trainer = trainer
        self.logged_metrics = []
        self.pbar_added_metrics = []
        self.saved_train_losses = []
        self.saved_val_losses = []
        self.saved_test_losses = []
        self.early_stopping_history = []
        self.checkpoint_callback_history = []
        self.events = []
        self.saved_lr_scheduler_updates = []
        self.train_dataloader_calls = []
        self.val_dataloader_calls = []
        self.test_dataloader_calls = []
        self.dataloader_sequence_calls = []

    def track_event(
            self,
            evt_type: str,
            evt_value: Any = None,
            global_rank: Optional[int] = None,
            local_rank: Optional[int] = None,
            comment: str = ''
    ) -> None:
        self.events.append({
            "timestamp": time.time(),
            "event": evt_type,
            "value": evt_value,
            "global_rank": global_rank,
            "local_rank": local_rank,
            "comment": comment,
        })

    def count_events(self, evt_type: str, strict=False) -> int:
        count = 0
        for evt in self.events:
            if strict and evt["event"] == evt_type:
                count += 1
            elif not strict and evt_type in evt["event"]:
                count += 1
        return count

    @enabled_only
    def track_load_dataloader_call(self, name, dataloaders):
        loader_counts = len(dataloaders)

        lengths = []
        for dl in dataloaders:
            try:
                length = len(dl)
            except Exception:
                length = -1
            lengths.append(length)

        values = {
            'global_step': self.trainer.global_step,
            'epoch': self.trainer.current_epoch,
            'num_loaders': loader_counts,
            'lengths': lengths,
            'name': name
        }

        # track the sequence in case we need to verify the sequence
        self.dataloader_sequence_calls.append(values)

        if 'train' in name:
            self.train_dataloader_calls.append(values)
        elif 'val' in name:
            self.val_dataloader_calls.append(values)
        elif 'test' in name:
            self.test_dataloader_calls.append(values)

    @enabled_only
    def track_logged_metrics_history(self, scalar_metrics):
        scalar_metrics['global_step'] = self.trainer.global_step
        self.logged_metrics.append(scalar_metrics)

    @enabled_only
    def track_train_loss_history(self, batch_idx, loss):
        loss_dict = {'batch_idx': batch_idx, 'epoch': self.trainer.current_epoch, 'loss': loss.detach()}
        self.saved_train_losses.append(loss_dict)

    @enabled_only
    def track_lr_schedulers_update(self, batch_idx, interval, scheduler_idx, old_lr, new_lr, monitor_key=None):
        loss_dict = {
            'batch_idx': batch_idx,
            'interval': interval,
            'scheduler_idx': scheduler_idx,
            'epoch': self.trainer.current_epoch,
            'monitor_key': monitor_key,
            'old_lr': old_lr,
            'new_lr': new_lr
        }
        self.saved_lr_scheduler_updates.append(loss_dict)

    @enabled_only
    def track_eval_loss_history(self, batch_idx, dataloader_idx, output):
        loss_dict = {
            'sanity_check': self.trainer.running_sanity_check,
            'dataloader_idx': dataloader_idx,
            'batch_idx': batch_idx,
            'epoch': self.trainer.current_epoch,
            'output': output
        }

        if self.trainer.testing:
            self.saved_test_losses.append(loss_dict)
        else:
            self.saved_val_losses.append(loss_dict)

    @enabled_only
    def track_pbar_metrics_history(self, metrics):
        metrics['debug_epoch'] = self.trainer.current_epoch
        self.pbar_added_metrics.append(metrics)

    @enabled_only
    def track_early_stopping_history(self, callback, current):
        debug_dict = {
            'epoch': self.trainer.current_epoch,
            'global_step': self.trainer.global_step,
            'rank': self.trainer.global_rank,
            'current': current,
            'best': callback.best_score,
            'patience': callback.wait_count
        }
        self.early_stopping_history.append(debug_dict)

    @enabled_only
    def track_checkpointing_history(self, filepath):
        cb = self.trainer.checkpoint_callback
        debug_dict = {
            'epoch': self.trainer.current_epoch,
            'global_step': self.trainer.global_step,
            'monitor': cb.monitor,
            'rank': self.trainer.global_rank,
            'filepath': filepath
        }
        self.checkpoint_callback_history.append(debug_dict)

    @property
    def num_seen_sanity_check_batches(self):
        count = len([x for x in self.saved_val_losses if x['sanity_check']])
        return count

    @property
    def num_seen_val_check_batches(self):
        counts = Counter()
        for x in self.saved_val_losses:
            if not x['sanity_check']:
                counts.update({x['dataloader_idx']: 1})
        return counts

    @property
    def num_seen_test_check_batches(self):
        counts = Counter()
        for x in self.saved_test_losses:
            if not x['sanity_check']:
                counts.update({x['dataloader_idx']: 1})
        return counts
