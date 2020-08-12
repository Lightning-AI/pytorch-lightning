import os
import time
from collections import Counter
from functools import wraps
from typing import Callable


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

        self.enabled = 'PL_DEV_DEBUG' in os.environ
        self.trainer = trainer
        self.logged_metrics = []
        self.pbar_added_metrics = []
        self.saved_train_losses = []
        self.saved_val_losses = []
        self.saved_test_losses = []
        self.early_stopping_history = []
        self.checkpoint_callback_history = []
        self.events = []

    def track_event(self, evt_type, evt_value=None, global_rank=None, local_rank=None, comment=''):
        self.events.append({
            "timestamp": time.time(),
            "event": evt_type,
            "value": evt_value,
            "global_rank": global_rank,
            "local_rank": local_rank,
            "comment": comment,
        })

    @enabled_only
    def track_logged_metrics_history(self, scalar_metrics):
        scalar_metrics['global_step'] = self.trainer.global_step
        self.logged_metrics.append(scalar_metrics)

    @enabled_only
    def track_train_loss_history(self, batch_idx, loss):
        loss_dict = {'batch_idx': batch_idx, 'epoch': self.trainer.current_epoch, 'loss': loss.detach()}
        self.saved_train_losses.append(loss_dict)

    @enabled_only
    def track_eval_loss_history(self, test_mode, batch_idx, dataloader_idx, output):
        loss_dict = {
            'sanity_check': self.trainer.running_sanity_check,
            'dataloader_idx': dataloader_idx,
            'batch_idx': batch_idx,
            'epoch': self.trainer.current_epoch,
            'output': output
        }

        if test_mode:
            self.saved_test_losses.append(loss_dict)
        else:
            self.saved_val_losses.append(loss_dict)

    @enabled_only
    def track_pbar_metrics_history(self, metrics):
        metrics['debug_epoch'] = self.trainer.current_epoch
        self.pbar_added_metrics.append(metrics)

    @enabled_only
    def track_early_stopping_history(self, current):
        es = self.trainer.early_stop_callback
        debug_dict = {
            'epoch': self.trainer.current_epoch,
            'global_step': self.trainer.global_step,
            'rank': self.trainer.global_rank,
            'current': current,
            'best': es.best_score,
            'patience': es.wait_count
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
