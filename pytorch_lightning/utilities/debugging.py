import os
from collections import Counter


class InternalDebugger(object):
    def __init__(self, trainer):

        self.enabled = 'PL_DEV_DEBUG' in os.environ
        self.trainer = trainer
        self.logged_metrics = []
        self.pbar_added_metrics = []
        self.saved_losses = []
        self.saved_val_losses = []
        self.saved_test_losses = []
        self.early_stopping_history = []
        self.checkpoint_callback_history = []

    def track_logged_metrics_history(self, scalar_metrics):
        if self.enabled:
            scalar_metrics['global_step'] = self.trainer.global_step
            self.logged_metrics.append(scalar_metrics)

    def track_train_loss_history(self, batch_idx, loss):
        if self.enabled:
            loss_dict = {'batch_idx': batch_idx, 'epoch': self.trainer.current_epoch, 'loss': loss.detach()}
            self.saved_losses.append(loss_dict)

    def track_eval_loss_history(self, test_mode, batch_idx, dataloader_idx, output):
        if self.enabled:
            loss_dict = {
                'sanity_check': self.trainer.running_sanity_check,
                'dataloader_idx': dataloader_idx,
                'batch_idx': batch_idx,
                'epoch': self.trainer.current_epoch,
                'output': output,
            }

            if test_mode:
                self.saved_test_losses.append(loss_dict)
            else:
                self.saved_val_losses.append(loss_dict)

    def track_pbar_metrics_history(self, metrics):
        if self.enabled:
            metrics['debug_epoch'] = self.trainer.current_epoch
            self.pbar_added_metrics.append(metrics)

    def track_early_stopping_history(self, current):
        if self.enabled:
            es = self.trainer.early_stop_callback
            debug_dict = {
                'epoch': self.trainer.current_epoch,
                'global_step': self.trainer.global_step,
                'rank': self.trainer.global_rank,
                'current': current,
                'best': es.best_score,
                'patience': es.wait_count,
            }
            self.early_stopping_history.append(debug_dict)

    def track_checkpointing_history(self, filepath):
        if self.enabled:
            cb = self.trainer.checkpoint_callback
            debug_dict = {
                'epoch': self.trainer.current_epoch,
                'global_step': self.trainer.global_step,
                'monitor': cb.monitor,
                'rank': self.trainer.global_rank,
                'filepath': filepath,
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
