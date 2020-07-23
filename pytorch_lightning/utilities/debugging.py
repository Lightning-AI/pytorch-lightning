import os


class InternalDebugger(object):

    def __init__(self, trainer):

        self.enabled = 'PL_DEV_DEBUG' in os.environ
        self.trainer = trainer
        self.logged_metrics = []
        self.pbar_added_metrics = []
        self.saved_losses = []
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
                'patience': es.wait_count
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
                'filepath': filepath
            }
            self.checkpoint_callback_history.append(debug_dict)
