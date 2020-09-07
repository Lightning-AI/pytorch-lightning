from pytorch_lightning.utilities.exceptions import MisconfigurationException


class LRSchedulerConnector:

    def __init__(self, trainer):
        self.trainer = trainer

    def update_learning_rates(self, interval: str, monitor_metrics=None):
        """Update learning rates.

        Args:
            interval: either 'epoch' or 'step'.
            monitor_metrics: dict of possible values to monitor
        """
        if not self.trainer.lr_schedulers:
            return

        for scheduler_idx, lr_scheduler in enumerate(self.trainer.lr_schedulers):
            current_idx = self.trainer.batch_idx if interval == 'step' else self.trainer.current_epoch
            current_idx += 1  # account for both batch and epoch starts from 0
            # Take step if call to update_learning_rates matches the interval key and
            # the current step modulo the schedulers frequency is zero
            if lr_scheduler['interval'] == interval and current_idx % lr_scheduler['frequency'] == 0:
                # If instance of ReduceLROnPlateau, we need to pass validation loss
                if lr_scheduler['reduce_on_plateau']:
                    monitor_key = lr_scheduler['monitor']

                    if monitor_metrics is not None:
                        monitor_val = monitor_metrics.get(monitor_key)
                    else:
                        monitor_val = self.trainer.callback_metrics.get(monitor_key)

                    if monitor_val is None:
                        avail_metrics = ','.join(list(self.trainer.callback_metrics.keys()))
                        raise MisconfigurationException(
                            f'ReduceLROnPlateau conditioned on metric {monitor_key}'
                            f' which is not available. Available metrics are: {avail_metrics}.'
                            ' Condition can be set using `monitor` key in lr scheduler dict'
                        )
                    if self.trainer.dev_debugger.enabled:
                        old_lr = lr_scheduler['scheduler'].optimizer.param_groups[0]['lr']

                    # update LR
                    lr_scheduler['scheduler'].step(monitor_val)

                    if self.trainer.dev_debugger.enabled:
                        new_lr = lr_scheduler['scheduler'].optimizer.param_groups[0]['lr']
                        self.trainer.dev_debugger.track_lr_schedulers_update(
                            self.trainer.batch_idx,
                            interval,
                            scheduler_idx,
                            old_lr,
                            new_lr,
                            monitor_key,
                        )
                else:
                    if self.trainer.dev_debugger.enabled:
                        old_lr = lr_scheduler['scheduler'].optimizer.param_groups[0]['lr']

                    # update LR
                    lr_scheduler['scheduler'].step()

                    if self.trainer.dev_debugger.enabled:
                        new_lr = lr_scheduler['scheduler'].optimizer.param_groups[0]['lr']
                        self.trainer.dev_debugger.track_lr_schedulers_update(
                            self.trainer.batch_idx,
                            interval,
                            scheduler_idx,
                            old_lr, new_lr
                        )
