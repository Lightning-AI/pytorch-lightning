r"""

Logging of learning rates
=========================

Log learning rate for lr schedulers during training

"""

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class LearningRateLogger(Callback):
    r"""
    Automatically logs learning rate for learning rate schedulers during training.

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import LearningRateLogger
        >>> lr_logger = LearningRateLogger()
        >>> trainer = Trainer(callbacks=[lr_logger])

    Logging names are automatically determined based on optimizer class name.
    In case of multiple optimizers of same type, they will be named `Adam`,
    `Adam-1` ect. If a optimizer have multiple parameter groups they will
    be named `Adam/pg1`, `Adam/pg2` ect. To control naming, pass in a
    `name` keyword in the construction of the learning rate schdulers

    Example::

        def configure_optimizer(self):
            optimizer = torch.optim.Adam(...)
            lr_scheduler = {'scheduler': torch.optim.lr_schedulers.LambdaLR(optimizer, ...)
                            'name': 'my_logging_name'}
            return [optimizer], [lr_scheduler]
    """
    def __init__(self):
        self.lrs = {}
        self.names = []

    def on_train_start(self, trainer, pl_module):
        """ Called before training, determines unique names for all lr
            schedulers in the case of multiple of the same type or in
            the case of multiple parameter groups
        """
        if trainer.lr_schedulers == []:
            raise MisconfigurationException(
                'Cannot use LearningRateLogger callback with models that have no'
                ' learning rate schedulers. Please see documentation for'
                ' `configure_optimizers` method.')

        if not trainer.logger:
            raise MisconfigurationException(
                'Cannot use LearningRateLogger callback with Trainer that have no logger.')

        # Create uniqe names in the case we have multiple of the same learning
        # rate schduler + multiple parameter groups
        names = []
        for scheduler in trainer.lr_schedulers:
            sch = scheduler['scheduler']
            if 'name' in scheduler:
                name = scheduler['name']
            else:
                opt_name = 'lr-' + sch.optimizer.__class__.__name__
                name = opt_name
                counter = 0
                # Multiple schduler of the same type
                while True:
                    counter += 1
                    if name in names:
                        name = opt_name + '-' + str(counter)
                    else:
                        break

            # Multiple param groups for the same schduler
            param_groups = sch.optimizer.param_groups
            if len(param_groups) != 1:
                for i, pg in enumerate(param_groups):
                    temp = name + '/pg' + str(i + 1)
                    names.append(temp)
            else:
                names.append(name)

            self.names.append(name)

        # Initialize for storing values
        for name in names:
            self.lrs[name] = []

    def on_batch_start(self, trainer, pl_module):
        latest_stat = self._extract_lr(trainer, 'step')
        if trainer.logger and latest_stat != {}:
            trainer.logger.log_metrics(latest_stat, step=trainer.global_step)

    def on_epoch_start(self, trainer, pl_module):
        latest_stat = self._extract_lr(trainer, 'epoch')
        if trainer.logger and latest_stat != {}:
            trainer.logger.log_metrics(latest_stat, step=trainer.global_step)

    def _extract_lr(self, trainer, interval):
        """ Extracts learning rates for lr schedulers and save information
            into dict structure. """
        latest_stat = {}
        for name, scheduler in zip(self.names, trainer.lr_schedulers):
            if scheduler['interval'] == interval:
                param_groups = scheduler['scheduler'].optimizer.param_groups
                if len(param_groups) != 1:
                    for i, pg in enumerate(param_groups):
                        lr = pg['lr']
                        self.lrs[name + '/' + str(i + 1)].append(lr)
                        latest_stat[name + '/' + str(i + 1)] = lr
                else:
                    self.lrs[name].append(param_groups[0]['lr'])
                    latest_stat[name] = param_groups[0]['lr']
        return latest_stat
