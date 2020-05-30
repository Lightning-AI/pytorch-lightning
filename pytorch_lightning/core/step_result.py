from typing import Optional, Dict
from torch import Tensor
from collections import OrderedDict
import torch


class Result(Dict):

    def __init__(self,
                 minimize:Optional[Tensor] = None,
                 logs: Optional[Dict] = None,
                 early_stop_on: Tensor = None,
                 checkpoint_on: Tensor = None,
                 progress_bar: Optional[Dict] = None,
                 hiddens: Optional[Tensor] = None):
        """
        Result is an OrderedDict that gives type hints, allowed fields and validation for bad user input.

        Use as the return value for:
        - training_step
        - validation_epoch_end
        - training_epoch_end

        .. note:: Plain dictionary returns are supported but are more prone to errors

        We automatically detach anything here for you to avoid holding references to graphs

        Args:
            minimize: Metric to minimize
            logs: dictionary that will be added to your logger(s)
            early_stop_on: Metric for early stopping. If none set, will use minimize by default.
            checkpoint_on: Metric for checkpointing. If none set, will use minimize by default.
            progress_bar: dictionary of values to add to the progress bar
            hiddens: tensor of hiddens to pass to next step when using TBPTT

        .. code-block: python

            # all options:
            def training_step(...):
                return Result(
                    minimize=loss,
                    checkpoint_on=loss,
                    early_stop_on=loss,
                    logs={'train_loss': loss},
                    progress_bar={'train_loss': loss}
                )

            # most of the time
            # will early stop and save checkpoints based on this metric by default
            return Result(loss)

            # to change what to early stop on
            return Result(loss, early_stop_on=accuracy)

            # to change what to checkpoint on
            return Result(loss, early_stop_on=accuracy, checkpoint_on=bleu_score)

            # shorthand for logging
            result = Result(loss)
            result.log('train_nce_loss', loss)

            # shorthand to put on progress bar
            result.to_bar('train_nce_loss', loss)
        """
        super().__init__()

        self.logs = logs
        self.early_stop_on = early_stop_on
        self.checkpoint_on = checkpoint_on
        self.progress_bar = progress_bar
        self.hiddens = hiddens
        self.minimize = minimize

    def __reduce_on_callback(self, callback_name, name, metric, log, pbar, reduce_fx):
        assert isinstance(metric, torch.Tensor), f'{name} must be a torch.Tensor'

        keys = [f'reduce_{callback_name}']
        if log:
            keys.append(f'log_{callback_name}')
        if pbar:
            keys.append(f'pbar_{callback_name}')

        for key in keys:
            if key not in self:
                self[key] = {}

            if 'log' in key or 'pbar' in key:
                metric = metric.detach()

            metrics = self[key]
            metrics[name] = metric

        key = f'reduce_fx_{callback_name}'
        if key not in self:
            self[key] = {}

        metrics = self[key]
        metrics[name] = reduce_fx

    def to_pbar(self, name: str, value: Tensor, on_batch_end=False, on_epoch_end=True, reduce_fx=torch.mean):
        if on_batch_end:
            self.__reduce_on_callback('on_batch_end', name, value, log=False, pbar=True, reduce_fx=reduce_fx)
        if on_epoch_end:
            self.__reduce_on_callback('on_epoch_end', name, value, log=False, pbar=True, reduce_fx=reduce_fx)

    def log(self, name: str, value: Tensor, on_batch_end=False, on_epoch_end=True, reduce_fx=torch.mean):
        if on_batch_end:
            self.__reduce_on_callback('on_batch_end', name, value, log=True, pbar=False, reduce_fx=reduce_fx)
        if on_epoch_end:
            self.__reduce_on_callback('on_epoch_end', name, value, log=True, pbar=False, reduce_fx=reduce_fx)

    @property
    def progress_bar(self):
        return self.__getitem__('progress_bar')

    @progress_bar.setter
    def progress_bar(self, x):
        if x is not None:
            assert isinstance(x, dict), 'progress_bar_logs must be a dict'
            self.__setitem__('progress_bar', x)

    @property
    def logs(self):
        return self.__getitem__('logs')

    @logs.setter
    def logs(self, x):
        if x is not None:
            assert isinstance(x, dict), 'logs must be a dict'
            self.__setitem__('logs', x)

    @property
    def hiddens(self):
        return self._hiddens

    @hiddens.setter
    def hiddens(self, x):
        if x is not None:
            assert isinstance(x, Tensor), 'hiddens must be a torch.Tensor'
            self._hiddens = x
            self.__setitem__('hiddens', x)

    @property
    def checkpoint_on(self):
        # use minimize as default if no checkpoint_on is passed
        if 'checkpoint_on' not in self:
            minimize = self.__getitem__('minimize')
            self.__setitem__('checkpoint_on', minimize)

        return self.__getitem__('checkpoint_on')

    @checkpoint_on.setter
    def checkpoint_on(self, x):
        if x is not None:
            assert isinstance(x, Tensor), 'checkpoint_on must be a torch.Tensor'
            self._checkpoint_on = x
            self.__setitem__('checkpoint_on', self._checkpoint_on)

    @property
    def early_stop_on(self):
        # use minimize as default if no checkpoint_on is passed
        if 'early_stop_on' not in self:
            minimize = self.__getitem__('minimize')
            self.__setitem__('early_stop_on', minimize)

        return self.__getitem__('early_stop_on')

    @early_stop_on.setter
    def early_stop_on(self, x):
        if x is not None:
            assert isinstance(x, Tensor), 'early_stop_on must be a torch.Tensor'
            self.__setitem__('early_stop_on', x)

    @property
    def minimize(self):
        return self.__getitem__('minimize')

    @minimize.setter
    def minimize(self, x):
        if x is not None:
            assert isinstance(x, Tensor), 'metric to minimize must be a torch.Tensor'
            self.__setitem__('minimize', x)


class TrainResult(Result):
    def log(self, name: str, value: Tensor, on_batch_end=True, on_epoch_end=False, reduce_fx=torch.mean):
        # no graph pointers for logs
        value = value.detach()
        super().log(name, value, on_batch_end, on_epoch_end, reduce_fx)

    def to_pbar(self, name: str, value: Tensor, on_batch_end=True, on_epoch_end=False, reduce_fx=torch.mean):
        # no graph pointers for progress bar
        value = value.detach()
        super().to_pbar(name, value, on_batch_end, on_epoch_end, reduce_fx)


class EvalResult(Result):
    def log(self, name: str, value: Tensor, on_batch_end=False, on_epoch_end=True, reduce_fx=torch.mean):
        # no graph pointers for logs
        value = value.detach()
        super().log(name, value, on_batch_end, on_epoch_end, reduce_fx)

    def to_pbar(self, name: str, value: Tensor, on_batch_end=False, on_epoch_end=True, reduce_fx=torch.mean):
        # no graph pointers for progress bar
        value = value.detach()
        super().to_pbar(name, value, on_batch_end, on_epoch_end, reduce_fx)


if __name__ == '__main__':
    import torch
    result = Result()
    result.minimize = torch.tensor(1)