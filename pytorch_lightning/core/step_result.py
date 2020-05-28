from typing import Optional, Dict
from torch import Tensor
from collections import OrderedDict


class Result(OrderedDict):

    def __init__(self,
                 logs: Optional[Dict] = None,
                 early_stop_on: Tensor = None,
                 checkpoint_on: Tensor = None,
                 progress_bar: Optional[Dict] = None,
                 hiddens: Optional[Tensor] = None):
        super().__init__()

        self.logs = logs
        self.early_stop_on = early_stop_on
        self.checkpoint_on = checkpoint_on
        self.progress_bar = progress_bar
        self.hiddens = hiddens

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

    def log(self, key, value):
        if 'logs' not in self:
            self.__setitem__('logs', {})

        logs = self.__getitem__('logs')
        logs[key] = value

    def display(self, key, value):
        if 'progress_bar' not in self:
            self.__setitem__('progress_bar', {})

        progress_bar = self.__getitem__('progress_bar')
        progress_bar[key] = value

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
        return self._checkpoint_on

    @checkpoint_on.setter
    def checkpoint_on(self, x):
        if x is not None:
            assert isinstance(x, Tensor), 'checkpoint_on must be a torch.Tensor'
            self._checkpoint_on = x
            self.__setitem__('checkpoint_on', self._checkpoint_on)

    @property
    def early_stop_on(self):
        return self._early_stop_on

    @early_stop_on.setter
    def early_stop_on(self, x):
        if x is not None:
            assert isinstance(x, Tensor), 'early_stop_on must be a torch.Tensor'
            self._early_stop_on = x
            self.__setitem__('early_stop_on', self._early_stop_on)


class TrainStepResult(Result):
    """
    A dictionary with type checking and error checking
    Return this in the training step.
    """

    def __init__(self,
                 early_stop_on: Tensor = None,
                 minimize: Tensor = None,
                 checkpoint_on: Tensor = None,
                 logs: Optional[Dict] = None,
                 progress_bar: Optional[Dict] = None,
                 hiddens: Optional[Tensor] = None):
        """

        Args:
            minimize: the metric to minimize (usually the loss) (Tensor)
            logs: a dictionary to pass to the logger
            progress_bar: a dictionary to pass to the progress bar
            hiddens: when using TBPTT return the hidden states here
        """
        super().__init__(logs, early_stop_on, checkpoint_on, progress_bar, hiddens)
        self.minimize = minimize

    @property
    def minimize(self):
        return self._minimize

    @minimize.setter
    def minimize(self, x):
        if x is not None:
            assert isinstance(x, Tensor), 'metric to minimize must be a torch.Tensor'

        self._minimize = x
        self.__setitem__('minimize', x)


class EvalStepResult(Result):
    """
    Return this in the validation step
    """

    def __init__(self,
                 early_stop_on: Tensor = None,
                 checkpoint_on: Tensor = None,
                 logs: Optional[Dict] = None,
                 progress_bar: Optional[Dict] = None,
                 hiddens: Optional[Tensor] = None):
        """

        Args:
            early_stop_on: the metric used for early Stopping
            checkpoint_on: the metric used for Model checkpoint
            logs: a dictionary to pass to the logger
            progress_bar: a dictionary to pass to the progress bar
            hiddens: when using TBPTT return the hidden states here
        """

        super().__init__(logs, early_stop_on, checkpoint_on, progress_bar, hiddens)

    def reduce_across_batches(self, key: str, value: Tensor, operation: str = 'mean', log: bool = True):
        """
        This metric will be reduced across batches and logged if requested.
        If you use this, there's no need to add the **_epoch_end method

        Args:
            key: name of this metric
            value: value of this metric
            operation: mean, sum
            log: bool
        Returns:

        """
        assert isinstance(value, Tensor), 'the value to reduce must be a torch.Tensor'

        if 'reduce' not in self:
            self.__setitem__('reduce', {})

        reduce = self.__getitem__('reduce')
        options = dict(value=value, operation=operation, log=log)
        reduce[key] = options
        self.__setitem__('reduce', reduce)


if __name__ == '__main__':
    import torch
    result = EvalStepResult()
    result.minimize = torch.tensor(1)