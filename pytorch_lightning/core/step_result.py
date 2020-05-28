from typing import Optional, Dict
from torch import Tensor
from collections import OrderedDict


class Result(OrderedDict):

    def __init__(self,
                 minimize:Optional[Tensor] = None,
                 logs: Optional[Dict] = None,
                 early_stop_on: Tensor = None,
                 checkpoint_on: Tensor = None,
                 progress_bar: Optional[Dict] = None,
                 hiddens: Optional[Tensor] = None):
        """
        Structured output for any *_step (training_step, validation_step, ...).
        We automatically detach anything here for you to avoid holding references to graphs

        Args:
            minimize: Metric to minimize
            logs: dictionary that will be added to your logger(s)
            early_stop_on: Metric for early stopping. If none set, will use minimize by default.
            checkpoint_on: Metric for checkpointing. If none set, will use minimize by default.
            progress_bar: dictionary of values to add to the progress bar
            hiddens: tensor of hiddens to pass to next step when using TBPTT
        """
        super().__init__()

        self.logs = logs
        self.early_stop_on = early_stop_on
        self.checkpoint_on = checkpoint_on
        self.progress_bar = progress_bar
        self.hiddens = hiddens
        self.minimize = minimize

    def to_progress_bar(self, key: str, value: Tensor):
        """
        Adds this key-value pair to the progress bar

        Args:
            key: a string
            value: a tensor

        Returns:

        """
        if 'progress_bar' not in self:
            self.__setitem__('progress_bar', {})

        progress_bar = self.__getitem__('progress_bar')
        progress_bar[key] = value

    def log(self, key: str, value: Tensor):
        """
        Adds this key-value pair to your logger(s)
        Args:
            key: a string
            value: a tensor

        Returns:

        """
        if 'logs' not in self:
            self.__setitem__('logs', {})

        logs = self.__getitem__('logs')
        logs[key] = value

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


if __name__ == '__main__':
    import torch
    result = Result()
    result.minimize = torch.tensor(1)