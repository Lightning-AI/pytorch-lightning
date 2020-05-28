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
        super().__init__()

        self.logs = logs
        self.early_stop_on = early_stop_on
        self.checkpoint_on = checkpoint_on
        self.progress_bar = progress_bar
        self.hiddens = hiddens
        self.minimize = minimize

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