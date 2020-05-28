from typing import Optional, Dict
from torch import Tensor


class Result(Dict):

    def __init__(self,
                 logs: Optional[Dict] = None,
                 progress_bar_logs: Optional[Dict] = None,
                 hiddens: Optional[Tensor] = None):
        super().__init__()

        self.logs = {} if logs is None else logs
        self.progress_bar_logs = {} if progress_bar_logs is None else progress_bar_logs
        self.hiddens = hiddens

    def log(self, key, value):
        self.logs[key] = value

    def display_in_progress_bar(self, key, value):
        self.progress_bar_logs[key] = value

    @property
    def hiddens(self):
        return self._hiddens

    @hiddens.setter
    def hiddens(self, x):
        assert isinstance(x, Tensor), 'hiddens must be a torch.Tensor'

        self._hiddens = x
        self.__setitem__('hiddens', self.hiddens)


class TrainStepResult(Result):
    """
    Return this in the training step
    """

    def __init__(self, loss: Tensor = None,
                 logs: Optional[Dict] = None,
                 progress_bar_logs: Optional[Dict] = None,
                 hiddens: Optional[Tensor] = None):
        """

        Args:
            loss: the loss to minimize (Tensor)
            logs: a dictionary to pass to the logger
            progress_bar_logs: a dictionary to pass to the progress bar
            hiddens: when using TBPTT return the hidden states here
        """
        super().__init__(logs, progress_bar_logs, hiddens)
        self.loss = loss

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, x):
        assert isinstance(x, Tensor), 'loss must be a torch.Tensor'

        self._loss = x
        self.__setitem__('loss', self.loss)


class EvalStepResult(Result):
    """
    Return this in the validation step
    """

    def __init__(self, early_stop_on: Tensor = None,
                 checkpoint_on: Tensor = None,
                 logs: Optional[Dict] = None,
                 progress_bar_logs: Optional[Dict] = None,
                 hiddens: Optional[Tensor] = None):
        """

        Args:
            early_stop_on: the metric used for early Stopping
            checkpoint_on: the metric used for Model checkpoint
            logs: a dictionary to pass to the logger
            progress_bar_logs: a dictionary to pass to the progress bar
            hiddens: when using TBPTT return the hidden states here
        """

        super().__init__(logs, progress_bar_logs, hiddens)
        self.early_stop_on = early_stop_on
        self.checkpoint_on = checkpoint_on

    @property
    def checkpoint_on(self):
        return self._checkpoint_on

    @checkpoint_on.setter
    def checkpoint_on(self, x):
        assert isinstance(x, Tensor), 'checkpoint_on must be a torch.Tensor'

        self._checkpoint_on = x
        self.__setitem__('checkpoint_on', self._checkpoint_on)

    @property
    def early_stop_on(self):
        return self._early_stop_on

    @early_stop_on.setter
    def early_stop_on(self, x):
        assert isinstance(x, Tensor), 'early_stop_on must be a torch.Tensor'

        self._early_stop_on = x
        self.__setitem__('early_stop_on', self._early_stop_on)


if __name__ == '__main__':
    import torch
    result = TrainStepResult()
    result.loss = torch.tensor(1)
    result['loss']