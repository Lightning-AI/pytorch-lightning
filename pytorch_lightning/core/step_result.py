from typing import Optional, Dict
from torch import Tensor


class Result(Dict):

    def __init__(self,
                 logs: Optional[Dict] = None,
                 early_stop_on: Tensor = None,
                 checkpoint_on: Tensor = None,
                 progress_bar_logs: Optional[Dict] = None,
                 hiddens: Optional[Tensor] = None):
        super().__init__()

        self.logs = {} if logs is None else logs
        self.__setitem__('logs', self.logs)

        self.progress_bar_logs = {} if progress_bar_logs is None else progress_bar_logs
        self.__setitem__('progress_bar_logs', self.progress_bar_logs)

        self.hiddens = hiddens
        self.__setitem__('hiddens', self.hiddens)

        self.checkpoint_on = checkpoint_on
        self.__setitem__('checkpoint_on', checkpoint_on)

        self.early_stop_on = early_stop_on
        self.__setitem__('early_stop_on', early_stop_on)

    def log(self, key, value):
        self.logs[key] = value

    def display_in_progress_bar(self, key, value):
        self.progress_bar_logs[key] = value

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
                 progress_bar_logs: Optional[Dict] = None,
                 hiddens: Optional[Tensor] = None):
        """

        Args:
            minimize: the metric to minimize (usually the loss) (Tensor)
            logs: a dictionary to pass to the logger
            progress_bar_logs: a dictionary to pass to the progress bar
            hiddens: when using TBPTT return the hidden states here
        """
        super().__init__(logs, early_stop_on, checkpoint_on, progress_bar_logs, hiddens)
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

        super().__init__(logs, early_stop_on, checkpoint_on, progress_bar_logs, hiddens)

        # metrics to reduce
        self.__setitem__('reduce', {})

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

        reduce = self.__getitem__('reduce')
        options = dict(value=value, operation=operation, log=log)
        reduce[key] = options
        self.__setitem__('reduce', reduce)

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
    result = EvalStepResult()
    result.minimize = torch.tensor(1)