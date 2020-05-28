from typing import Optional, Dict
from torch import Tensor


class Result:

    def __init__(self, loss: Tensor = None,
                 logs: Optional[Dict] = None,
                 progress_bar_logs: Optional[Dict] = None):
        self.loss = loss
        self.logs = logs
        self.progress_bar_logs = progress_bar_logs

    def log(self, key, value):
        self.logs[key] = value

    def display_in_progress_bar(self, key, value):
        self.progress_bar_logs[key] = value


class TrainStepResult(Result):
    """
    Return this in the training step
    """

    def __init__(self, loss: Tensor = None,
                 logs: Optional[Dict] = None,
                 progress_bar_logs: Optional[Dict] = None):
        """

        Args:
            loss: the loss to minimize (Tensor)
            logs: a dictionary to pass to the logger
            progress_bar_logs: a dictionary to pass to the progress bar
        """
        super().__init__(loss, logs, progress_bar_logs)


class EvalStepResult(Result):
    """
    Return this in the validation step
    """

    def __init__(self, monitor_metric: Tensor = None,
                 logs: Optional[Dict] = None,
                 progress_bar_logs: Optional[Dict] = None):
        """

        Args:
            monitor_metric: the metric to monitor for callbacks (Tensor)
            logs: a dictionary to pass to the logger
            progress_bar_logs: a dictionary to pass to the progress bar
        """

        super().__init__(monitor_metric, logs, progress_bar_logs)
