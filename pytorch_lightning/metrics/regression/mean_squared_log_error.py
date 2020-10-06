import torch
from typing import Any, Callable, Optional, Union

from pytorch_lightning.metrics.metric import Metric


class MeanSquaredLogError(Metric):
    """
    Computes mean squared logarithmic error.

    Args:
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        ddp_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)

    Example:

        >>> from pytorch_lightning.metrics import MeanSquaredLogError
        >>> target = torch.tensor([2.5, 5, 4, 8])
        >>> preds = torch.tensor([3, 5, 2.5, 7])
        >>> mean_squared_log_error = MeanSquaredLogError()
        >>> mean_squared_log_error(preds, target)
        tensor(0.0397)

    """

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        self.add_state("sum_squared_log_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        assert preds.shape == target.shape, \
            'Predictions and targets are expected to have the same shape'
        squared_log_error = torch.pow(torch.log1p(preds) - torch.log1p(target), 2)

        self.sum_squared_log_error += torch.sum(squared_log_error)
        self.total += target.numel()

    def compute(self):
        """
        Compute mean squared logarithmic error over state.
        """
        return self.sum_squared_log_error / self.total
