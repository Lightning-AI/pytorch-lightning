import torch
from typing import Any, Callable, Optional, Union

from pytorch_lightning.metrics.metric import Metric


class MeanSquaredLogError(Metric):
    """
    Computes mean squared logarithmic error.

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
        ddp_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            ddp_sync_on_step=ddp_sync_on_step,
            process_group=process_group,
        )
        self.add_state("sum_squared_log_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        squared_log_error = torch.pow(torch.log1p(preds) - torch.log1p(target), 2)
        self.sum_squared_log_error += torch.sum(squared_log_error)
        self.total += target.numel()

    def compute(self):
        return self.sum_squared_log_error / self.total
