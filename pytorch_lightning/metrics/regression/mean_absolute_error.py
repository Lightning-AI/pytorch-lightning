import torch
from typing import Any, Callable, Optional, Union

from pytorch_lightning.metrics.metric import Metric



class MeanAbsoluteError(Metric):
    """
    Computes mean absolute error.

    Example:

        >>> from pytorch_lightning.metrics import MeanAbsoluteError
        >>> target = torch.tensor([3.0, -0.5, 2.0, 7.0])
        >>> preds = torch.tensor([2.5, 0.0, 2.0, 8.0])
        >>> mean_absolute_error = MeanAbsoluteError()
        >>> mean_absolute_error(preds, target)
        tensor(0.5)
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
        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        abs_error = torch.abs(preds - target)
        self.sum_abs_error += torch.sum(abs_error)
        self.total += target.numel()

    def compute(self):
        return self.sum_abs_error / self.total


