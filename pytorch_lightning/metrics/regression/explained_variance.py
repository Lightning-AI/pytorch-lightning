import torch
from typing import Any, Callable, Optional, Union

from pytorch_lightning.metrics.metric import Metric


class ExplainedVariance(Metric):
    """
    Computes explained variance.

    Example:

        >>> from pytorch_lightning.metrics import ExplainedVariance
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> explained_variance = ExplainedVariance()
        >>> explained_variance(preds, target)
        tensor(0.9572)


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

        self.add_state("y", default=[], dist_reduce_fx=None)
        self.add_state("y_pred", default=[], dist_reduce_fx=None)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        self.y.append(target)
        self.y_pred.append(preds)

    def compute(self):
        """
        Computes explained variance over state.
        """
        y_true = torch.cat(self.y, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)

        y_diff_avg = torch.mean(y_true - y_pred, dim=0)
        numerator = torch.mean((y_true - y_pred - y_diff_avg) ** 2, dim=0)

        y_true_avg = torch.mean(y_true, dim=0)
        denominator = torch.mean((y_true - y_true_avg) ** 2, dim=0)

        # TODO: multioutput
        return 1.0 - torch.mean(numerator / denominator)
