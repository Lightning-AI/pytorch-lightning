from typing import Any, Optional, Union

import numpy as np

import torch

from pytorch_lightning import _logger as lightning_logger
from pytorch_lightning.metrics.metric import NumpyMetric


class SklearnMetric(NumpyMetric):
    def __init__(self, metric_name: str,
                 reduce_group: Any = torch.distributed.group.WORLD,
                 reduce_op: Any = torch.distributed.ReduceOp.SUM, **kwargs):
        """
        Bridge between PyTorch Lightning and scikit-learn metrics

        .. warning::
            Every metric call will cause a GPU synchronization, which may slow down your code

        .. note::
            The order of targets and predictions may be different from the order typically used in PyTorch

        Args:
            metric_name: the metric name to import anc compute from scikit-learn.metrics
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
            **kwargs: additonal keyword arguments (will be forwarded to metric call)
        """
        super().__init__(name=metric_name, reduce_group=reduce_group,
                         reduce_op=reduce_op)

        self.metric_kwargs = kwargs

        lightning_logger.debug(
            'Every metric call will cause a GPU synchronization, which may slow down your code')

    @property
    def metric_fn(self):
        import sklearn.metrics
        return getattr(sklearn.metrics, self.name)

    def forward(self, *args, **kwargs) -> Union[np.ndarray, int, float]:
        """
        Carries the actual metric computation and therefore co
        Args:
            *args: Positional arguments forwarded to metric call (should be already converted to numpy)
            **kwargs: keyword arguments forwarded to metric call (should be already converted to numpy)

        Returns:
            the metric value (will be converted to tensor by baseclass

        """
        return self.metric_fn(*args, **kwargs)


# metrics : accuracy, auc, average_precision (AP), confusion_matrix, f1, fbeta, hamm, precision, recall, precision_recall_curve, roc, roc_auc, r2, jaccard

class Accuracy(SklearnMetric):
    def __init__(self, normalize: bool = True,
                 reduce_group: Any = torch.distributed.group.WORLD,
                 reduce_op: Any = torch.distributed.ReduceOp.SUM):
        """
        Calculates the Accuracy Score

        .. warning::
            Every metric call will cause a GPU synchronization, which may slow down your code

        Args:
            normalize: If ``False``, return the number of correctly classified samples.
                Otherwise, return the fraction of correctly classified samples.
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__(metric_name='accuracy_score',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op,
                         normalize=normalize)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray,
                sample_weight: Optional[np.ndarray] = None) -> float:
        """
        Computes the accuracy
        Args:
            y_pred: the array containing the predictions (already in categorical form)
            y_true: the array containing the targets (in categorical form)
            sample_weight:

        Returns:
            Accuracy Score


        """
        return super().forward(y_pred=y_pred, y_true=y_true, sample_weight=sample_weight)

class AUC(SklearnMetric):
    def __init__(self, reorder: bool = False,
                 reduce_group: Any = torch.distributed.group.WORLD,
                 reduce_op: Any = torch.distributed.ReduceOp.SUM
                 ):
        """
        Calculates the Area Under the Curve using the trapoezoidal rule

        .. warning::
            Every metric call will cause a GPU synchronization, which may slow down your code

        Args:
            reorder: If ``True``, assume that the curve is ascending in the case of ties, as for an ROC curve.
                If the curve is non-ascending, the result will be wrong.
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """

        super().__init__(metric_name='auc',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op,
                         reorder=reorder)

    def forward(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the AUC
        Args:
            x: x coordinates.
            y: y coordinates.

        Returns:
            AUC calculated with trapezoidal rule

        """
        return super().forward(x=x, y=y)





